/*******************************************************************************
#  Copyright (C) 2022 Xilinx, Inc
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#
*******************************************************************************/

#include "accl.hpp"
#include <arpa/inet.h>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <mpi.h>
#include <random>
#include <sstream>
#include <string>
#include <tclap/CmdLine.h>
#include <vector>

using namespace ACCL;

// Set the tolerance for compressed datatypes high enough, since we do currently
// not replicate the float32 -> float16 conversion for our reference results
#define FLOAT16RTOL 0.005
#define FLOAT16ATOL 0.05

#define FREQ 250
#define MAX_PKT_SIZE 4096

int mpi_rank, mpi_size;
unsigned failed_tests;
unsigned skipped_tests;

// leave options be for now to avoid any argument parsing issues

struct options_t {
  int start_port;
  unsigned int rxbuf_size;
  unsigned int seg_size;
  unsigned int count;
  unsigned int nruns;
  unsigned int device_index;
  unsigned int num_rxbufmem;
  unsigned int test_mode;
  bool debug;
  bool hardware;
  bool axis3;
  bool udp;
  bool tcp;
  bool rdma;
  unsigned int host;
  unsigned int protoc;
  std::string xclbin;
  std::string fpgaIP;
};

struct timestamp_t {
  uint64_t cmdSeq;
  uint64_t scenario;
  uint64_t len;
  uint64_t comm;
  uint64_t root_src_dst;
  uint64_t function;
  uint64_t msg_tag;
  uint64_t datapath_cfg;
  uint64_t compression_flags;
  uint64_t stream_flags;
  uint64_t addra_l;
  uint64_t addra_h;
  uint64_t addrb_l;
  uint64_t addrb_h;
  uint64_t addrc_l;
  uint64_t addrc_h;
  uint64_t cmdTimestamp;
  uint64_t cmdEnd;
  uint64_t stsSeq;
  uint64_t sts;
  uint64_t stsTimestamp;
  uint64_t stsEnd;
};

//******************************
//**  XCC Operations          **
//******************************
// Housekeeping
#define ACCL_CONFIG 0
// Primitives
#define ACCL_COPY 1
#define ACCL_COMBINE 2
#define ACCL_SEND 3
#define ACCL_RECV 4
// Collectives
#define ACCL_BCAST 5
#define ACCL_SCATTER 6
#define ACCL_GATHER 7
#define ACCL_REDUCE 8
#define ACCL_ALLGATHER 9
#define ACCL_ALLREDUCE 10
#define ACCL_REDUCE_SCATTER 11
#define ACCL_BARRIER 12
#define ACCL_ALLTOALL 13
#define ACCL_NOP 14
#define ACCL_FINN 15

// ACCL_CONFIG SUBFUNCTIONS
#define HOUSEKEEP_SWRST 0
#define HOUSEKEEP_PKTEN 1
#define HOUSEKEEP_TIMEOUT 2
#define HOUSEKEEP_OPEN_PORT 3
#define HOUSEKEEP_OPEN_CON 4
#define HOUSEKEEP_SET_STACK_TYPE 5
#define HOUSEKEEP_SET_MAX_SEGMENT_SIZE 6
#define HOUSEKEEP_CLOSE_CON 7

std::string format_log(std::string collective, options_t options, double time,
                       double tput) {
  std::string host_str;
  std::string protoc_str;
  std::string stack_str;
  if (options.host == 1) {
    host_str = "host";
  } else {
    host_str = "device";
  }
  if (options.protoc == 0) {
    protoc_str = "eager";
  } else if (options.protoc == 1) {
    protoc_str = "rndzvs";
  }
  if (options.tcp) {
    stack_str = "tcp";
  } else if (options.rdma) {
    stack_str = "rdma";
  }
  std::string log_str =
      collective + "," + std::to_string(mpi_size) + "," +
      std::to_string(mpi_rank) + "," + std::to_string(options.num_rxbufmem) +
      "," + std::to_string(options.count * sizeof(float)) + "," +
      std::to_string(options.rxbuf_size) + "," +
      std::to_string(options.rxbuf_size) + "," + std::to_string(MAX_PKT_SIZE) +
      "," + std::to_string(time) + "," + std::to_string(tput) + "," + host_str +
      "," + protoc_str + "," + stack_str;
  return log_str;
}

inline void swap_endianness(uint32_t *ip) {
  uint8_t *ip_bytes = reinterpret_cast<uint8_t *>(ip);
  *ip = (ip_bytes[3] << 0) | (ip_bytes[2] << 8) | (ip_bytes[1] << 16) |
        (ip_bytes[0] << 24);
}

uint32_t _ip_encode(std::string ip) {
  struct sockaddr_in sa;
  inet_pton(AF_INET, ip.c_str(), &(sa.sin_addr));
  swap_endianness(&sa.sin_addr.s_addr);
  return sa.sin_addr.s_addr;
}

std::string ip_decode(uint32_t ip) {
  char buffer[INET_ADDRSTRLEN];
  struct in_addr sa;
  sa.s_addr = ip;
  swap_endianness(&sa.s_addr);
  inet_ntop(AF_INET, &sa, buffer, INET_ADDRSTRLEN);
  return std::string(buffer, INET_ADDRSTRLEN);
}

void test_debug(std::string message, options_t &options) {
  if (options.debug) {
    std::cerr << message << std::endl;
  }
}

void check_usage(int argc, char *argv[]) {}

std::string prepend_process() {
  return "[process " + std::to_string(mpi_rank) + "] ";
}

template <typename T>
bool is_close(T a, T b, double rtol = 1e-5, double atol = 1e-8) {
  // std::cout << abs(a - b) << " <= " << (atol + rtol * abs(b)) << "? " <<
  // (abs(a - b) <= (atol + rtol * abs(b))) << std::endl;
  return abs(a - b) <= (atol + rtol * abs(b));
}

template <typename T> static void random_array(T *data, size_t count) {
  std::uniform_real_distribution<T> distribution(-1000, 1000);
  std::mt19937 engine;
  auto generator = std::bind(distribution, engine);
  for (size_t i = 0; i < count; ++i) {
    data[i] = generator();
  }
}

template <typename T> std::unique_ptr<T> random_array(size_t count) {
  std::unique_ptr<T> data(new T[count]);
  random_array(data.get(), count);
  return data;
}

options_t parse_options(int argc, char *argv[]) {
  try {
    TCLAP::CmdLine cmd("Test ACCL C++ driver");
    TCLAP::ValueArg<unsigned int> nruns_arg("n", "nruns",
                                            "How many times to run each test",
                                            false, 1, "positive integer");
    cmd.add(nruns_arg);
    TCLAP::ValueArg<uint16_t> start_port_arg(
        "s", "start-port", "Start of range of ports usable for sim", false,
        5005, "positive integer");
    cmd.add(start_port_arg);
    TCLAP::ValueArg<uint32_t> count_arg("c", "count",
                                        "How many element per buffer", false,
                                        16, "positive integer");
    cmd.add(count_arg);
    TCLAP::ValueArg<uint16_t> bufsize_arg("b", "rxbuf-size",
                                          "How many KB per RX buffer", false,
                                          4096, "positive integer");
    cmd.add(bufsize_arg);
    TCLAP::ValueArg<uint32_t> seg_arg("g", "max_segment_size",
                                      "Maximum segmentation size in KB (should "
                                      "be samller than Max DMA transaction)",
                                      false, 4096, "positive integer");
    cmd.add(seg_arg);
    TCLAP::ValueArg<uint16_t> num_rxbufmem_arg(
        "m", "num_rxbufmem", "Number of memory banks used for rxbuf", false, 2,
        "positive integer");
    cmd.add(num_rxbufmem_arg);
    TCLAP::ValueArg<uint16_t> test_mode_arg(
        "y", "test_mode", "Test mode, by default run all the collective tests",
        false, 0, "integer");
    cmd.add(test_mode_arg);
    TCLAP::ValueArg<uint16_t> host_arg("z", "host_buffer",
                                       "Enable host buffer mode with 1", false,
                                       0, "integer");
    cmd.add(host_arg);
    TCLAP::ValueArg<uint16_t> protoc_arg(
        "p", "protocol", "Eager Protocol with 0 and Rendezvous with 1", false,
        0, "integer");
    cmd.add(protoc_arg);
    TCLAP::SwitchArg debug_arg("d", "debug", "Enable debug mode", cmd, false);
    TCLAP::SwitchArg hardware_arg("f", "hardware", "enable hardware mode", cmd,
                                  false);
    TCLAP::SwitchArg axis3_arg("a", "axis3", "Use axis3 hardware setup", cmd,
                               false);
    TCLAP::SwitchArg udp_arg("u", "udp", "Use UDP hardware setup", cmd, false);
    TCLAP::SwitchArg tcp_arg("t", "tcp", "Use TCP hardware setup", cmd, false);
    TCLAP::SwitchArg rdma_arg("r", "rdma", "Use RDMA hardware setup", cmd,
                              false);
    TCLAP::SwitchArg userkernel_arg(
        "k", "userkernel", "Enable user kernel(by default vadd kernel)", cmd,
        false);
    TCLAP::ValueArg<std::string> xclbin_arg(
        "x", "xclbin", "xclbin of accl driver if hardware mode is used", false,
        "accl.xclbin", "file");
    cmd.add(xclbin_arg);
    TCLAP::ValueArg<std::string> fpgaIP_arg(
        "l", "ipList", "ip list of FPGAs if hardware mode is used", false,
        "fpga", "file");
    cmd.add(fpgaIP_arg);
    TCLAP::ValueArg<uint16_t> device_index_arg(
        "i", "device-index", "device index of FPGA if hardware mode is used",
        false, 0, "positive integer");
    cmd.add(device_index_arg);
    cmd.parse(argc, argv);
    if (hardware_arg.getValue()) {
      if (axis3_arg.getValue()) {
        if (udp_arg.getValue() || tcp_arg.getValue() || rdma_arg.getValue()) {
          throw std::runtime_error("When using hardware axis3 mode, tcp or "
                                   "rdma or udp can not be used.");
        }
        std::cout << "Hardware axis3 mode" << std::endl;
      }
      if (udp_arg.getValue()) {
        if (axis3_arg.getValue() || tcp_arg.getValue() || rdma_arg.getValue()) {
          throw std::runtime_error("When using hardware udp mode, tcp or rdma "
                                   "or axis3 can not be used.");
        }
        std::cout << "Hardware udp mode" << std::endl;
      }
      if (tcp_arg.getValue()) {
        if (axis3_arg.getValue() || udp_arg.getValue() || rdma_arg.getValue()) {
          throw std::runtime_error("When using hardware tcp mode, udp or rdma "
                                   "or axis3 can not be used.");
        }
        std::cout << "Hardware tcp mode" << std::endl;
      }
      if (rdma_arg.getValue()) {
        if (axis3_arg.getValue() || udp_arg.getValue() || tcp_arg.getValue()) {
          throw std::runtime_error("When using hardware rdma mode, udp or tcp "
                                   "or axis3 can not be used.");
        }
        std::cout << "Hardware rdma mode" << std::endl;
      }
      if ((axis3_arg.getValue() || udp_arg.getValue() || tcp_arg.getValue() ||
           rdma_arg.getValue()) == false) {
        throw std::runtime_error(
            "When using hardware, specify either axis3 or tcp or"
            "udp or rdma mode.");
      }
    }

    options_t opts;
    opts.start_port = start_port_arg.getValue();
    opts.count = count_arg.getValue();
    opts.rxbuf_size = bufsize_arg.getValue() * 1024; // convert to bytes
    opts.seg_size = seg_arg.getValue() * 1024;       // convert to bytes
    opts.num_rxbufmem = num_rxbufmem_arg.getValue();
    opts.nruns = nruns_arg.getValue();
    opts.debug = debug_arg.getValue();
    opts.host = host_arg.getValue();
    opts.hardware = hardware_arg.getValue();
    opts.axis3 = axis3_arg.getValue();
    opts.udp = udp_arg.getValue();
    opts.tcp = tcp_arg.getValue();
    opts.rdma = rdma_arg.getValue();
    opts.test_mode = test_mode_arg.getValue();
    opts.device_index = device_index_arg.getValue();
    opts.xclbin = xclbin_arg.getValue();
    opts.fpgaIP = fpgaIP_arg.getValue();
    opts.protoc = protoc_arg.getValue();

    std::cout << "count:" << opts.count << " rxbuf_size:" << opts.rxbuf_size
              << " seg_size:" << opts.seg_size
              << " num_rxbufmem:" << opts.num_rxbufmem << std::endl;
    return opts;
  } catch (std::exception &e) {
    if (mpi_rank == 0) {
      std::cout << "Error: " << e.what() << std::endl;
    }

    MPI_Finalize();
    exit(1);
  }
}

void exchange_qp(unsigned int master_rank, unsigned int slave_rank,
                 unsigned int local_rank,
                 std::vector<fpga::ibvQpConn *> &ibvQpConn_vec,
                 std::vector<rank_t> &ranks) {

  if (local_rank == master_rank) {
    std::cout << "Local rank " << local_rank
              << " sending local QP to remote rank " << slave_rank << std::endl;
    // Send the local queue pair information to the slave rank
    MPI_Send(&(ibvQpConn_vec[slave_rank]->getQpairStruct()->local),
             sizeof(fpga::ibvQ), MPI_CHAR, slave_rank, 0, MPI_COMM_WORLD);
  } else if (local_rank == slave_rank) {
    std::cout << "Local rank " << local_rank
              << " receiving remote QP from remote rank " << master_rank
              << std::endl;
    // Receive the queue pair information from the master rank
    fpga::ibvQ received_q;
    MPI_Recv(&received_q, sizeof(fpga::ibvQ), MPI_CHAR, master_rank, 0,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Copy the received data to the remote queue pair
    ibvQpConn_vec[master_rank]->getQpairStruct()->remote = received_q;
  }

  // Synchronize after the first exchange to avoid race conditions
  MPI_Barrier(MPI_COMM_WORLD);

  if (local_rank == slave_rank) {
    std::cout << "Local rank " << local_rank
              << " sending local QP to remote rank " << master_rank
              << std::endl;
    // Send the local queue pair information to the master rank
    MPI_Send(&(ibvQpConn_vec[master_rank]->getQpairStruct()->local),
             sizeof(fpga::ibvQ), MPI_CHAR, master_rank, 0, MPI_COMM_WORLD);
  } else if (local_rank == master_rank) {
    std::cout << "Local rank " << local_rank
              << " receiving remote QP from remote rank " << slave_rank
              << std::endl;
    // Receive the queue pair information from the slave rank
    fpga::ibvQ received_q;
    MPI_Recv(&received_q, sizeof(fpga::ibvQ), MPI_CHAR, slave_rank, 0,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Copy the received data to the remote queue pair
    ibvQpConn_vec[slave_rank]->getQpairStruct()->remote = received_q;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // write established connection to hardware and perform arp lookup
  if (local_rank == master_rank) {
    int connection =
        (ibvQpConn_vec[slave_rank]->getQpairStruct()->local.qpn & 0xFFFF) |
        ((ibvQpConn_vec[slave_rank]->getQpairStruct()->remote.qpn & 0xFFFF)
         << 16);
    ibvQpConn_vec[slave_rank]->getQpairStruct()->print();
    ibvQpConn_vec[slave_rank]->setConnection(connection);
    ibvQpConn_vec[slave_rank]->writeContext(ranks[slave_rank].port);
    ibvQpConn_vec[slave_rank]->doArpLookup();
    ranks[slave_rank].session_id =
        ibvQpConn_vec[slave_rank]->getQpairStruct()->local.qpn;
  } else if (local_rank == slave_rank) {
    int connection =
        (ibvQpConn_vec[master_rank]->getQpairStruct()->local.qpn & 0xFFFF) |
        ((ibvQpConn_vec[master_rank]->getQpairStruct()->remote.qpn & 0xFFFF)
         << 16);
    ibvQpConn_vec[master_rank]->getQpairStruct()->print();
    ibvQpConn_vec[master_rank]->setConnection(connection);
    ibvQpConn_vec[master_rank]->writeContext(ranks[master_rank].port);
    ibvQpConn_vec[master_rank]->doArpLookup();
    ranks[master_rank].session_id =
        ibvQpConn_vec[master_rank]->getQpairStruct()->local.qpn;
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

void configure_cyt_rdma(std::vector<rank_t> &ranks, int local_rank,
                        ACCL::CoyoteDevice *device) {

  std::cout << "Initializing QP connections..." << std::endl;
  // create queue pair connections
  std::vector<fpga::ibvQpConn *> ibvQpConn_vec;
  // create single page dummy memory space for each qp
  uint32_t n_pages = 1;
  for (int i = 0; i < ranks.size(); i++) {
    fpga::ibvQpConn *qpConn = new fpga::ibvQpConn(
        device->coyote_qProc_vec[i], ranks[local_rank].ip, n_pages);
    ibvQpConn_vec.push_back(qpConn);
    // qpConn->getQpairStruct()->print();
  }

  std::cout << "Exchanging QP..." << std::endl;
  for (int i = 0; i < ranks.size(); i++) {
    for (int j = i + 1; j < ranks.size(); j++) {
      exchange_qp(i, j, local_rank, ibvQpConn_vec, ranks);
    }
  }
}

void configure_cyt_tcp(std::vector<rank_t> &ranks, int local_rank,
                       ACCL::CoyoteDevice *device) {
  std::cout << "Configuring Coyote TCP..." << std::endl;
  // arp lookup
  for (int i = 0; i < ranks.size(); i++) {
    if (local_rank != i) {
      device->get_device()->doArpLookup(_ip_encode(ranks[i].ip));
    }
  }

  // open port
  for (int i = 0; i < ranks.size(); i++) {
    uint32_t dstPort = ranks[i].port;
    bool open_port_status = device->get_device()->tcpOpenPort(dstPort);
  }

  std::this_thread::sleep_for(10ms);

  // open con
  for (int i = 0; i < ranks.size(); i++) {
    uint32_t dstPort = ranks[i].port;
    uint32_t dstIp = _ip_encode(ranks[i].ip);
    uint32_t dstRank = i;
    uint32_t session = 0;
    if (local_rank != dstRank) {
      bool success = device->get_device()->tcpOpenCon(dstIp, dstPort, &session);
      ranks[i].session_id = session;
    }
  }
}

void test_finn(ACCL::ACCL &accl, options_t &options,
               ACCL::CoyoteDevice *coyotedevice) {
  std::cout << "Start FINN test..." << std::endl << std::flush;

  const uint64_t max_size = 1ULL << 17ULL;
  uint32_t n_pages = ((max_size + pageSize - 1) / pageSize);

  try {

    const auto nbBatches = 1ULL;
    const auto batchSize = 16ULL;
    constexpr auto FINN_CTRL_OFFSET = 0x8000;

    // Configuring accl out node
    if (mpi_rank < mpi_size - 1) {
      uint64_t accl_node_offset;
      if (mpi_rank == 0) {
        accl_node_offset = 0x400;
      } else if (mpi_rank == 1) {
        accl_node_offset = 0x420;
      }

      std::cout << "Configuring accl out for rank " << mpi_rank << std::endl;

      coyotedevice->write((FINN_CTRL_OFFSET + 0x10) >> 1,
                          accl.get_communicator_addr());
      coyotedevice->write((FINN_CTRL_OFFSET + 0x18) >> 1,
                          accl_node_offset + 0x10);
      coyotedevice->write((FINN_CTRL_OFFSET + 0x0) >> 1, 1);
      while (!(coyotedevice->read((FINN_CTRL_OFFSET + 0x0) >> 1) & 2))
        ;

      coyotedevice->write(
          (FINN_CTRL_OFFSET + 0x10) >> 1,
          accl.get_arithmetic_config_addr({dataType::int32, dataType::int32}));
      coyotedevice->write((FINN_CTRL_OFFSET + 0x18) >> 1,
                          accl_node_offset + 0x18);
      coyotedevice->write((FINN_CTRL_OFFSET + 0x0) >> 1, 1);
      while (!(coyotedevice->read((FINN_CTRL_OFFSET + 0x0) >> 1) & 2))
        ;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Configuring tlast
    if (mpi_rank == mpi_size - 1) {

      std::cout << "Configuring TLast for rank " << mpi_rank << std::endl;

      constexpr auto TLAST_OFF = 0x120;
      coyotedevice->write((FINN_CTRL_OFFSET + 0x10) >> 1, nbBatches);
      coyotedevice->write((FINN_CTRL_OFFSET + 0x18) >> 1, TLAST_OFF + 0x10);
      coyotedevice->write((FINN_CTRL_OFFSET + 0x0) >> 1, 1);
      while (!(coyotedevice->read((FINN_CTRL_OFFSET + 0x0) >> 1) & 2))
        ;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (/*mpi_rank == 0 || mpi_rank == (mpi_size - 1)*/ true) {
      int8_t *hMem = nullptr;
      const auto inferenceDataSize2D = nbBatches * batchSize;
      if (mpi_rank == 0 || mpi_rank == (mpi_size - 1)) {
        hMem = static_cast<int8_t *>(
            coyotedevice->coyote_proc->getMem({CoyoteAlloc::REG_4K, n_pages}));
        if (!hMem) {
          std::cout << "Unable to allocate mem" << std::endl;
          return;
        }

        memset(hMem, 0, max_size);

        for (uint64_t i = 0; i < nbBatches; ++i) {
          for (uint64_t j = 0; j < batchSize; ++j) {
            hMem[i * batchSize + j] = -1;
          }
        }
      }
      constexpr auto NB_EXPERIMENTS = 100;
      csInvoke invokeRead = {.oper = CoyoteOper::READ,
                             .addr = &hMem[0],
                             .len = batchSize,
                             .dest = 3};
      csInvoke invokeWrite = {.oper = CoyoteOper::WRITE,
                              .addr = &hMem[inferenceDataSize2D],
                              .len = nbBatches,
                              .dest = 3};
      std::ofstream outfile;
      outfile.open("results.txt"); // append instead of overwrite
      for (int j = 0; j < NB_EXPERIMENTS; ++j) {
        const auto begin = std::chrono::high_resolution_clock::now();
        if (mpi_rank == 0) {
          for (int i = 0; i < NB_EXPERIMENTS; ++i) {
            coyotedevice->coyote_proc->invoke(invokeRead);
          }
        } else if (mpi_rank == mpi_size - 1) {
          for (int i = 0; i < NB_EXPERIMENTS; ++i) {
            coyotedevice->coyote_proc->invoke(invokeWrite);
          }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        const auto end = std::chrono::high_resolution_clock::now();
        const auto diffNs =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
                .count();
        outfile << diffNs << '\n';
      }
    }
  } catch (const std::runtime_error &e) {
    std::cout << "Exception: " << e.what() << std::endl;
  }
  // The middle nodes have to wait for the first and the last node to finish
  // before exiting and cleaning up.
  MPI_Barrier(MPI_COMM_WORLD);
}

void test_accl_base(options_t options) {
  std::cout << "Testing ACCL base functionality..." << std::endl;

  // initialize ACCL
  std::vector<ACCL::rank_t> ranks;
  int local_rank = mpi_rank;
  failed_tests = 0;
  // construct CoyoteDevice out here already, since it is necessary for creating
  // buffers before the ACCL instance exists.
  ACCL::CoyoteDevice *device;

  // load ip addresses for targets
  std::ifstream myfile;
  myfile.open(options.fpgaIP);
  if (!myfile.is_open()) {
    perror("Error open fpgaIP file");
    exit(EXIT_FAILURE);
  }
  std::vector<std::string> ipList;
  for (int i = 0; i < mpi_size; ++i) {
    std::string ip;
    if (options.hardware && !options.axis3) {
      ip = "10.10.10." + std::to_string(i);
      getline(myfile, ip);
      std::cout << ip << std::endl;
      ipList.push_back(ip);
    } else {
      ip = "127.0.0.1";
    }

    if (options.hardware && options.rdma) {
      rank_t new_rank = {ip, options.start_port, i, options.rxbuf_size};
      ranks.emplace_back(new_rank);
    } else {
      rank_t new_rank = {ip, options.start_port + i, 0, options.rxbuf_size};
      ranks.emplace_back(new_rank);
    }
  }

  std::unique_ptr<ACCL::ACCL> accl;

  MPI_Barrier(MPI_COMM_WORLD);

  if (options.tcp) {
    device = new ACCL::CoyoteDevice();
    configure_cyt_tcp(ranks, local_rank, device);
  } else if (options.rdma) {
    device = new ACCL::CoyoteDevice(mpi_size);
    configure_cyt_rdma(ranks, local_rank, device);
  }

  if (options.hardware) {
    if (options.udp) {
      debug("ERROR: we don't support UDP for now!!!");
      exit(1);
    } else if (options.tcp || options.rdma) {
      uint localFPGAIP = _ip_encode(ipList[mpi_rank]);
      std::cout << "rank: " << mpi_rank << " FPGA IP: " << std::hex
                << localFPGAIP << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (options.protoc == 0) {
      std::cout << "Eager Protocol" << std::endl;
      accl = std::make_unique<ACCL::ACCL>(device, ranks, mpi_rank, mpi_size + 2,
                                          options.rxbuf_size, options.seg_size,
                                          4096 * 1024 * 2);
    } else if (options.protoc == 1) {
      std::cout << "Rendezvous Protocol" << std::endl;
      accl = std::make_unique<ACCL::ACCL>(device, ranks, mpi_rank, mpi_size, 64,
                                          64, options.seg_size);
    }

    debug(accl->dump_communicator());

    MPI_Barrier(MPI_COMM_WORLD);

  } else {
    debug("unsupported situation!!!");
    exit(1);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  std::cerr << "Rank " << mpi_rank << " passed last barrier before test!"
            << std::endl
            << std::flush;

  MPI_Barrier(MPI_COMM_WORLD);
  if (options.test_mode == ACCL_FINN) {
    debug(accl->dump_eager_rx_buffers(false));
    MPI_Barrier(MPI_COMM_WORLD);
    test_finn(*accl, options, device);
    debug(accl->dump_communicator());
    debug(accl->dump_eager_rx_buffers(false));
  }

  if (failed_tests == 0) {
    std::cout << "\nACCL base functionality test completed successfully!\n"
              << std::endl;
  } else {
    std::cout << "\nERROR: ACCL base functionality test failed!\n" << std::endl;
  }
}

template <typename T> struct aligned_allocator {
  using value_type = T;
  T *allocate(std::size_t num) {
    void *ptr = nullptr;
    if (posix_memalign(&ptr, 4096, num * sizeof(T)))
      throw std::bad_alloc();
    return reinterpret_cast<T *>(ptr);
  }
  void deallocate(T *p, std::size_t num) { free(p); }
};

int main(int argc, char *argv[]) {
  std::cout << "Argumnents: ";
  for (int i = 0; i < argc; i++)
    std::cout << "'" << argv[i] << "' ";
  std::cout << std::endl;
  std::cout << "Running ACCL test in coyote..." << std::endl;
  std::cout << "Initializing MPI..." << std::endl;
  MPI_Init(&argc, &argv);

  std::cout << "Reading MPI rank and size values..." << std::endl;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  std::cout << "Parsing options" << std::endl;
  options_t options = parse_options(argc, argv);

  std::cout << "Getting MPI Processor name..." << std::endl;
  int len;
  char name[MPI_MAX_PROCESSOR_NAME];
  MPI_Get_processor_name(name, &len);

  std::ostringstream stream;
  stream << prepend_process() << "rank " << mpi_rank << " size " << mpi_size
         << " " << name << std::endl;
  std::cout << stream.str();

  MPI_Barrier(MPI_COMM_WORLD);

  test_accl_base(options);

  MPI_Barrier(MPI_COMM_WORLD);

  std::cout << "Finalizing MPI..." << std::endl;
  MPI_Finalize();
  std::cout << "Done. Terminating..." << std::endl;
  return 0;
}

