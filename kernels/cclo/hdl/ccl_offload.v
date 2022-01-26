/*******************************************************************************
#  Copyright (C) 2021 Xilinx, Inc
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
# *******************************************************************************/

`timescale 1 ns / 1 ps

module ccl_offload
(
  input ap_clk,
  input ap_rst_n,

`ifdef MB_DEBUG_ENABLE
  input bscan_0_bscanid_en,
  input bscan_0_capture,
  input bscan_0_drck,
  input bscan_0_reset,
  input bscan_0_sel,
  input bscan_0_shift,
  input bscan_0_tck,
  input bscan_0_tdi,
  output bscan_0_tdo,
  input bscan_0_tms,
  input bscan_0_update,
`endif

`ifdef DMA_ENABLE
  output [63:0] m_axi_0_araddr,
  output [1:0] m_axi_0_arburst,
  output [3:0] m_axi_0_arcache,
  output [7:0] m_axi_0_arlen,
  output [2:0] m_axi_0_arprot,
  input m_axi_0_arready,
  output [2:0] m_axi_0_arsize,
  output [3:0] m_axi_0_aruser,
  output m_axi_0_arvalid,
  output [63:0] m_axi_0_awaddr,
  output [1:0] m_axi_0_awburst,
  output [3:0] m_axi_0_awcache,
  output [7:0] m_axi_0_awlen,
  output [2:0] m_axi_0_awprot,
  input m_axi_0_awready,
  output [2:0] m_axi_0_awsize,
  output [3:0] m_axi_0_awuser,
  output m_axi_0_awvalid,
  output m_axi_0_bready,
  input [1:0] m_axi_0_bresp,
  input m_axi_0_bvalid,
  input [511:0] m_axi_0_rdata,
  input m_axi_0_rlast,
  output m_axi_0_rready,
  input [1:0] m_axi_0_rresp,
  input m_axi_0_rvalid,
  output [511:0] m_axi_0_wdata,
  output m_axi_0_wlast,
  input m_axi_0_wready,
  output [63:0] m_axi_0_wstrb,
  output m_axi_0_wvalid,

  output [63:0] m_axi_1_araddr,
  output [1:0] m_axi_1_arburst,
  output [3:0] m_axi_1_arcache,
  output [7:0] m_axi_1_arlen,
  output [2:0] m_axi_1_arprot,
  input m_axi_1_arready,
  output [2:0] m_axi_1_arsize,
  output [3:0] m_axi_1_aruser,
  output m_axi_1_arvalid,
  output [63:0] m_axi_1_awaddr,
  output [1:0] m_axi_1_awburst,
  output [3:0] m_axi_1_awcache,
  output [7:0] m_axi_1_awlen,
  output [2:0] m_axi_1_awprot,
  input m_axi_1_awready,
  output [2:0] m_axi_1_awsize,
  output [3:0] m_axi_1_awuser,
  output m_axi_1_awvalid,
  output m_axi_1_bready,
  input [1:0] m_axi_1_bresp,
  input m_axi_1_bvalid,
  input [511:0] m_axi_1_rdata,
  input m_axi_1_rlast,
  output m_axi_1_rready,
  input [1:0] m_axi_1_rresp,
  input m_axi_1_rvalid,
  output [511:0] m_axi_1_wdata,
  output m_axi_1_wlast,
  input m_axi_1_wready,
  output [63:0] m_axi_1_wstrb,
  output m_axi_1_wvalid,
`endif

`ifdef ARITH_ENABLE
  output [1023:0] m_axis_arith_op_tdata,
  output [127:0] m_axis_arith_op_tkeep,
  output [7:0] m_axis_arith_op_tdest,
  output [0:0] m_axis_arith_op_tlast,
  input [0:0] m_axis_arith_op_tready,
  output [0:0] m_axis_arith_op_tvalid,

  input [511:0] s_axis_arith_res_tdata,
  input [63:0] s_axis_arith_res_tkeep,
  input [0:0] s_axis_arith_res_tlast,
  output [0:0] s_axis_arith_res_tready,
  input [0:0] s_axis_arith_res_tvalid,
`endif

`ifdef STREAM_ENABLE
  output [511:0] m_axis_krnl_tdata,
  output [63:0] m_axis_krnl_tkeep,
  output [0:0] m_axis_krnl_tlast,
  input [0:0] m_axis_krnl_tready,
  output [0:0] m_axis_krnl_tvalid,

  input [511:0] s_axis_krnl_tdata,
  input [63:0] s_axis_krnl_tkeep,
  input [0:0] s_axis_krnl_tlast,
  output [0:0] s_axis_krnl_tready,
  input [0:0] s_axis_krnl_tvalid,
`endif

`ifdef COMPRESSION_ENABLE
  output [511:0] m_axis_compression0_tdata,
  output [63:0] m_axis_compression0_tkeep,
  output [3:0] m_axis_compression0_tdest,
  output [0:0] m_axis_compression0_tlast,
  input [0:0] m_axis_compression0_tready,
  output [0:0] m_axis_compression0_tvalid,

  output [511:0] m_axis_compression1_tdata,
  output [63:0] m_axis_compression1_tkeep,
  output [3:0] m_axis_compression1_tdest,
  output [0:0] m_axis_compression1_tlast,
  input [0:0] m_axis_compression1_tready,
  output [0:0] m_axis_compression1_tvalid,

  output [511:0] m_axis_compression2_tdata,
  output [63:0] m_axis_compression2_tkeep,
  output [3:0] m_axis_compression2_tdest,
  output [0:0] m_axis_compression2_tlast,
  input [0:0] m_axis_compression2_tready,
  output [0:0] m_axis_compression2_tvalid,

  input [511:0] s_axis_compression0_tdata,
  input [63:0] s_axis_compression0_tkeep,
  input [0:0] s_axis_compression0_tlast,
  output [0:0] s_axis_compression0_tready,
  input [0:0] s_axis_compression0_tvalid,

  input [511:0] s_axis_compression1_tdata,
  input [63:0] s_axis_compression1_tkeep,
  input [0:0] s_axis_compression1_tlast,
  output [0:0] s_axis_compression1_tready,
  input [0:0] s_axis_compression1_tvalid,

  input [511:0] s_axis_compression2_tdata,
  input [63:0] s_axis_compression2_tkeep,
  input [0:0] s_axis_compression2_tlast,
  output [0:0] s_axis_compression2_tready,
  input [0:0] s_axis_compression2_tvalid,
`endif

`ifdef TCP_ENABLE
  output [15:0] m_axis_eth_listen_port_tdata,
  output [1:0] m_axis_eth_listen_port_tkeep,
  output [0:0] m_axis_eth_listen_port_tlast,
  input m_axis_eth_listen_port_tready,
  output [1:0] m_axis_eth_listen_port_tstrb,
  output m_axis_eth_listen_port_tvalid,

  output [63:0] m_axis_eth_open_connection_tdata,
  output [7:0] m_axis_eth_open_connection_tkeep,
  output [0:0] m_axis_eth_open_connection_tlast,
  input m_axis_eth_open_connection_tready,
  output [7:0] m_axis_eth_open_connection_tstrb,
  output m_axis_eth_open_connection_tvalid,

  output [31:0] m_axis_eth_read_pkg_tdata,
  output [3:0] m_axis_eth_read_pkg_tkeep,
  output [0:0] m_axis_eth_read_pkg_tlast,
  input m_axis_eth_read_pkg_tready,
  output [3:0] m_axis_eth_read_pkg_tstrb,
  output m_axis_eth_read_pkg_tvalid,

  output [31:0] m_axis_eth_tx_meta_tdata,
  output [3:0] m_axis_eth_tx_meta_tkeep,
  output [0:0] m_axis_eth_tx_meta_tlast,
  input m_axis_eth_tx_meta_tready,
  output [3:0] m_axis_eth_tx_meta_tstrb,
  output m_axis_eth_tx_meta_tvalid,

  input [127:0] s_axis_eth_notification_tdata,
  input [15:0] s_axis_eth_notification_tkeep,
  input [0:0] s_axis_eth_notification_tlast,
  output s_axis_eth_notification_tready,
  input [15:0] s_axis_eth_notification_tstrb,
  input s_axis_eth_notification_tvalid,

  input [127:0] s_axis_eth_open_status_tdata,
  input [15:0] s_axis_eth_open_status_tkeep,
  input [0:0] s_axis_eth_open_status_tlast,
  output s_axis_eth_open_status_tready,
  input [15:0] s_axis_eth_open_status_tstrb,
  input s_axis_eth_open_status_tvalid,

  input [7:0] s_axis_eth_port_status_tdata,
  input [0:0] s_axis_eth_port_status_tkeep,
  input [0:0] s_axis_eth_port_status_tlast,
  output s_axis_eth_port_status_tready,
  input [0:0] s_axis_eth_port_status_tstrb,
  input s_axis_eth_port_status_tvalid,

  input [15:0] s_axis_eth_rx_meta_tdata,
  input [1:0] s_axis_eth_rx_meta_tkeep,
  input [0:0] s_axis_eth_rx_meta_tlast,
  output s_axis_eth_rx_meta_tready,
  input [1:0] s_axis_eth_rx_meta_tstrb,
  input s_axis_eth_rx_meta_tvalid,

  input [63:0] s_axis_eth_tx_status_tdata,
  input [7:0] s_axis_eth_tx_status_tkeep,
  input [0:0] s_axis_eth_tx_status_tlast,
  output s_axis_eth_tx_status_tready,
  input [7:0] s_axis_eth_tx_status_tstrb,
  input s_axis_eth_tx_status_tvalid,
`endif

`ifdef AXI_DATA_ACCESS
  input [63:0] s_axi_data_araddr,
  input [1:0] s_axi_data_arburst,
  input [3:0] s_axi_data_arcache,
  input [3:0] s_axi_data_arid,
  input [7:0] s_axi_data_arlen,
  input [0:0] s_axi_data_arlock,
  input [2:0] s_axi_data_arprot,
  input [3:0] s_axi_data_arqos,
  output [0:0] s_axi_data_arready,
  input [2:0] s_axi_data_arsize,
  input [3:0] s_axi_data_aruser,
  input [0:0] s_axi_data_arvalid,
  input [63:0] s_axi_data_awaddr,
  input [1:0] s_axi_data_awburst,
  input [3:0] s_axi_data_awcache,
  input [3:0] s_axi_data_awid,
  input [7:0] s_axi_data_awlen,
  input [0:0] s_axi_data_awlock,
  input [2:0] s_axi_data_awprot,
  input [3:0] s_axi_data_awqos,
  output [0:0] s_axi_data_awready,
  input [2:0] s_axi_data_awsize,
  input [3:0] s_axi_data_awuser,
  input [0:0] s_axi_data_awvalid,
  output [3:0] s_axi_data_bid,
  input [0:0] s_axi_data_bready,
  output [1:0] s_axi_data_bresp,
  output [0:0] s_axi_data_bvalid,
  output [511:0] s_axi_data_rdata,
  output [3:0] s_axi_data_rid,
  output [0:0] s_axi_data_rlast,
  input [0:0] s_axi_data_rready,
  output [1:0] s_axi_data_rresp,
  output [0:0] s_axi_data_rvalid,
  input [511:0] s_axi_data_wdata,
  input [0:0] s_axi_data_wlast,
  output [0:0] s_axi_data_wready,
  input [63:0] s_axi_data_wstrb,
  input [0:0] s_axi_data_wvalid,
`endif

  output [511:0] m_axis_eth_tx_data_tdata,
  output [7:0] m_axis_eth_tx_data_tdest,
  output [63:0] m_axis_eth_tx_data_tkeep,
  output m_axis_eth_tx_data_tlast,
  input m_axis_eth_tx_data_tready,
  output m_axis_eth_tx_data_tvalid,

  input [511:0] s_axis_eth_rx_data_tdata,
  input [7:0] s_axis_eth_rx_data_tdest,
  input [63:0] s_axis_eth_rx_data_tkeep,
  input s_axis_eth_rx_data_tlast,
  output s_axis_eth_rx_data_tready,
  input s_axis_eth_rx_data_tvalid,

  output [31:0] m_axis_call_ack_tdata,
  output m_axis_call_ack_tlast,
  input m_axis_call_ack_tready,
  output m_axis_call_ack_tvalid,

  input [31:0] s_axis_call_req_tdata,
  input s_axis_call_req_tlast,
  output s_axis_call_req_tready,
  input s_axis_call_req_tvalid,

  input [12:0] s_axi_control_araddr,
  input [2:0] s_axi_control_arprot,
  output s_axi_control_arready,
  input s_axi_control_arvalid,
  input [12:0] s_axi_control_awaddr,
  input [2:0] s_axi_control_awprot,
  output s_axi_control_awready,
  input s_axi_control_awvalid,
  input s_axi_control_bready,
  output [1:0] s_axi_control_bresp,
  output s_axi_control_bvalid,
  output [31:0] s_axi_control_rdata,
  input s_axi_control_rready,
  output [1:0] s_axi_control_rresp,
  output s_axi_control_rvalid,
  input [31:0] s_axi_control_wdata,
  output s_axi_control_wready,
  input [3:0] s_axi_control_wstrb,
  input s_axi_control_wvalid
);

  ccl_offload_bd ccl_offload_bd_i
       (.ap_clk(ap_clk),
        .ap_rst_n(ap_rst_n),

`ifdef MB_DEBUG_ENABLE
        .bscan_0_bscanid_en(bscan_0_bscanid_en),
        .bscan_0_capture(bscan_0_capture),
        .bscan_0_drck(bscan_0_drck),
        .bscan_0_reset(bscan_0_reset),
        .bscan_0_sel(bscan_0_sel),
        .bscan_0_shift(bscan_0_shift),
        .bscan_0_tck(bscan_0_tck),
        .bscan_0_tdi(bscan_0_tdi),
        .bscan_0_tdo(bscan_0_tdo),
        .bscan_0_tms(bscan_0_tms),
        .bscan_0_update(bscan_0_update),
`endif

`ifdef DMA_ENABLE
        .m_axi_0_araddr(m_axi_0_araddr),
        .m_axi_0_arburst(m_axi_0_arburst),
        .m_axi_0_arcache(m_axi_0_arcache),
        .m_axi_0_arlen(m_axi_0_arlen),
        .m_axi_0_arprot(m_axi_0_arprot),
        .m_axi_0_arready(m_axi_0_arready),
        .m_axi_0_arsize(m_axi_0_arsize),
        .m_axi_0_aruser(m_axi_0_aruser),
        .m_axi_0_arvalid(m_axi_0_arvalid),
        .m_axi_0_awaddr(m_axi_0_awaddr),
        .m_axi_0_awburst(m_axi_0_awburst),
        .m_axi_0_awcache(m_axi_0_awcache),
        .m_axi_0_awlen(m_axi_0_awlen),
        .m_axi_0_awprot(m_axi_0_awprot),
        .m_axi_0_awready(m_axi_0_awready),
        .m_axi_0_awsize(m_axi_0_awsize),
        .m_axi_0_awuser(m_axi_0_awuser),
        .m_axi_0_awvalid(m_axi_0_awvalid),
        .m_axi_0_bready(m_axi_0_bready),
        .m_axi_0_bresp(m_axi_0_bresp),
        .m_axi_0_bvalid(m_axi_0_bvalid),
        .m_axi_0_rdata(m_axi_0_rdata),
        .m_axi_0_rlast(m_axi_0_rlast),
        .m_axi_0_rready(m_axi_0_rready),
        .m_axi_0_rresp(m_axi_0_rresp),
        .m_axi_0_rvalid(m_axi_0_rvalid),
        .m_axi_0_wdata(m_axi_0_wdata),
        .m_axi_0_wlast(m_axi_0_wlast),
        .m_axi_0_wready(m_axi_0_wready),
        .m_axi_0_wstrb(m_axi_0_wstrb),
        .m_axi_0_wvalid(m_axi_0_wvalid),
        .m_axi_1_araddr(m_axi_1_araddr),
        .m_axi_1_arburst(m_axi_1_arburst),
        .m_axi_1_arcache(m_axi_1_arcache),
        .m_axi_1_arlen(m_axi_1_arlen),
        .m_axi_1_arprot(m_axi_1_arprot),
        .m_axi_1_arready(m_axi_1_arready),
        .m_axi_1_arsize(m_axi_1_arsize),
        .m_axi_1_aruser(m_axi_1_aruser),
        .m_axi_1_arvalid(m_axi_1_arvalid),
        .m_axi_1_awaddr(m_axi_1_awaddr),
        .m_axi_1_awburst(m_axi_1_awburst),
        .m_axi_1_awcache(m_axi_1_awcache),
        .m_axi_1_awlen(m_axi_1_awlen),
        .m_axi_1_awprot(m_axi_1_awprot),
        .m_axi_1_awready(m_axi_1_awready),
        .m_axi_1_awsize(m_axi_1_awsize),
        .m_axi_1_awuser(m_axi_1_awuser),
        .m_axi_1_awvalid(m_axi_1_awvalid),
        .m_axi_1_bready(m_axi_1_bready),
        .m_axi_1_bresp(m_axi_1_bresp),
        .m_axi_1_bvalid(m_axi_1_bvalid),
        .m_axi_1_rdata(m_axi_1_rdata),
        .m_axi_1_rlast(m_axi_1_rlast),
        .m_axi_1_rready(m_axi_1_rready),
        .m_axi_1_rresp(m_axi_1_rresp),
        .m_axi_1_rvalid(m_axi_1_rvalid),
        .m_axi_1_wdata(m_axi_1_wdata),
        .m_axi_1_wlast(m_axi_1_wlast),
        .m_axi_1_wready(m_axi_1_wready),
        .m_axi_1_wstrb(m_axi_1_wstrb),
        .m_axi_1_wvalid(m_axi_1_wvalid),
`endif

`ifdef ARITH_ENABLE
        .m_axis_arith_op_tdata(m_axis_arith_op_tdata),
        .m_axis_arith_op_tkeep(m_axis_arith_op_tkeep),
        .m_axis_arith_op_tdest(m_axis_arith_op_tdest),
        .m_axis_arith_op_tlast(m_axis_arith_op_tlast),
        .m_axis_arith_op_tready(m_axis_arith_op_tready),
        .m_axis_arith_op_tvalid(m_axis_arith_op_tvalid),

        .s_axis_arith_res_tdata(s_axis_arith_res_tdata),
        .s_axis_arith_res_tkeep(s_axis_arith_res_tkeep),
        .s_axis_arith_res_tlast(s_axis_arith_res_tlast),
        .s_axis_arith_res_tready(s_axis_arith_res_tready),
        .s_axis_arith_res_tvalid(s_axis_arith_res_tvalid),
`endif

`ifdef STREAM_ENABLE
        .m_axis_krnl_tdata(m_axis_krnl_tdata),
        .m_axis_krnl_tkeep(m_axis_krnl_tkeep),
        .m_axis_krnl_tlast(m_axis_krnl_tlast),
        .m_axis_krnl_tready(m_axis_krnl_tready),
        .m_axis_krnl_tvalid(m_axis_krnl_tvalid),

        .s_axis_krnl_tdata(s_axis_krnl_tdata),
        .s_axis_krnl_tkeep(s_axis_krnl_tkeep),
        .s_axis_krnl_tlast(s_axis_krnl_tlast),
        .s_axis_krnl_tready(s_axis_krnl_tready),
        .s_axis_krnl_tvalid(s_axis_krnl_tvalid),
`endif

`ifdef COMPRESSION_ENABLE
        .m_axis_compression0_tdata(m_axis_compression0_tdata),
        .m_axis_compression0_tkeep(m_axis_compression0_tkeep),
        .m_axis_compression0_tdest(m_axis_compression0_tdest),
        .m_axis_compression0_tlast(m_axis_compression0_tlast),
        .m_axis_compression0_tready(m_axis_compression0_tready),
        .m_axis_compression0_tvalid(m_axis_compression0_tvalid),

        .m_axis_compression1_tdata(m_axis_compression1_tdata),
        .m_axis_compression1_tkeep(m_axis_compression1_tkeep),
        .m_axis_compression1_tdest(m_axis_compression1_tdest),
        .m_axis_compression1_tlast(m_axis_compression1_tlast),
        .m_axis_compression1_tready(m_axis_compression1_tready),
        .m_axis_compression1_tvalid(m_axis_compression1_tvalid),
        
        .m_axis_compression2_tdata(m_axis_compression2_tdata),
        .m_axis_compression2_tkeep(m_axis_compression2_tkeep),
        .m_axis_compression2_tdest(m_axis_compression2_tdest),
        .m_axis_compression2_tlast(m_axis_compression2_tlast),
        .m_axis_compression2_tready(m_axis_compression2_tready),
        .m_axis_compression2_tvalid(m_axis_compression2_tvalid),

        .s_axis_compression0_tdata(s_axis_compression0_tdata),
        .s_axis_compression0_tkeep(s_axis_compression0_tkeep),
        .s_axis_compression0_tlast(s_axis_compression0_tlast),
        .s_axis_compression0_tready(s_axis_compression0_tready),
        .s_axis_compression0_tvalid(s_axis_compression0_tvalid),
        
        .s_axis_compression1_tdata(s_axis_compression1_tdata),
        .s_axis_compression1_tkeep(s_axis_compression1_tkeep),
        .s_axis_compression1_tlast(s_axis_compression1_tlast),
        .s_axis_compression1_tready(s_axis_compression1_tready),
        .s_axis_compression1_tvalid(s_axis_compression1_tvalid),
        
        .s_axis_compression2_tdata(s_axis_compression2_tdata),
        .s_axis_compression2_tkeep(s_axis_compression2_tkeep),
        .s_axis_compression2_tlast(s_axis_compression2_tlast),
        .s_axis_compression2_tready(s_axis_compression2_tready),
        .s_axis_compression2_tvalid(s_axis_compression2_tvalid),
`endif

`ifdef TCP_ENABLE
        .m_axis_eth_listen_port_tdata(m_axis_eth_listen_port_tdata),
        .m_axis_eth_listen_port_tkeep(m_axis_eth_listen_port_tkeep),
        .m_axis_eth_listen_port_tlast(m_axis_eth_listen_port_tlast),
        .m_axis_eth_listen_port_tready(m_axis_eth_listen_port_tready),
        .m_axis_eth_listen_port_tstrb(m_axis_eth_listen_port_tstrb),
        .m_axis_eth_listen_port_tvalid(m_axis_eth_listen_port_tvalid),

        .m_axis_eth_open_connection_tdata(m_axis_eth_open_connection_tdata),
        .m_axis_eth_open_connection_tkeep(m_axis_eth_open_connection_tkeep),
        .m_axis_eth_open_connection_tlast(m_axis_eth_open_connection_tlast),
        .m_axis_eth_open_connection_tready(m_axis_eth_open_connection_tready),
        .m_axis_eth_open_connection_tstrb(m_axis_eth_open_connection_tstrb),
        .m_axis_eth_open_connection_tvalid(m_axis_eth_open_connection_tvalid),

        .m_axis_eth_read_pkg_tdata(m_axis_eth_read_pkg_tdata),
        .m_axis_eth_read_pkg_tkeep(m_axis_eth_read_pkg_tkeep),
        .m_axis_eth_read_pkg_tlast(m_axis_eth_read_pkg_tlast),
        .m_axis_eth_read_pkg_tready(m_axis_eth_read_pkg_tready),
        .m_axis_eth_read_pkg_tstrb(m_axis_eth_read_pkg_tstrb),
        .m_axis_eth_read_pkg_tvalid(m_axis_eth_read_pkg_tvalid),

        .m_axis_eth_tx_meta_tdata(m_axis_eth_tx_meta_tdata),
        .m_axis_eth_tx_meta_tkeep(m_axis_eth_tx_meta_tkeep),
        .m_axis_eth_tx_meta_tlast(m_axis_eth_tx_meta_tlast),
        .m_axis_eth_tx_meta_tready(m_axis_eth_tx_meta_tready),
        .m_axis_eth_tx_meta_tstrb(m_axis_eth_tx_meta_tstrb),
        .m_axis_eth_tx_meta_tvalid(m_axis_eth_tx_meta_tvalid),
        
        .s_axis_eth_notification_tdata(s_axis_eth_notification_tdata),
        .s_axis_eth_notification_tkeep(s_axis_eth_notification_tkeep),
        .s_axis_eth_notification_tlast(s_axis_eth_notification_tlast),
        .s_axis_eth_notification_tready(s_axis_eth_notification_tready),
        .s_axis_eth_notification_tstrb(s_axis_eth_notification_tstrb),
        .s_axis_eth_notification_tvalid(s_axis_eth_notification_tvalid),

        .s_axis_eth_open_status_tdata(s_axis_eth_open_status_tdata),
        .s_axis_eth_open_status_tkeep(s_axis_eth_open_status_tkeep),
        .s_axis_eth_open_status_tlast(s_axis_eth_open_status_tlast),
        .s_axis_eth_open_status_tready(s_axis_eth_open_status_tready),
        .s_axis_eth_open_status_tstrb(s_axis_eth_open_status_tstrb),
        .s_axis_eth_open_status_tvalid(s_axis_eth_open_status_tvalid),

        .s_axis_eth_port_status_tdata(s_axis_eth_port_status_tdata),
        .s_axis_eth_port_status_tkeep(s_axis_eth_port_status_tkeep),
        .s_axis_eth_port_status_tlast(s_axis_eth_port_status_tlast),
        .s_axis_eth_port_status_tready(s_axis_eth_port_status_tready),
        .s_axis_eth_port_status_tstrb(s_axis_eth_port_status_tstrb),
        .s_axis_eth_port_status_tvalid(s_axis_eth_port_status_tvalid),

        .s_axis_eth_rx_meta_tdata(s_axis_eth_rx_meta_tdata),
        .s_axis_eth_rx_meta_tkeep(s_axis_eth_rx_meta_tkeep),
        .s_axis_eth_rx_meta_tlast(s_axis_eth_rx_meta_tlast),
        .s_axis_eth_rx_meta_tready(s_axis_eth_rx_meta_tready),
        .s_axis_eth_rx_meta_tstrb(s_axis_eth_rx_meta_tstrb),
        .s_axis_eth_rx_meta_tvalid(s_axis_eth_rx_meta_tvalid),

        .s_axis_eth_tx_status_tdata(s_axis_eth_tx_status_tdata),
        .s_axis_eth_tx_status_tkeep(s_axis_eth_tx_status_tkeep),
        .s_axis_eth_tx_status_tlast(s_axis_eth_tx_status_tlast),
        .s_axis_eth_tx_status_tready(s_axis_eth_tx_status_tready),
        .s_axis_eth_tx_status_tstrb(s_axis_eth_tx_status_tstrb),
        .s_axis_eth_tx_status_tvalid(s_axis_eth_tx_status_tvalid),
`endif

`ifdef AXI_DATA_ACCESS
        .s_axi_data_araddr(s_axi_data_araddr),
        .s_axi_data_arburst(s_axi_data_arburst),
        .s_axi_data_arcache(s_axi_data_arcache),
        .s_axi_data_arid(s_axi_data_arid),
        .s_axi_data_arlen(s_axi_data_arlen),
        .s_axi_data_arlock(s_axi_data_arlock),
        .s_axi_data_arprot(s_axi_data_arprot),
        .s_axi_data_arqos(s_axi_data_arqos),
        .s_axi_data_arready(s_axi_data_arready),
        .s_axi_data_arsize(s_axi_data_arsize),
        .s_axi_data_aruser(s_axi_data_aruser),
        .s_axi_data_arvalid(s_axi_data_arvalid),
        .s_axi_data_awaddr(s_axi_data_awaddr),
        .s_axi_data_awburst(s_axi_data_awburst),
        .s_axi_data_awcache(s_axi_data_awcache),
        .s_axi_data_awid(s_axi_data_awid),
        .s_axi_data_awlen(s_axi_data_awlen),
        .s_axi_data_awlock(s_axi_data_awlock),
        .s_axi_data_awprot(s_axi_data_awprot),
        .s_axi_data_awqos(s_axi_data_awqos),
        .s_axi_data_awready(s_axi_data_awready),
        .s_axi_data_awsize(s_axi_data_awsize),
        .s_axi_data_awuser(s_axi_data_awuser),
        .s_axi_data_awvalid(s_axi_data_awvalid),
        .s_axi_data_bid(s_axi_data_bid),
        .s_axi_data_bready(s_axi_data_bready),
        .s_axi_data_bresp(s_axi_data_bresp),
        .s_axi_data_bvalid(s_axi_data_bvalid),
        .s_axi_data_rdata(s_axi_data_rdata),
        .s_axi_data_rid(s_axi_data_rid),
        .s_axi_data_rlast(s_axi_data_rlast),
        .s_axi_data_rready(s_axi_data_rready),
        .s_axi_data_rresp(s_axi_data_rresp),
        .s_axi_data_rvalid(s_axi_data_rvalid),
        .s_axi_data_wdata(s_axi_data_wdata),
        .s_axi_data_wlast(s_axi_data_wlast),
        .s_axi_data_wready(s_axi_data_wready),
        .s_axi_data_wstrb(s_axi_data_wstrb),
        .s_axi_data_wvalid(s_axi_data_wvalid),
`endif

        .s_axis_eth_rx_data_tdata(s_axis_eth_rx_data_tdata),
        .s_axis_eth_rx_data_tdest(s_axis_eth_rx_data_tdest),
        .s_axis_eth_rx_data_tkeep(s_axis_eth_rx_data_tkeep),
        .s_axis_eth_rx_data_tlast(s_axis_eth_rx_data_tlast),
        .s_axis_eth_rx_data_tready(s_axis_eth_rx_data_tready),
        .s_axis_eth_rx_data_tvalid(s_axis_eth_rx_data_tvalid),

        .m_axis_eth_tx_data_tdata(m_axis_eth_tx_data_tdata),
        .m_axis_eth_tx_data_tdest(m_axis_eth_tx_data_tdest),
        .m_axis_eth_tx_data_tkeep(m_axis_eth_tx_data_tkeep),
        .m_axis_eth_tx_data_tlast(m_axis_eth_tx_data_tlast),
        .m_axis_eth_tx_data_tready(m_axis_eth_tx_data_tready),
        .m_axis_eth_tx_data_tvalid(m_axis_eth_tx_data_tvalid),

        .s_axis_call_req_tvalid(s_axis_call_req_tvalid),
        .s_axis_call_req_tready(s_axis_call_req_tready),
        .s_axis_call_req_tlast(s_axis_call_req_tlast),
        .s_axis_call_req_tdata(s_axis_call_req_tdata),

        .m_axis_call_ack_tvalid(m_axis_call_ack_tvalid),
        .m_axis_call_ack_tready(m_axis_call_ack_tready),
        .m_axis_call_ack_tlast(m_axis_call_ack_tlast),
        .m_axis_call_ack_tdata(m_axis_call_ack_tdata),

        .s_axi_control_araddr(s_axi_control_araddr),
        .s_axi_control_arprot(s_axi_control_arprot),
        .s_axi_control_arready(s_axi_control_arready),
        .s_axi_control_arvalid(s_axi_control_arvalid),
        .s_axi_control_awaddr(s_axi_control_awaddr),
        .s_axi_control_awprot(s_axi_control_awprot),
        .s_axi_control_awready(s_axi_control_awready),
        .s_axi_control_awvalid(s_axi_control_awvalid),
        .s_axi_control_bready(s_axi_control_bready),
        .s_axi_control_bresp(s_axi_control_bresp),
        .s_axi_control_bvalid(s_axi_control_bvalid),
        .s_axi_control_rdata(s_axi_control_rdata),
        .s_axi_control_rready(s_axi_control_rready),
        .s_axi_control_rresp(s_axi_control_rresp),
        .s_axi_control_rvalid(s_axi_control_rvalid),
        .s_axi_control_wdata(s_axi_control_wdata),
        .s_axi_control_wready(s_axi_control_wready),
        .s_axi_control_wstrb(s_axi_control_wstrb),
        .s_axi_control_wvalid(s_axi_control_wvalid)
        );

endmodule
