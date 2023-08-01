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
#include "eth_intf.h"

using namespace std;



void rdma_sq_handler(
	STREAM<rdma_req_t> & rdma_sq,
	STREAM<eth_header> & cmd_in,
	STREAM<eth_header> & cmd_out
)
{
#pragma HLS INTERFACE axis register both port=rdma_sq
#pragma HLS INTERFACE axis register both port=cmd_in
#pragma HLS INTERFACE axis register both port=cmd_out
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS PIPELINE II=1


	unsigned const bytes_per_word = DATA_WIDTH/8;

	static rdma_req_t rdma_req;
	static eth_header cmd_in_word;
	
	enum fsmStateType {WAIT_CMD, RNDZVS_INIT_CMD, RNDZVS_MSG_CMD, RNDZVS_DONE_CMD, EGR_MSG_CMD};
    static fsmStateType  fsmState = WAIT_CMD;


	switch (fsmState)
    {
		case WAIT_CMD:
			if (!STREAM_IS_EMPTY(cmd_in)){
				//read commands from the command stream
				cmd_in_word = STREAM_READ(cmd_in);
				if (cmd_in_word.msg_type == EGR_MSG){
					fsmState = EGR_MSG_CMD;
				} else if (cmd_in_word.msg_type == RNDZVS_MSG) {
					fsmState = RNDZVS_MSG_CMD;
				} else if (cmd_in_word.msg_type == RNDZVS_INIT) {
					fsmState = RNDZVS_INIT_CMD;
				}
			}
			break;
		case RNDZVS_INIT_CMD:
			// issue an RDMA SEND to tell the remote end about the local vaddr and other meta
			rdma_req.opcode = RDMA_SEND;
			rdma_req.qpn = cmd_in_word.dst;
			rdma_req.host = cmd_in_word.host;
			rdma_req.len = bytes_per_word; // only the header len
			rdma_req.vaddr = 0; // not used in SEND Verb
			STREAM_WRITE(rdma_sq, rdma_req);

			// and issue the eth cmd to the rdma packetizer
			cmd_in_word.count = 0; // no message payload, only the header
			cmd_in_word.msg_type = RNDZVS_INIT;
			STREAM_WRITE(cmd_out, cmd_in_word);

			fsmState = WAIT_CMD;	
			break;
		case RNDZVS_MSG_CMD:
			// issue an RDMA WRITE to target remote address
			// and issue the eth cmd to the rdma packetizer
			rdma_req.opcode = RDMA_WRITE;
			rdma_req.qpn = cmd_in_word.dst;
			rdma_req.host = cmd_in_word.host;
			rdma_req.len = cmd_in_word.count; // msg size as no header will be packed into WRITE Verb
			rdma_req.vaddr = cmd_in_word.vaddr;
			STREAM_WRITE(rdma_sq, rdma_req);
			STREAM_WRITE(cmd_out, cmd_in_word);

			fsmState = RNDZVS_DONE_CMD;
			break;
		case RNDZVS_DONE_CMD:
			// issue an RDMA SEND to tell the remote end that the WRITE is done
			rdma_req.opcode = RDMA_SEND;
			rdma_req.qpn = cmd_in_word.dst;
			rdma_req.host = cmd_in_word.host;
			rdma_req.len = bytes_per_word; // only the header len
			rdma_req.vaddr = 0; // not used in SEND Verb
			STREAM_WRITE(rdma_sq, rdma_req);

			// and issue the eth cmd to the rdma packetizer
			cmd_in_word.count = 0; // no message payload, only the header
			cmd_in_word.msg_type = RNDZVS_WR_DONE;
			STREAM_WRITE(cmd_out, cmd_in_word);

			fsmState = WAIT_CMD;	
			break;
		case EGR_MSG_CMD:
			// issue an RDMA SEND
			// and issue the eth cmd to the rdma packetizer
			rdma_req.opcode = RDMA_SEND;
			rdma_req.qpn = cmd_in_word.dst;
			rdma_req.host = cmd_in_word.host;
			rdma_req.len = cmd_in_word.count + bytes_per_word; // msg size plus the header size
			rdma_req.vaddr = 0; // not used in SEND Verb
			STREAM_WRITE(rdma_sq, rdma_req);
			STREAM_WRITE(cmd_out, cmd_in_word);

			fsmState = WAIT_CMD;
			break;  
    }

}