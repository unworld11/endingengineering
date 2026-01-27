set moduleName FracNet_T
set isTopModule 1
set isTaskLevelControl 1
set isCombinational 0
set isDatapathOnly 0
set isFreeRunPipelineModule 0
set isPipelined 0
set pipeline_type none
set FunctionProtocol ap_ctrl_hs
set isOneStateSeq 0
set ProfileFlag 0
set StallSigGenFlag 0
set isEnableWaveformDebug 1
set C_modelName {FracNet_T}
set C_modelType { void 0 }
set C_modelArgList {
	{ IMG int 64 regular {axi_master 0}  }
	{ RESULT float 32 regular {axi_master 1}  }
	{ image_V int 32 regular {axi_slave 0}  }
	{ output_r int 32 regular {axi_slave 0}  }
}
set C_modelArgMapList {[ 
	{ "Name" : "IMG", "interface" : "axi_master", "bitwidth" : 64, "direction" : "READONLY", "bitSlice":[{"low":0,"up":63,"cElement": [{"cName": "image.V","cData": "uint64","bit_use": { "low": 0,"up": 63},"offset": { "type": "dynamic","port_name": "image_V","bundle": "AXILiteS"},"direction": "READONLY","cArray": [{"low" : 0,"up" : 2,"step" : 1},{"low" : 0,"up" : 31,"step" : 1},{"low" : 0,"up" : 31,"step" : 1}]}]}]} , 
 	{ "Name" : "RESULT", "interface" : "axi_master", "bitwidth" : 32, "direction" : "WRITEONLY", "bitSlice":[{"low":0,"up":31,"cElement": [{"cName": "output","cData": "float","bit_use": { "low": 0,"up": 31},"offset": { "type": "dynamic","port_name": "output_r","bundle": "AXILiteS"},"direction": "WRITEONLY","cArray": [{"low" : 0,"up" : 9,"step" : 1}]}]}]} , 
 	{ "Name" : "image_V", "interface" : "axi_slave", "bundle":"CTRL","type":"ap_none","bitwidth" : 32, "direction" : "READONLY", "offset" : {"in":16}, "offset_end" : {"in":23}} , 
 	{ "Name" : "output_r", "interface" : "axi_slave", "bundle":"CTRL","type":"ap_none","bitwidth" : 32, "direction" : "READONLY", "offset" : {"in":24}, "offset_end" : {"in":31}} ]}
# RTL Port declarations: 
set portNum 110
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst_n sc_in sc_logic 1 reset -1 active_low_sync } 
	{ m_axi_IMG_AWVALID sc_out sc_logic 1 signal 0 } 
	{ m_axi_IMG_AWREADY sc_in sc_logic 1 signal 0 } 
	{ m_axi_IMG_AWADDR sc_out sc_lv 32 signal 0 } 
	{ m_axi_IMG_AWID sc_out sc_lv 1 signal 0 } 
	{ m_axi_IMG_AWLEN sc_out sc_lv 8 signal 0 } 
	{ m_axi_IMG_AWSIZE sc_out sc_lv 3 signal 0 } 
	{ m_axi_IMG_AWBURST sc_out sc_lv 2 signal 0 } 
	{ m_axi_IMG_AWLOCK sc_out sc_lv 2 signal 0 } 
	{ m_axi_IMG_AWCACHE sc_out sc_lv 4 signal 0 } 
	{ m_axi_IMG_AWPROT sc_out sc_lv 3 signal 0 } 
	{ m_axi_IMG_AWQOS sc_out sc_lv 4 signal 0 } 
	{ m_axi_IMG_AWREGION sc_out sc_lv 4 signal 0 } 
	{ m_axi_IMG_AWUSER sc_out sc_lv 1 signal 0 } 
	{ m_axi_IMG_WVALID sc_out sc_logic 1 signal 0 } 
	{ m_axi_IMG_WREADY sc_in sc_logic 1 signal 0 } 
	{ m_axi_IMG_WDATA sc_out sc_lv 64 signal 0 } 
	{ m_axi_IMG_WSTRB sc_out sc_lv 8 signal 0 } 
	{ m_axi_IMG_WLAST sc_out sc_logic 1 signal 0 } 
	{ m_axi_IMG_WID sc_out sc_lv 1 signal 0 } 
	{ m_axi_IMG_WUSER sc_out sc_lv 1 signal 0 } 
	{ m_axi_IMG_ARVALID sc_out sc_logic 1 signal 0 } 
	{ m_axi_IMG_ARREADY sc_in sc_logic 1 signal 0 } 
	{ m_axi_IMG_ARADDR sc_out sc_lv 32 signal 0 } 
	{ m_axi_IMG_ARID sc_out sc_lv 1 signal 0 } 
	{ m_axi_IMG_ARLEN sc_out sc_lv 8 signal 0 } 
	{ m_axi_IMG_ARSIZE sc_out sc_lv 3 signal 0 } 
	{ m_axi_IMG_ARBURST sc_out sc_lv 2 signal 0 } 
	{ m_axi_IMG_ARLOCK sc_out sc_lv 2 signal 0 } 
	{ m_axi_IMG_ARCACHE sc_out sc_lv 4 signal 0 } 
	{ m_axi_IMG_ARPROT sc_out sc_lv 3 signal 0 } 
	{ m_axi_IMG_ARQOS sc_out sc_lv 4 signal 0 } 
	{ m_axi_IMG_ARREGION sc_out sc_lv 4 signal 0 } 
	{ m_axi_IMG_ARUSER sc_out sc_lv 1 signal 0 } 
	{ m_axi_IMG_RVALID sc_in sc_logic 1 signal 0 } 
	{ m_axi_IMG_RREADY sc_out sc_logic 1 signal 0 } 
	{ m_axi_IMG_RDATA sc_in sc_lv 64 signal 0 } 
	{ m_axi_IMG_RLAST sc_in sc_logic 1 signal 0 } 
	{ m_axi_IMG_RID sc_in sc_lv 1 signal 0 } 
	{ m_axi_IMG_RUSER sc_in sc_lv 1 signal 0 } 
	{ m_axi_IMG_RRESP sc_in sc_lv 2 signal 0 } 
	{ m_axi_IMG_BVALID sc_in sc_logic 1 signal 0 } 
	{ m_axi_IMG_BREADY sc_out sc_logic 1 signal 0 } 
	{ m_axi_IMG_BRESP sc_in sc_lv 2 signal 0 } 
	{ m_axi_IMG_BID sc_in sc_lv 1 signal 0 } 
	{ m_axi_IMG_BUSER sc_in sc_lv 1 signal 0 } 
	{ m_axi_RESULT_AWVALID sc_out sc_logic 1 signal 1 } 
	{ m_axi_RESULT_AWREADY sc_in sc_logic 1 signal 1 } 
	{ m_axi_RESULT_AWADDR sc_out sc_lv 32 signal 1 } 
	{ m_axi_RESULT_AWID sc_out sc_lv 1 signal 1 } 
	{ m_axi_RESULT_AWLEN sc_out sc_lv 8 signal 1 } 
	{ m_axi_RESULT_AWSIZE sc_out sc_lv 3 signal 1 } 
	{ m_axi_RESULT_AWBURST sc_out sc_lv 2 signal 1 } 
	{ m_axi_RESULT_AWLOCK sc_out sc_lv 2 signal 1 } 
	{ m_axi_RESULT_AWCACHE sc_out sc_lv 4 signal 1 } 
	{ m_axi_RESULT_AWPROT sc_out sc_lv 3 signal 1 } 
	{ m_axi_RESULT_AWQOS sc_out sc_lv 4 signal 1 } 
	{ m_axi_RESULT_AWREGION sc_out sc_lv 4 signal 1 } 
	{ m_axi_RESULT_AWUSER sc_out sc_lv 1 signal 1 } 
	{ m_axi_RESULT_WVALID sc_out sc_logic 1 signal 1 } 
	{ m_axi_RESULT_WREADY sc_in sc_logic 1 signal 1 } 
	{ m_axi_RESULT_WDATA sc_out sc_lv 32 signal 1 } 
	{ m_axi_RESULT_WSTRB sc_out sc_lv 4 signal 1 } 
	{ m_axi_RESULT_WLAST sc_out sc_logic 1 signal 1 } 
	{ m_axi_RESULT_WID sc_out sc_lv 1 signal 1 } 
	{ m_axi_RESULT_WUSER sc_out sc_lv 1 signal 1 } 
	{ m_axi_RESULT_ARVALID sc_out sc_logic 1 signal 1 } 
	{ m_axi_RESULT_ARREADY sc_in sc_logic 1 signal 1 } 
	{ m_axi_RESULT_ARADDR sc_out sc_lv 32 signal 1 } 
	{ m_axi_RESULT_ARID sc_out sc_lv 1 signal 1 } 
	{ m_axi_RESULT_ARLEN sc_out sc_lv 8 signal 1 } 
	{ m_axi_RESULT_ARSIZE sc_out sc_lv 3 signal 1 } 
	{ m_axi_RESULT_ARBURST sc_out sc_lv 2 signal 1 } 
	{ m_axi_RESULT_ARLOCK sc_out sc_lv 2 signal 1 } 
	{ m_axi_RESULT_ARCACHE sc_out sc_lv 4 signal 1 } 
	{ m_axi_RESULT_ARPROT sc_out sc_lv 3 signal 1 } 
	{ m_axi_RESULT_ARQOS sc_out sc_lv 4 signal 1 } 
	{ m_axi_RESULT_ARREGION sc_out sc_lv 4 signal 1 } 
	{ m_axi_RESULT_ARUSER sc_out sc_lv 1 signal 1 } 
	{ m_axi_RESULT_RVALID sc_in sc_logic 1 signal 1 } 
	{ m_axi_RESULT_RREADY sc_out sc_logic 1 signal 1 } 
	{ m_axi_RESULT_RDATA sc_in sc_lv 32 signal 1 } 
	{ m_axi_RESULT_RLAST sc_in sc_logic 1 signal 1 } 
	{ m_axi_RESULT_RID sc_in sc_lv 1 signal 1 } 
	{ m_axi_RESULT_RUSER sc_in sc_lv 1 signal 1 } 
	{ m_axi_RESULT_RRESP sc_in sc_lv 2 signal 1 } 
	{ m_axi_RESULT_BVALID sc_in sc_logic 1 signal 1 } 
	{ m_axi_RESULT_BREADY sc_out sc_logic 1 signal 1 } 
	{ m_axi_RESULT_BRESP sc_in sc_lv 2 signal 1 } 
	{ m_axi_RESULT_BID sc_in sc_lv 1 signal 1 } 
	{ m_axi_RESULT_BUSER sc_in sc_lv 1 signal 1 } 
	{ s_axi_CTRL_AWVALID sc_in sc_logic 1 signal -1 } 
	{ s_axi_CTRL_AWREADY sc_out sc_logic 1 signal -1 } 
	{ s_axi_CTRL_AWADDR sc_in sc_lv 5 signal -1 } 
	{ s_axi_CTRL_WVALID sc_in sc_logic 1 signal -1 } 
	{ s_axi_CTRL_WREADY sc_out sc_logic 1 signal -1 } 
	{ s_axi_CTRL_WDATA sc_in sc_lv 32 signal -1 } 
	{ s_axi_CTRL_WSTRB sc_in sc_lv 4 signal -1 } 
	{ s_axi_CTRL_ARVALID sc_in sc_logic 1 signal -1 } 
	{ s_axi_CTRL_ARREADY sc_out sc_logic 1 signal -1 } 
	{ s_axi_CTRL_ARADDR sc_in sc_lv 5 signal -1 } 
	{ s_axi_CTRL_RVALID sc_out sc_logic 1 signal -1 } 
	{ s_axi_CTRL_RREADY sc_in sc_logic 1 signal -1 } 
	{ s_axi_CTRL_RDATA sc_out sc_lv 32 signal -1 } 
	{ s_axi_CTRL_RRESP sc_out sc_lv 2 signal -1 } 
	{ s_axi_CTRL_BVALID sc_out sc_logic 1 signal -1 } 
	{ s_axi_CTRL_BREADY sc_in sc_logic 1 signal -1 } 
	{ s_axi_CTRL_BRESP sc_out sc_lv 2 signal -1 } 
	{ interrupt sc_out sc_logic 1 signal -1 } 
}
set NewPortList {[ 
	{ "name": "s_axi_CTRL_AWADDR", "direction": "in", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "CTRL", "role": "AWADDR" },"address":[{"name":"FracNet_T","role":"start","value":"0","valid_bit":"0"},{"name":"FracNet_T","role":"continue","value":"0","valid_bit":"4"},{"name":"FracNet_T","role":"auto_start","value":"0","valid_bit":"7"},{"name":"image_V","role":"data","value":"16"},{"name":"output_r","role":"data","value":"24"}] },
	{ "name": "s_axi_CTRL_AWVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "CTRL", "role": "AWVALID" } },
	{ "name": "s_axi_CTRL_AWREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "CTRL", "role": "AWREADY" } },
	{ "name": "s_axi_CTRL_WVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "CTRL", "role": "WVALID" } },
	{ "name": "s_axi_CTRL_WREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "CTRL", "role": "WREADY" } },
	{ "name": "s_axi_CTRL_WDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "CTRL", "role": "WDATA" } },
	{ "name": "s_axi_CTRL_WSTRB", "direction": "in", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "CTRL", "role": "WSTRB" } },
	{ "name": "s_axi_CTRL_ARADDR", "direction": "in", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "CTRL", "role": "ARADDR" },"address":[{"name":"FracNet_T","role":"start","value":"0","valid_bit":"0"},{"name":"FracNet_T","role":"done","value":"0","valid_bit":"1"},{"name":"FracNet_T","role":"idle","value":"0","valid_bit":"2"},{"name":"FracNet_T","role":"ready","value":"0","valid_bit":"3"},{"name":"FracNet_T","role":"auto_start","value":"0","valid_bit":"7"}] },
	{ "name": "s_axi_CTRL_ARVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "CTRL", "role": "ARVALID" } },
	{ "name": "s_axi_CTRL_ARREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "CTRL", "role": "ARREADY" } },
	{ "name": "s_axi_CTRL_RVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "CTRL", "role": "RVALID" } },
	{ "name": "s_axi_CTRL_RREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "CTRL", "role": "RREADY" } },
	{ "name": "s_axi_CTRL_RDATA", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "CTRL", "role": "RDATA" } },
	{ "name": "s_axi_CTRL_RRESP", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "CTRL", "role": "RRESP" } },
	{ "name": "s_axi_CTRL_BVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "CTRL", "role": "BVALID" } },
	{ "name": "s_axi_CTRL_BREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "CTRL", "role": "BREADY" } },
	{ "name": "s_axi_CTRL_BRESP", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "CTRL", "role": "BRESP" } },
	{ "name": "interrupt", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "CTRL", "role": "interrupt" } }, 
 	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst_n", "role": "default" }} , 
 	{ "name": "m_axi_IMG_AWVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "IMG", "role": "AWVALID" }} , 
 	{ "name": "m_axi_IMG_AWREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "IMG", "role": "AWREADY" }} , 
 	{ "name": "m_axi_IMG_AWADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "IMG", "role": "AWADDR" }} , 
 	{ "name": "m_axi_IMG_AWID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "IMG", "role": "AWID" }} , 
 	{ "name": "m_axi_IMG_AWLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "IMG", "role": "AWLEN" }} , 
 	{ "name": "m_axi_IMG_AWSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "IMG", "role": "AWSIZE" }} , 
 	{ "name": "m_axi_IMG_AWBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "IMG", "role": "AWBURST" }} , 
 	{ "name": "m_axi_IMG_AWLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "IMG", "role": "AWLOCK" }} , 
 	{ "name": "m_axi_IMG_AWCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "IMG", "role": "AWCACHE" }} , 
 	{ "name": "m_axi_IMG_AWPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "IMG", "role": "AWPROT" }} , 
 	{ "name": "m_axi_IMG_AWQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "IMG", "role": "AWQOS" }} , 
 	{ "name": "m_axi_IMG_AWREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "IMG", "role": "AWREGION" }} , 
 	{ "name": "m_axi_IMG_AWUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "IMG", "role": "AWUSER" }} , 
 	{ "name": "m_axi_IMG_WVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "IMG", "role": "WVALID" }} , 
 	{ "name": "m_axi_IMG_WREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "IMG", "role": "WREADY" }} , 
 	{ "name": "m_axi_IMG_WDATA", "direction": "out", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "IMG", "role": "WDATA" }} , 
 	{ "name": "m_axi_IMG_WSTRB", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "IMG", "role": "WSTRB" }} , 
 	{ "name": "m_axi_IMG_WLAST", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "IMG", "role": "WLAST" }} , 
 	{ "name": "m_axi_IMG_WID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "IMG", "role": "WID" }} , 
 	{ "name": "m_axi_IMG_WUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "IMG", "role": "WUSER" }} , 
 	{ "name": "m_axi_IMG_ARVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "IMG", "role": "ARVALID" }} , 
 	{ "name": "m_axi_IMG_ARREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "IMG", "role": "ARREADY" }} , 
 	{ "name": "m_axi_IMG_ARADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "IMG", "role": "ARADDR" }} , 
 	{ "name": "m_axi_IMG_ARID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "IMG", "role": "ARID" }} , 
 	{ "name": "m_axi_IMG_ARLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "IMG", "role": "ARLEN" }} , 
 	{ "name": "m_axi_IMG_ARSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "IMG", "role": "ARSIZE" }} , 
 	{ "name": "m_axi_IMG_ARBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "IMG", "role": "ARBURST" }} , 
 	{ "name": "m_axi_IMG_ARLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "IMG", "role": "ARLOCK" }} , 
 	{ "name": "m_axi_IMG_ARCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "IMG", "role": "ARCACHE" }} , 
 	{ "name": "m_axi_IMG_ARPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "IMG", "role": "ARPROT" }} , 
 	{ "name": "m_axi_IMG_ARQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "IMG", "role": "ARQOS" }} , 
 	{ "name": "m_axi_IMG_ARREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "IMG", "role": "ARREGION" }} , 
 	{ "name": "m_axi_IMG_ARUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "IMG", "role": "ARUSER" }} , 
 	{ "name": "m_axi_IMG_RVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "IMG", "role": "RVALID" }} , 
 	{ "name": "m_axi_IMG_RREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "IMG", "role": "RREADY" }} , 
 	{ "name": "m_axi_IMG_RDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "IMG", "role": "RDATA" }} , 
 	{ "name": "m_axi_IMG_RLAST", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "IMG", "role": "RLAST" }} , 
 	{ "name": "m_axi_IMG_RID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "IMG", "role": "RID" }} , 
 	{ "name": "m_axi_IMG_RUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "IMG", "role": "RUSER" }} , 
 	{ "name": "m_axi_IMG_RRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "IMG", "role": "RRESP" }} , 
 	{ "name": "m_axi_IMG_BVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "IMG", "role": "BVALID" }} , 
 	{ "name": "m_axi_IMG_BREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "IMG", "role": "BREADY" }} , 
 	{ "name": "m_axi_IMG_BRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "IMG", "role": "BRESP" }} , 
 	{ "name": "m_axi_IMG_BID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "IMG", "role": "BID" }} , 
 	{ "name": "m_axi_IMG_BUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "IMG", "role": "BUSER" }} , 
 	{ "name": "m_axi_RESULT_AWVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "RESULT", "role": "AWVALID" }} , 
 	{ "name": "m_axi_RESULT_AWREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "RESULT", "role": "AWREADY" }} , 
 	{ "name": "m_axi_RESULT_AWADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "RESULT", "role": "AWADDR" }} , 
 	{ "name": "m_axi_RESULT_AWID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "RESULT", "role": "AWID" }} , 
 	{ "name": "m_axi_RESULT_AWLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "RESULT", "role": "AWLEN" }} , 
 	{ "name": "m_axi_RESULT_AWSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "RESULT", "role": "AWSIZE" }} , 
 	{ "name": "m_axi_RESULT_AWBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "RESULT", "role": "AWBURST" }} , 
 	{ "name": "m_axi_RESULT_AWLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "RESULT", "role": "AWLOCK" }} , 
 	{ "name": "m_axi_RESULT_AWCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "RESULT", "role": "AWCACHE" }} , 
 	{ "name": "m_axi_RESULT_AWPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "RESULT", "role": "AWPROT" }} , 
 	{ "name": "m_axi_RESULT_AWQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "RESULT", "role": "AWQOS" }} , 
 	{ "name": "m_axi_RESULT_AWREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "RESULT", "role": "AWREGION" }} , 
 	{ "name": "m_axi_RESULT_AWUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "RESULT", "role": "AWUSER" }} , 
 	{ "name": "m_axi_RESULT_WVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "RESULT", "role": "WVALID" }} , 
 	{ "name": "m_axi_RESULT_WREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "RESULT", "role": "WREADY" }} , 
 	{ "name": "m_axi_RESULT_WDATA", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "RESULT", "role": "WDATA" }} , 
 	{ "name": "m_axi_RESULT_WSTRB", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "RESULT", "role": "WSTRB" }} , 
 	{ "name": "m_axi_RESULT_WLAST", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "RESULT", "role": "WLAST" }} , 
 	{ "name": "m_axi_RESULT_WID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "RESULT", "role": "WID" }} , 
 	{ "name": "m_axi_RESULT_WUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "RESULT", "role": "WUSER" }} , 
 	{ "name": "m_axi_RESULT_ARVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "RESULT", "role": "ARVALID" }} , 
 	{ "name": "m_axi_RESULT_ARREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "RESULT", "role": "ARREADY" }} , 
 	{ "name": "m_axi_RESULT_ARADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "RESULT", "role": "ARADDR" }} , 
 	{ "name": "m_axi_RESULT_ARID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "RESULT", "role": "ARID" }} , 
 	{ "name": "m_axi_RESULT_ARLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "RESULT", "role": "ARLEN" }} , 
 	{ "name": "m_axi_RESULT_ARSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "RESULT", "role": "ARSIZE" }} , 
 	{ "name": "m_axi_RESULT_ARBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "RESULT", "role": "ARBURST" }} , 
 	{ "name": "m_axi_RESULT_ARLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "RESULT", "role": "ARLOCK" }} , 
 	{ "name": "m_axi_RESULT_ARCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "RESULT", "role": "ARCACHE" }} , 
 	{ "name": "m_axi_RESULT_ARPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "RESULT", "role": "ARPROT" }} , 
 	{ "name": "m_axi_RESULT_ARQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "RESULT", "role": "ARQOS" }} , 
 	{ "name": "m_axi_RESULT_ARREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "RESULT", "role": "ARREGION" }} , 
 	{ "name": "m_axi_RESULT_ARUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "RESULT", "role": "ARUSER" }} , 
 	{ "name": "m_axi_RESULT_RVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "RESULT", "role": "RVALID" }} , 
 	{ "name": "m_axi_RESULT_RREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "RESULT", "role": "RREADY" }} , 
 	{ "name": "m_axi_RESULT_RDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "RESULT", "role": "RDATA" }} , 
 	{ "name": "m_axi_RESULT_RLAST", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "RESULT", "role": "RLAST" }} , 
 	{ "name": "m_axi_RESULT_RID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "RESULT", "role": "RID" }} , 
 	{ "name": "m_axi_RESULT_RUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "RESULT", "role": "RUSER" }} , 
 	{ "name": "m_axi_RESULT_RRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "RESULT", "role": "RRESP" }} , 
 	{ "name": "m_axi_RESULT_BVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "RESULT", "role": "BVALID" }} , 
 	{ "name": "m_axi_RESULT_BREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "RESULT", "role": "BREADY" }} , 
 	{ "name": "m_axi_RESULT_BRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "RESULT", "role": "BRESP" }} , 
 	{ "name": "m_axi_RESULT_BID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "RESULT", "role": "BID" }} , 
 	{ "name": "m_axi_RESULT_BUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "RESULT", "role": "BUSER" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60", "61", "62", "63", "64", "65", "66", "67", "68", "69", "70", "71", "72", "73", "74", "75", "76", "77", "78", "79", "80", "81", "82", "83", "84", "85", "86", "87", "88", "89", "90", "91", "92", "93", "94", "95", "96", "97", "98", "99", "100", "101", "102", "103", "104", "105", "106", "429", "462", "673", "706", "724", "725", "748"],
		"CDFG" : "FracNet_T",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "28120", "EstimateLatencyMax" : "232782",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state16", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968"},
			{"State" : "ap_ST_fsm_state18", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968"},
			{"State" : "ap_ST_fsm_state20", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968"},
			{"State" : "ap_ST_fsm_state22", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968"},
			{"State" : "ap_ST_fsm_state24", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968"},
			{"State" : "ap_ST_fsm_state26", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968"},
			{"State" : "ap_ST_fsm_state32", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968"},
			{"State" : "ap_ST_fsm_state34", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968"},
			{"State" : "ap_ST_fsm_state40", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968"},
			{"State" : "ap_ST_fsm_state42", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968"},
			{"State" : "ap_ST_fsm_state48", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968"},
			{"State" : "ap_ST_fsm_state50", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968"},
			{"State" : "ap_ST_fsm_state56", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968"},
			{"State" : "ap_ST_fsm_state58", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968"},
			{"State" : "ap_ST_fsm_state64", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968"},
			{"State" : "ap_ST_fsm_state66", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968"},
			{"State" : "ap_ST_fsm_state72", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968"},
			{"State" : "ap_ST_fsm_state74", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968"},
			{"State" : "ap_ST_fsm_state85", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968"},
			{"State" : "ap_ST_fsm_state87", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968"},
			{"State" : "ap_ST_fsm_state95", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968"},
			{"State" : "ap_ST_fsm_state97", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968"},
			{"State" : "ap_ST_fsm_state105", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968"},
			{"State" : "ap_ST_fsm_state107", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968"},
			{"State" : "ap_ST_fsm_state115", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968"},
			{"State" : "ap_ST_fsm_state117", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968"},
			{"State" : "ap_ST_fsm_state125", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968"},
			{"State" : "ap_ST_fsm_state127", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968"},
			{"State" : "ap_ST_fsm_state135", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968"},
			{"State" : "ap_ST_fsm_state137", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968"},
			{"State" : "ap_ST_fsm_state147", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968"},
			{"State" : "ap_ST_fsm_state149", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968"},
			{"State" : "ap_ST_fsm_state157", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968"},
			{"State" : "ap_ST_fsm_state159", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968"},
			{"State" : "ap_ST_fsm_state167", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968"},
			{"State" : "ap_ST_fsm_state169", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968"},
			{"State" : "ap_ST_fsm_state177", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968"},
			{"State" : "ap_ST_fsm_state179", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968"},
			{"State" : "ap_ST_fsm_state187", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968"},
			{"State" : "ap_ST_fsm_state189", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968"},
			{"State" : "ap_ST_fsm_state197", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968"},
			{"State" : "ap_ST_fsm_state199", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968"},
			{"State" : "ap_ST_fsm_state30", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_quant_and_pack_fu_7897"},
			{"State" : "ap_ST_fsm_state38", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_quant_and_pack_fu_7897"},
			{"State" : "ap_ST_fsm_state46", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_quant_and_pack_fu_7897"},
			{"State" : "ap_ST_fsm_state54", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_quant_and_pack_fu_7897"},
			{"State" : "ap_ST_fsm_state62", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_quant_and_pack_fu_7897"},
			{"State" : "ap_ST_fsm_state70", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_quant_and_pack_fu_7897"},
			{"State" : "ap_ST_fsm_state78", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_quant_and_pack_fu_7897"},
			{"State" : "ap_ST_fsm_state90", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_quant_and_pack_fu_7897"},
			{"State" : "ap_ST_fsm_state100", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_quant_and_pack_fu_7897"},
			{"State" : "ap_ST_fsm_state110", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_quant_and_pack_fu_7897"},
			{"State" : "ap_ST_fsm_state120", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_quant_and_pack_fu_7897"},
			{"State" : "ap_ST_fsm_state130", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_quant_and_pack_fu_7897"},
			{"State" : "ap_ST_fsm_state140", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_quant_and_pack_fu_7897"},
			{"State" : "ap_ST_fsm_state152", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_quant_and_pack_fu_7897"},
			{"State" : "ap_ST_fsm_state162", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_quant_and_pack_fu_7897"},
			{"State" : "ap_ST_fsm_state172", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_quant_and_pack_fu_7897"},
			{"State" : "ap_ST_fsm_state182", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_quant_and_pack_fu_7897"},
			{"State" : "ap_ST_fsm_state192", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_quant_and_pack_fu_7897"},
			{"State" : "ap_ST_fsm_state36", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_bn_relu_shortcut_fu_8106"},
			{"State" : "ap_ST_fsm_state44", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_bn_relu_shortcut_fu_8106"},
			{"State" : "ap_ST_fsm_state52", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_bn_relu_shortcut_fu_8106"},
			{"State" : "ap_ST_fsm_state60", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_bn_relu_shortcut_fu_8106"},
			{"State" : "ap_ST_fsm_state68", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_bn_relu_shortcut_fu_8106"},
			{"State" : "ap_ST_fsm_state76", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_bn_relu_shortcut_fu_8106"},
			{"State" : "ap_ST_fsm_state89", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_bn_relu_shortcut_fu_8106"},
			{"State" : "ap_ST_fsm_state99", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_bn_relu_shortcut_fu_8106"},
			{"State" : "ap_ST_fsm_state109", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_bn_relu_shortcut_fu_8106"},
			{"State" : "ap_ST_fsm_state119", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_bn_relu_shortcut_fu_8106"},
			{"State" : "ap_ST_fsm_state129", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_bn_relu_shortcut_fu_8106"},
			{"State" : "ap_ST_fsm_state139", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_bn_relu_shortcut_fu_8106"},
			{"State" : "ap_ST_fsm_state151", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_bn_relu_shortcut_fu_8106"},
			{"State" : "ap_ST_fsm_state161", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_bn_relu_shortcut_fu_8106"},
			{"State" : "ap_ST_fsm_state171", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_bn_relu_shortcut_fu_8106"},
			{"State" : "ap_ST_fsm_state181", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_bn_relu_shortcut_fu_8106"},
			{"State" : "ap_ST_fsm_state191", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_bn_relu_shortcut_fu_8106"},
			{"State" : "ap_ST_fsm_state201", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_bn_relu_shortcut_fu_8106"},
			{"State" : "ap_ST_fsm_state80", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_avgpool_concat_fu_18852"},
			{"State" : "ap_ST_fsm_state142", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_avgpool_concat_fu_18852"},
			{"State" : "ap_ST_fsm_state203", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_avgpool_8x8_fu_18926"},
			{"State" : "ap_ST_fsm_state28", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_bn1_fu_18996"},
			{"State" : "ap_ST_fsm_state205", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_matmul_fu_19032"}],
		"Port" : [
			{"Name" : "IMG", "Type" : "MAXI", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "IMG_blk_n_AR", "Type" : "RtlSignal"},
					{"Name" : "IMG_blk_n_R", "Type" : "RtlSignal"}]},
			{"Name" : "RESULT", "Type" : "MAXI", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "RESULT_blk_n_AW", "Type" : "RtlSignal"},
					{"Name" : "RESULT_blk_n_W", "Type" : "RtlSignal"},
					{"Name" : "RESULT_blk_n_B", "Type" : "RtlSignal"}]},
			{"Name" : "image_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "output_r", "Type" : "None", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_0_8", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_0_8"}]},
			{"Name" : "conv_weight_all_V_0_7", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_0_7"}]},
			{"Name" : "conv_weight_all_V_0_6", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_0_6"}]},
			{"Name" : "conv_weight_all_V_0_5", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_0_5"}]},
			{"Name" : "conv_weight_all_V_0_4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_0_4"}]},
			{"Name" : "conv_weight_all_V_0_3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_0_3"}]},
			{"Name" : "conv_weight_all_V_0_2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_0_2"}]},
			{"Name" : "conv_weight_all_V_0_1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_0_1"}]},
			{"Name" : "conv_weight_all_V_0_s", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_0_s"}]},
			{"Name" : "conv_weight_all_V_1_8", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_1_8"}]},
			{"Name" : "conv_weight_all_V_1_7", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_1_7"}]},
			{"Name" : "conv_weight_all_V_1_6", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_1_6"}]},
			{"Name" : "conv_weight_all_V_1_5", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_1_5"}]},
			{"Name" : "conv_weight_all_V_1_4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_1_4"}]},
			{"Name" : "conv_weight_all_V_1_3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_1_3"}]},
			{"Name" : "conv_weight_all_V_1_2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_1_2"}]},
			{"Name" : "conv_weight_all_V_1_1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_1_1"}]},
			{"Name" : "conv_weight_all_V_1_s", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_1_s"}]},
			{"Name" : "conv_weight_all_V_2_8", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_2_8"}]},
			{"Name" : "conv_weight_all_V_2_7", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_2_7"}]},
			{"Name" : "conv_weight_all_V_2_6", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_2_6"}]},
			{"Name" : "conv_weight_all_V_2_5", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_2_5"}]},
			{"Name" : "conv_weight_all_V_2_4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_2_4"}]},
			{"Name" : "conv_weight_all_V_2_3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_2_3"}]},
			{"Name" : "conv_weight_all_V_2_2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_2_2"}]},
			{"Name" : "conv_weight_all_V_2_1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_2_1"}]},
			{"Name" : "conv_weight_all_V_2_s", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_2_s"}]},
			{"Name" : "conv_weight_all_V_3_8", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_3_8"}]},
			{"Name" : "conv_weight_all_V_3_7", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_3_7"}]},
			{"Name" : "conv_weight_all_V_3_6", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_3_6"}]},
			{"Name" : "conv_weight_all_V_3_5", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_3_5"}]},
			{"Name" : "conv_weight_all_V_3_4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_3_4"}]},
			{"Name" : "conv_weight_all_V_3_3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_3_3"}]},
			{"Name" : "conv_weight_all_V_3_2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_3_2"}]},
			{"Name" : "conv_weight_all_V_3_1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_3_1"}]},
			{"Name" : "conv_weight_all_V_3_s", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_3_s"}]},
			{"Name" : "conv_weight_all_V_4_8", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_4_8"}]},
			{"Name" : "conv_weight_all_V_4_7", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_4_7"}]},
			{"Name" : "conv_weight_all_V_4_6", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_4_6"}]},
			{"Name" : "conv_weight_all_V_4_5", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_4_5"}]},
			{"Name" : "conv_weight_all_V_4_4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_4_4"}]},
			{"Name" : "conv_weight_all_V_4_3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_4_3"}]},
			{"Name" : "conv_weight_all_V_4_2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_4_2"}]},
			{"Name" : "conv_weight_all_V_4_1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_4_1"}]},
			{"Name" : "conv_weight_all_V_4_s", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_4_s"}]},
			{"Name" : "conv_weight_all_V_5_8", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_5_8"}]},
			{"Name" : "conv_weight_all_V_5_7", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_5_7"}]},
			{"Name" : "conv_weight_all_V_5_6", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_5_6"}]},
			{"Name" : "conv_weight_all_V_5_5", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_5_5"}]},
			{"Name" : "conv_weight_all_V_5_4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_5_4"}]},
			{"Name" : "conv_weight_all_V_5_3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_5_3"}]},
			{"Name" : "conv_weight_all_V_5_2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_5_2"}]},
			{"Name" : "conv_weight_all_V_5_1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_5_1"}]},
			{"Name" : "conv_weight_all_V_5_s", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_5_s"}]},
			{"Name" : "conv_weight_all_V_6_8", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_6_8"}]},
			{"Name" : "conv_weight_all_V_6_7", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_6_7"}]},
			{"Name" : "conv_weight_all_V_6_6", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_6_6"}]},
			{"Name" : "conv_weight_all_V_6_5", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_6_5"}]},
			{"Name" : "conv_weight_all_V_6_4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_6_4"}]},
			{"Name" : "conv_weight_all_V_6_3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_6_3"}]},
			{"Name" : "conv_weight_all_V_6_2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_6_2"}]},
			{"Name" : "conv_weight_all_V_6_1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_6_1"}]},
			{"Name" : "conv_weight_all_V_6_s", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_6_s"}]},
			{"Name" : "conv_weight_all_V_7_8", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_7_8"}]},
			{"Name" : "conv_weight_all_V_7_7", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_7_7"}]},
			{"Name" : "conv_weight_all_V_7_6", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_7_6"}]},
			{"Name" : "conv_weight_all_V_7_5", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_7_5"}]},
			{"Name" : "conv_weight_all_V_7_4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_7_4"}]},
			{"Name" : "conv_weight_all_V_7_3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_7_3"}]},
			{"Name" : "conv_weight_all_V_7_2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_7_2"}]},
			{"Name" : "conv_weight_all_V_7_1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_7_1"}]},
			{"Name" : "conv_weight_all_V_7_s", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_7_s"}]},
			{"Name" : "conv_weight_all_V_8_8", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_8_8"}]},
			{"Name" : "conv_weight_all_V_8_7", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_8_7"}]},
			{"Name" : "conv_weight_all_V_8_6", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_8_6"}]},
			{"Name" : "conv_weight_all_V_8_5", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_8_5"}]},
			{"Name" : "conv_weight_all_V_8_4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_8_4"}]},
			{"Name" : "conv_weight_all_V_8_3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_8_3"}]},
			{"Name" : "conv_weight_all_V_8_2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_8_2"}]},
			{"Name" : "conv_weight_all_V_8_1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_8_1"}]},
			{"Name" : "conv_weight_all_V_8_s", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_8_s"}]},
			{"Name" : "conv_weight_all_V_9_8", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_9_8"}]},
			{"Name" : "conv_weight_all_V_9_7", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_9_7"}]},
			{"Name" : "conv_weight_all_V_9_6", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_9_6"}]},
			{"Name" : "conv_weight_all_V_9_5", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_9_5"}]},
			{"Name" : "conv_weight_all_V_9_4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_9_4"}]},
			{"Name" : "conv_weight_all_V_9_3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_9_3"}]},
			{"Name" : "conv_weight_all_V_9_2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_9_2"}]},
			{"Name" : "conv_weight_all_V_9_1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_9_1"}]},
			{"Name" : "conv_weight_all_V_9_s", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_9_s"}]},
			{"Name" : "conv_weight_all_V_10_8", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_10_8"}]},
			{"Name" : "conv_weight_all_V_10_7", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_10_7"}]},
			{"Name" : "conv_weight_all_V_10_6", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_10_6"}]},
			{"Name" : "conv_weight_all_V_10_5", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_10_5"}]},
			{"Name" : "conv_weight_all_V_10_4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_10_4"}]},
			{"Name" : "conv_weight_all_V_10_3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_10_3"}]},
			{"Name" : "conv_weight_all_V_10_2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_10_2"}]},
			{"Name" : "conv_weight_all_V_10_1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_10_1"}]},
			{"Name" : "conv_weight_all_V_10", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_10"}]},
			{"Name" : "conv_weight_all_V_11_8", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_11_8"}]},
			{"Name" : "conv_weight_all_V_11_7", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_11_7"}]},
			{"Name" : "conv_weight_all_V_11_6", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_11_6"}]},
			{"Name" : "conv_weight_all_V_11_5", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_11_5"}]},
			{"Name" : "conv_weight_all_V_11_4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_11_4"}]},
			{"Name" : "conv_weight_all_V_11_3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_11_3"}]},
			{"Name" : "conv_weight_all_V_11_2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_11_2"}]},
			{"Name" : "conv_weight_all_V_11_1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_11_1"}]},
			{"Name" : "conv_weight_all_V_11", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_11"}]},
			{"Name" : "conv_weight_all_V_12_8", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_12_8"}]},
			{"Name" : "conv_weight_all_V_12_7", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_12_7"}]},
			{"Name" : "conv_weight_all_V_12_6", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_12_6"}]},
			{"Name" : "conv_weight_all_V_12_5", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_12_5"}]},
			{"Name" : "conv_weight_all_V_12_4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_12_4"}]},
			{"Name" : "conv_weight_all_V_12_3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_12_3"}]},
			{"Name" : "conv_weight_all_V_12_2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_12_2"}]},
			{"Name" : "conv_weight_all_V_12_1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_12_1"}]},
			{"Name" : "conv_weight_all_V_12", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_12"}]},
			{"Name" : "conv_weight_all_V_13_8", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_13_8"}]},
			{"Name" : "conv_weight_all_V_13_7", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_13_7"}]},
			{"Name" : "conv_weight_all_V_13_6", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_13_6"}]},
			{"Name" : "conv_weight_all_V_13_5", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_13_5"}]},
			{"Name" : "conv_weight_all_V_13_4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_13_4"}]},
			{"Name" : "conv_weight_all_V_13_3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_13_3"}]},
			{"Name" : "conv_weight_all_V_13_2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_13_2"}]},
			{"Name" : "conv_weight_all_V_13_1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_13_1"}]},
			{"Name" : "conv_weight_all_V_13", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_13"}]},
			{"Name" : "conv_weight_all_V_14_8", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_14_8"}]},
			{"Name" : "conv_weight_all_V_14_7", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_14_7"}]},
			{"Name" : "conv_weight_all_V_14_6", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_14_6"}]},
			{"Name" : "conv_weight_all_V_14_5", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_14_5"}]},
			{"Name" : "conv_weight_all_V_14_4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_14_4"}]},
			{"Name" : "conv_weight_all_V_14_3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_14_3"}]},
			{"Name" : "conv_weight_all_V_14_2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_14_2"}]},
			{"Name" : "conv_weight_all_V_14_1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_14_1"}]},
			{"Name" : "conv_weight_all_V_14", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_14"}]},
			{"Name" : "conv_weight_all_V_15_8", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_15_8"}]},
			{"Name" : "conv_weight_all_V_15_7", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_15_7"}]},
			{"Name" : "conv_weight_all_V_15_6", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_15_6"}]},
			{"Name" : "conv_weight_all_V_15_5", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_15_5"}]},
			{"Name" : "conv_weight_all_V_15_4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_15_4"}]},
			{"Name" : "conv_weight_all_V_15_3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_15_3"}]},
			{"Name" : "conv_weight_all_V_15_2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_15_2"}]},
			{"Name" : "conv_weight_all_V_15_1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_15_1"}]},
			{"Name" : "conv_weight_all_V_15", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "106", "SubInstance" : "grp_binary_conv3x3_tile_fu_5968", "Port" : "conv_weight_all_V_15"}]},
			{"Name" : "gate_mask", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "pool_out_buf", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "725", "SubInstance" : "grp_matmul_fu_19032", "Port" : "pool_out_buf"},
					{"ID" : "706", "SubInstance" : "grp_avgpool_8x8_fu_18926", "Port" : "outputs_V"}]},
			{"Name" : "linear_weight_fix_V_s", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "725", "SubInstance" : "grp_matmul_fu_19032", "Port" : "linear_weight_fix_V_s"}]},
			{"Name" : "linear_weight_fix_V_1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "725", "SubInstance" : "grp_matmul_fu_19032", "Port" : "linear_weight_fix_V_1"}]},
			{"Name" : "linear_weight_fix_V_2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "725", "SubInstance" : "grp_matmul_fu_19032", "Port" : "linear_weight_fix_V_2"}]},
			{"Name" : "linear_weight_fix_V_3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "725", "SubInstance" : "grp_matmul_fu_19032", "Port" : "linear_weight_fix_V_3"}]},
			{"Name" : "linear_weight_fix_V_4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "725", "SubInstance" : "grp_matmul_fu_19032", "Port" : "linear_weight_fix_V_4"}]},
			{"Name" : "linear_weight_fix_V_5", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "725", "SubInstance" : "grp_matmul_fu_19032", "Port" : "linear_weight_fix_V_5"}]},
			{"Name" : "linear_weight_fix_V_6", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "725", "SubInstance" : "grp_matmul_fu_19032", "Port" : "linear_weight_fix_V_6"}]},
			{"Name" : "linear_weight_fix_V_7", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "725", "SubInstance" : "grp_matmul_fu_19032", "Port" : "linear_weight_fix_V_7"}]},
			{"Name" : "linear_weight_fix_V_8", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "725", "SubInstance" : "grp_matmul_fu_19032", "Port" : "linear_weight_fix_V_8"}]},
			{"Name" : "linear_weight_fix_V_9", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "725", "SubInstance" : "grp_matmul_fu_19032", "Port" : "linear_weight_fix_V_9"}]}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.gate_mask_U", "Parent" : "0"},
	{"ID" : "2", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.pool_out_buf_U", "Parent" : "0"},
	{"ID" : "3", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.FracNet_T_CTRL_s_axi_U", "Parent" : "0"},
	{"ID" : "4", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.FracNet_T_IMG_m_axi_U", "Parent" : "0"},
	{"ID" : "5", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.FracNet_T_RESULT_m_axi_U", "Parent" : "0"},
	{"ID" : "6", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.msb_fmap_0_V_U", "Parent" : "0"},
	{"ID" : "7", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.msb_fmap_1_V_U", "Parent" : "0"},
	{"ID" : "8", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.msb_fmap_2_V_U", "Parent" : "0"},
	{"ID" : "9", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_0_0_V_U", "Parent" : "0"},
	{"ID" : "10", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_0_1_V_U", "Parent" : "0"},
	{"ID" : "11", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_0_2_V_U", "Parent" : "0"},
	{"ID" : "12", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_0_3_V_U", "Parent" : "0"},
	{"ID" : "13", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_0_4_V_U", "Parent" : "0"},
	{"ID" : "14", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_0_5_V_U", "Parent" : "0"},
	{"ID" : "15", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_0_6_V_U", "Parent" : "0"},
	{"ID" : "16", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_0_7_V_U", "Parent" : "0"},
	{"ID" : "17", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_0_8_V_U", "Parent" : "0"},
	{"ID" : "18", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_0_9_V_U", "Parent" : "0"},
	{"ID" : "19", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_0_10_V_U", "Parent" : "0"},
	{"ID" : "20", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_0_11_V_U", "Parent" : "0"},
	{"ID" : "21", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_0_12_V_U", "Parent" : "0"},
	{"ID" : "22", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_0_13_V_U", "Parent" : "0"},
	{"ID" : "23", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_0_14_V_U", "Parent" : "0"},
	{"ID" : "24", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_0_15_V_U", "Parent" : "0"},
	{"ID" : "25", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_1_0_V_U", "Parent" : "0"},
	{"ID" : "26", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_1_1_V_U", "Parent" : "0"},
	{"ID" : "27", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_1_2_V_U", "Parent" : "0"},
	{"ID" : "28", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_1_3_V_U", "Parent" : "0"},
	{"ID" : "29", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_1_4_V_U", "Parent" : "0"},
	{"ID" : "30", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_1_5_V_U", "Parent" : "0"},
	{"ID" : "31", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_1_6_V_U", "Parent" : "0"},
	{"ID" : "32", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_1_7_V_U", "Parent" : "0"},
	{"ID" : "33", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_1_8_V_U", "Parent" : "0"},
	{"ID" : "34", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_1_9_V_U", "Parent" : "0"},
	{"ID" : "35", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_1_10_V_U", "Parent" : "0"},
	{"ID" : "36", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_1_11_V_U", "Parent" : "0"},
	{"ID" : "37", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_1_12_V_U", "Parent" : "0"},
	{"ID" : "38", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_1_13_V_U", "Parent" : "0"},
	{"ID" : "39", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_1_14_V_U", "Parent" : "0"},
	{"ID" : "40", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_1_15_V_U", "Parent" : "0"},
	{"ID" : "41", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_2_0_V_U", "Parent" : "0"},
	{"ID" : "42", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_2_1_V_U", "Parent" : "0"},
	{"ID" : "43", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_2_2_V_U", "Parent" : "0"},
	{"ID" : "44", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_2_3_V_U", "Parent" : "0"},
	{"ID" : "45", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_2_4_V_U", "Parent" : "0"},
	{"ID" : "46", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_2_5_V_U", "Parent" : "0"},
	{"ID" : "47", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_2_6_V_U", "Parent" : "0"},
	{"ID" : "48", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_2_7_V_U", "Parent" : "0"},
	{"ID" : "49", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_2_8_V_U", "Parent" : "0"},
	{"ID" : "50", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_2_9_V_U", "Parent" : "0"},
	{"ID" : "51", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_2_10_V_U", "Parent" : "0"},
	{"ID" : "52", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_2_11_V_U", "Parent" : "0"},
	{"ID" : "53", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_2_12_V_U", "Parent" : "0"},
	{"ID" : "54", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_2_13_V_U", "Parent" : "0"},
	{"ID" : "55", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_2_14_V_U", "Parent" : "0"},
	{"ID" : "56", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_2_15_V_U", "Parent" : "0"},
	{"ID" : "57", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_3_0_V_U", "Parent" : "0"},
	{"ID" : "58", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_3_1_V_U", "Parent" : "0"},
	{"ID" : "59", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_3_2_V_U", "Parent" : "0"},
	{"ID" : "60", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_3_3_V_U", "Parent" : "0"},
	{"ID" : "61", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_3_4_V_U", "Parent" : "0"},
	{"ID" : "62", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_3_5_V_U", "Parent" : "0"},
	{"ID" : "63", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_3_6_V_U", "Parent" : "0"},
	{"ID" : "64", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_3_7_V_U", "Parent" : "0"},
	{"ID" : "65", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_3_8_V_U", "Parent" : "0"},
	{"ID" : "66", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_3_9_V_U", "Parent" : "0"},
	{"ID" : "67", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_3_10_V_U", "Parent" : "0"},
	{"ID" : "68", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_3_11_V_U", "Parent" : "0"},
	{"ID" : "69", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_3_12_V_U", "Parent" : "0"},
	{"ID" : "70", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_3_13_V_U", "Parent" : "0"},
	{"ID" : "71", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_3_14_V_U", "Parent" : "0"},
	{"ID" : "72", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_0_3_15_V_U", "Parent" : "0"},
	{"ID" : "73", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_t0_0_V_U", "Parent" : "0"},
	{"ID" : "74", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_t0_1_V_U", "Parent" : "0"},
	{"ID" : "75", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_t0_2_V_U", "Parent" : "0"},
	{"ID" : "76", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_t0_3_V_U", "Parent" : "0"},
	{"ID" : "77", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_t0_4_V_U", "Parent" : "0"},
	{"ID" : "78", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_t0_5_V_U", "Parent" : "0"},
	{"ID" : "79", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_t0_6_V_U", "Parent" : "0"},
	{"ID" : "80", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_t0_7_V_U", "Parent" : "0"},
	{"ID" : "81", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_t0_8_V_U", "Parent" : "0"},
	{"ID" : "82", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_t0_9_V_U", "Parent" : "0"},
	{"ID" : "83", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_t0_10_V_U", "Parent" : "0"},
	{"ID" : "84", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_t0_11_V_U", "Parent" : "0"},
	{"ID" : "85", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_t0_12_V_U", "Parent" : "0"},
	{"ID" : "86", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_t0_13_V_U", "Parent" : "0"},
	{"ID" : "87", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_t0_14_V_U", "Parent" : "0"},
	{"ID" : "88", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_t0_15_V_U", "Parent" : "0"},
	{"ID" : "89", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_t1_0_V_U", "Parent" : "0"},
	{"ID" : "90", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_t1_1_V_U", "Parent" : "0"},
	{"ID" : "91", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_t1_2_V_U", "Parent" : "0"},
	{"ID" : "92", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_t1_3_V_U", "Parent" : "0"},
	{"ID" : "93", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_t1_4_V_U", "Parent" : "0"},
	{"ID" : "94", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_t1_5_V_U", "Parent" : "0"},
	{"ID" : "95", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_t1_6_V_U", "Parent" : "0"},
	{"ID" : "96", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_t1_7_V_U", "Parent" : "0"},
	{"ID" : "97", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_t1_8_V_U", "Parent" : "0"},
	{"ID" : "98", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_t1_9_V_U", "Parent" : "0"},
	{"ID" : "99", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_t1_10_V_U", "Parent" : "0"},
	{"ID" : "100", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_t1_11_V_U", "Parent" : "0"},
	{"ID" : "101", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_t1_12_V_U", "Parent" : "0"},
	{"ID" : "102", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_t1_13_V_U", "Parent" : "0"},
	{"ID" : "103", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_t1_14_V_U", "Parent" : "0"},
	{"ID" : "104", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.out_buf_t1_15_V_U", "Parent" : "0"},
	{"ID" : "105", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.linear_out_buf_V_U", "Parent" : "0"},
	{"ID" : "106", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968", "Parent" : "0", "Child" : ["107", "108", "109", "110", "111", "112", "113", "114", "115", "116", "117", "118", "119", "120", "121", "122", "123", "124", "125", "126", "127", "128", "129", "130", "131", "132", "133", "134", "135", "136", "137", "138", "139", "140", "141", "142", "143", "144", "145", "146", "147", "148", "149", "150", "151", "152", "153", "154", "155", "156", "157", "158", "159", "160", "161", "162", "163", "164", "165", "166", "167", "168", "169", "170", "171", "172", "173", "174", "175", "176", "177", "178", "179", "180", "181", "182", "183", "184", "185", "186", "187", "188", "189", "190", "191", "192", "193", "194", "195", "196", "197", "198", "199", "200", "201", "202", "203", "204", "205", "206", "207", "208", "209", "210", "211", "212", "213", "214", "215", "216", "217", "218", "219", "220", "221", "222", "223", "224", "225", "226", "227", "228", "229", "230", "231", "232", "233", "234", "235", "236", "237", "238", "239", "240", "241", "242", "243", "244", "245", "246", "247", "248", "249", "250", "251", "252", "253", "254", "255", "256", "257", "258", "259", "260", "261", "262", "263", "264", "265", "266", "267", "268", "269", "270", "271", "272", "273", "274", "275", "276", "277", "278", "279", "280", "281", "282", "283", "284", "285", "286", "287", "288", "289", "290", "291", "292", "293", "294", "295", "296", "297", "298", "299", "300", "301", "302", "303", "304", "305", "306", "307", "308", "309", "310", "311", "312", "313", "314", "315", "316", "317", "318", "319", "320", "321", "322", "323", "324", "325", "326", "327", "328", "329", "330", "331", "332", "333", "334", "335", "336", "337", "338", "339", "340", "341", "342", "343", "344", "345", "346", "347", "348", "349", "350", "351", "352", "353", "354", "355", "356", "357", "358", "359", "360", "361", "362", "363", "364", "365", "366", "367", "368", "369", "370", "371", "372", "373", "374", "375", "376", "377", "378", "379", "380", "381", "382", "383", "384", "385", "386", "387", "388", "389", "390", "391", "392", "393", "394", "395", "396", "397", "398", "399", "400", "401", "402", "403", "404", "405", "406", "407", "408", "409", "410", "411", "412", "413", "414", "415", "416", "417", "418", "419", "420", "421", "422", "423", "424", "425", "426", "427", "428"],
		"CDFG" : "binary_conv3x3_tile",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "96", "EstimateLatencyMax" : "1104",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "msb_inputs_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "weights_V_offset", "Type" : "None", "Direction" : "I"},
			{"Name" : "msb_outputs_0_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "msb_outputs_1_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "msb_outputs_2_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "msb_outputs_3_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "msb_outputs_4_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "msb_outputs_5_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "msb_outputs_6_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "msb_outputs_7_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "msb_outputs_8_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "msb_outputs_9_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "msb_outputs_10_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "msb_outputs_11_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "msb_outputs_12_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "msb_outputs_13_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "msb_outputs_14_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "msb_outputs_15_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "comparator_0_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "comparator_1_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "comparator_2_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "comparator_3_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "comparator_4_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "comparator_5_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "comparator_6_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "comparator_7_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "comparator_8_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "comparator_9_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "comparator_10_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "comparator_11_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "comparator_12_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "comparator_13_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "comparator_14_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "comparator_15_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "p_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_read1", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_read2", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_read3", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_read4", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_read5", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_read6", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_read7", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_read8", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_read9", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_read10", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_read11", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_read12", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_read13", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_read14", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_read15", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_read16", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_read17", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_read18", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_read19", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_read20", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_read21", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_read22", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_read23", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_read24", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_read25", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_read26", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_read27", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_read28", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_read29", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_read30", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_read31", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_read32", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_read33", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_read34", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_read35", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_read36", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_read37", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_read38", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_read39", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_read40", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_read41", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_read42", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_read43", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_read44", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_read45", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_read46", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_read47", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_read48", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_read49", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_read50", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_read51", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_read52", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_read53", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_read54", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_read55", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_read56", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_read57", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_read58", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_read59", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_read60", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_read61", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_read62", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_read63", "Type" : "None", "Direction" : "I"},
			{"Name" : "threshold_V_offset", "Type" : "None", "Direction" : "I"},
			{"Name" : "switch_on", "Type" : "None", "Direction" : "I"},
			{"Name" : "c_in", "Type" : "None", "Direction" : "I"},
			{"Name" : "in_channels", "Type" : "None", "Direction" : "I"},
			{"Name" : "H_fmap_out", "Type" : "None", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_0_8", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_0_7", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_0_6", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_0_5", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_0_4", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_0_3", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_0_2", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_0_1", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_0_s", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_1_8", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_1_7", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_1_6", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_1_5", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_1_4", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_1_3", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_1_2", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_1_1", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_1_s", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_2_8", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_2_7", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_2_6", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_2_5", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_2_4", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_2_3", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_2_2", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_2_1", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_2_s", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_3_8", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_3_7", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_3_6", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_3_5", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_3_4", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_3_3", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_3_2", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_3_1", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_3_s", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_4_8", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_4_7", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_4_6", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_4_5", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_4_4", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_4_3", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_4_2", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_4_1", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_4_s", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_5_8", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_5_7", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_5_6", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_5_5", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_5_4", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_5_3", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_5_2", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_5_1", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_5_s", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_6_8", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_6_7", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_6_6", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_6_5", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_6_4", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_6_3", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_6_2", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_6_1", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_6_s", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_7_8", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_7_7", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_7_6", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_7_5", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_7_4", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_7_3", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_7_2", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_7_1", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_7_s", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_8_8", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_8_7", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_8_6", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_8_5", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_8_4", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_8_3", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_8_2", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_8_1", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_8_s", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_9_8", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_9_7", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_9_6", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_9_5", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_9_4", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_9_3", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_9_2", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_9_1", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_9_s", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_10_8", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_10_7", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_10_6", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_10_5", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_10_4", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_10_3", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_10_2", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_10_1", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_10", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_11_8", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_11_7", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_11_6", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_11_5", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_11_4", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_11_3", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_11_2", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_11_1", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_11", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_12_8", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_12_7", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_12_6", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_12_5", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_12_4", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_12_3", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_12_2", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_12_1", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_12", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_13_8", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_13_7", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_13_6", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_13_5", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_13_4", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_13_3", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_13_2", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_13_1", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_13", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_14_8", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_14_7", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_14_6", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_14_5", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_14_4", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_14_3", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_14_2", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_14_1", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_14", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_15_8", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_15_7", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_15_6", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_15_5", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_15_4", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_15_3", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_15_2", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_15_1", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "conv_weight_all_V_15", "Type" : "Memory", "Direction" : "I"}]},
	{"ID" : "107", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_0_8_U", "Parent" : "106"},
	{"ID" : "108", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_0_7_U", "Parent" : "106"},
	{"ID" : "109", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_0_6_U", "Parent" : "106"},
	{"ID" : "110", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_0_5_U", "Parent" : "106"},
	{"ID" : "111", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_0_4_U", "Parent" : "106"},
	{"ID" : "112", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_0_3_U", "Parent" : "106"},
	{"ID" : "113", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_0_2_U", "Parent" : "106"},
	{"ID" : "114", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_0_1_U", "Parent" : "106"},
	{"ID" : "115", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_0_s_U", "Parent" : "106"},
	{"ID" : "116", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_1_8_U", "Parent" : "106"},
	{"ID" : "117", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_1_7_U", "Parent" : "106"},
	{"ID" : "118", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_1_6_U", "Parent" : "106"},
	{"ID" : "119", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_1_5_U", "Parent" : "106"},
	{"ID" : "120", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_1_4_U", "Parent" : "106"},
	{"ID" : "121", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_1_3_U", "Parent" : "106"},
	{"ID" : "122", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_1_2_U", "Parent" : "106"},
	{"ID" : "123", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_1_1_U", "Parent" : "106"},
	{"ID" : "124", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_1_s_U", "Parent" : "106"},
	{"ID" : "125", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_2_8_U", "Parent" : "106"},
	{"ID" : "126", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_2_7_U", "Parent" : "106"},
	{"ID" : "127", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_2_6_U", "Parent" : "106"},
	{"ID" : "128", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_2_5_U", "Parent" : "106"},
	{"ID" : "129", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_2_4_U", "Parent" : "106"},
	{"ID" : "130", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_2_3_U", "Parent" : "106"},
	{"ID" : "131", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_2_2_U", "Parent" : "106"},
	{"ID" : "132", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_2_1_U", "Parent" : "106"},
	{"ID" : "133", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_2_s_U", "Parent" : "106"},
	{"ID" : "134", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_3_8_U", "Parent" : "106"},
	{"ID" : "135", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_3_7_U", "Parent" : "106"},
	{"ID" : "136", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_3_6_U", "Parent" : "106"},
	{"ID" : "137", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_3_5_U", "Parent" : "106"},
	{"ID" : "138", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_3_4_U", "Parent" : "106"},
	{"ID" : "139", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_3_3_U", "Parent" : "106"},
	{"ID" : "140", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_3_2_U", "Parent" : "106"},
	{"ID" : "141", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_3_1_U", "Parent" : "106"},
	{"ID" : "142", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_3_s_U", "Parent" : "106"},
	{"ID" : "143", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_4_8_U", "Parent" : "106"},
	{"ID" : "144", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_4_7_U", "Parent" : "106"},
	{"ID" : "145", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_4_6_U", "Parent" : "106"},
	{"ID" : "146", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_4_5_U", "Parent" : "106"},
	{"ID" : "147", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_4_4_U", "Parent" : "106"},
	{"ID" : "148", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_4_3_U", "Parent" : "106"},
	{"ID" : "149", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_4_2_U", "Parent" : "106"},
	{"ID" : "150", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_4_1_U", "Parent" : "106"},
	{"ID" : "151", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_4_s_U", "Parent" : "106"},
	{"ID" : "152", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_5_8_U", "Parent" : "106"},
	{"ID" : "153", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_5_7_U", "Parent" : "106"},
	{"ID" : "154", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_5_6_U", "Parent" : "106"},
	{"ID" : "155", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_5_5_U", "Parent" : "106"},
	{"ID" : "156", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_5_4_U", "Parent" : "106"},
	{"ID" : "157", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_5_3_U", "Parent" : "106"},
	{"ID" : "158", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_5_2_U", "Parent" : "106"},
	{"ID" : "159", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_5_1_U", "Parent" : "106"},
	{"ID" : "160", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_5_s_U", "Parent" : "106"},
	{"ID" : "161", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_6_8_U", "Parent" : "106"},
	{"ID" : "162", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_6_7_U", "Parent" : "106"},
	{"ID" : "163", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_6_6_U", "Parent" : "106"},
	{"ID" : "164", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_6_5_U", "Parent" : "106"},
	{"ID" : "165", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_6_4_U", "Parent" : "106"},
	{"ID" : "166", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_6_3_U", "Parent" : "106"},
	{"ID" : "167", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_6_2_U", "Parent" : "106"},
	{"ID" : "168", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_6_1_U", "Parent" : "106"},
	{"ID" : "169", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_6_s_U", "Parent" : "106"},
	{"ID" : "170", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_7_8_U", "Parent" : "106"},
	{"ID" : "171", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_7_7_U", "Parent" : "106"},
	{"ID" : "172", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_7_6_U", "Parent" : "106"},
	{"ID" : "173", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_7_5_U", "Parent" : "106"},
	{"ID" : "174", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_7_4_U", "Parent" : "106"},
	{"ID" : "175", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_7_3_U", "Parent" : "106"},
	{"ID" : "176", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_7_2_U", "Parent" : "106"},
	{"ID" : "177", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_7_1_U", "Parent" : "106"},
	{"ID" : "178", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_7_s_U", "Parent" : "106"},
	{"ID" : "179", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_8_8_U", "Parent" : "106"},
	{"ID" : "180", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_8_7_U", "Parent" : "106"},
	{"ID" : "181", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_8_6_U", "Parent" : "106"},
	{"ID" : "182", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_8_5_U", "Parent" : "106"},
	{"ID" : "183", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_8_4_U", "Parent" : "106"},
	{"ID" : "184", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_8_3_U", "Parent" : "106"},
	{"ID" : "185", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_8_2_U", "Parent" : "106"},
	{"ID" : "186", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_8_1_U", "Parent" : "106"},
	{"ID" : "187", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_8_s_U", "Parent" : "106"},
	{"ID" : "188", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_9_8_U", "Parent" : "106"},
	{"ID" : "189", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_9_7_U", "Parent" : "106"},
	{"ID" : "190", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_9_6_U", "Parent" : "106"},
	{"ID" : "191", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_9_5_U", "Parent" : "106"},
	{"ID" : "192", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_9_4_U", "Parent" : "106"},
	{"ID" : "193", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_9_3_U", "Parent" : "106"},
	{"ID" : "194", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_9_2_U", "Parent" : "106"},
	{"ID" : "195", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_9_1_U", "Parent" : "106"},
	{"ID" : "196", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_9_s_U", "Parent" : "106"},
	{"ID" : "197", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_10_8_U", "Parent" : "106"},
	{"ID" : "198", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_10_7_U", "Parent" : "106"},
	{"ID" : "199", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_10_6_U", "Parent" : "106"},
	{"ID" : "200", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_10_5_U", "Parent" : "106"},
	{"ID" : "201", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_10_4_U", "Parent" : "106"},
	{"ID" : "202", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_10_3_U", "Parent" : "106"},
	{"ID" : "203", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_10_2_U", "Parent" : "106"},
	{"ID" : "204", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_10_1_U", "Parent" : "106"},
	{"ID" : "205", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_10_U", "Parent" : "106"},
	{"ID" : "206", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_11_8_U", "Parent" : "106"},
	{"ID" : "207", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_11_7_U", "Parent" : "106"},
	{"ID" : "208", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_11_6_U", "Parent" : "106"},
	{"ID" : "209", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_11_5_U", "Parent" : "106"},
	{"ID" : "210", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_11_4_U", "Parent" : "106"},
	{"ID" : "211", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_11_3_U", "Parent" : "106"},
	{"ID" : "212", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_11_2_U", "Parent" : "106"},
	{"ID" : "213", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_11_1_U", "Parent" : "106"},
	{"ID" : "214", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_11_U", "Parent" : "106"},
	{"ID" : "215", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_12_8_U", "Parent" : "106"},
	{"ID" : "216", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_12_7_U", "Parent" : "106"},
	{"ID" : "217", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_12_6_U", "Parent" : "106"},
	{"ID" : "218", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_12_5_U", "Parent" : "106"},
	{"ID" : "219", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_12_4_U", "Parent" : "106"},
	{"ID" : "220", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_12_3_U", "Parent" : "106"},
	{"ID" : "221", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_12_2_U", "Parent" : "106"},
	{"ID" : "222", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_12_1_U", "Parent" : "106"},
	{"ID" : "223", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_12_U", "Parent" : "106"},
	{"ID" : "224", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_13_8_U", "Parent" : "106"},
	{"ID" : "225", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_13_7_U", "Parent" : "106"},
	{"ID" : "226", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_13_6_U", "Parent" : "106"},
	{"ID" : "227", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_13_5_U", "Parent" : "106"},
	{"ID" : "228", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_13_4_U", "Parent" : "106"},
	{"ID" : "229", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_13_3_U", "Parent" : "106"},
	{"ID" : "230", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_13_2_U", "Parent" : "106"},
	{"ID" : "231", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_13_1_U", "Parent" : "106"},
	{"ID" : "232", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_13_U", "Parent" : "106"},
	{"ID" : "233", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_14_8_U", "Parent" : "106"},
	{"ID" : "234", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_14_7_U", "Parent" : "106"},
	{"ID" : "235", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_14_6_U", "Parent" : "106"},
	{"ID" : "236", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_14_5_U", "Parent" : "106"},
	{"ID" : "237", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_14_4_U", "Parent" : "106"},
	{"ID" : "238", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_14_3_U", "Parent" : "106"},
	{"ID" : "239", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_14_2_U", "Parent" : "106"},
	{"ID" : "240", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_14_1_U", "Parent" : "106"},
	{"ID" : "241", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_14_U", "Parent" : "106"},
	{"ID" : "242", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_15_8_U", "Parent" : "106"},
	{"ID" : "243", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_15_7_U", "Parent" : "106"},
	{"ID" : "244", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_15_6_U", "Parent" : "106"},
	{"ID" : "245", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_15_5_U", "Parent" : "106"},
	{"ID" : "246", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_15_4_U", "Parent" : "106"},
	{"ID" : "247", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_15_3_U", "Parent" : "106"},
	{"ID" : "248", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_15_2_U", "Parent" : "106"},
	{"ID" : "249", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_15_1_U", "Parent" : "106"},
	{"ID" : "250", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.conv_weight_all_V_15_U", "Parent" : "106"},
	{"ID" : "251", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5304", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "252", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5310", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "253", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5316", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "254", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5322", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "255", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5328", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "256", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5334", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "257", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5340", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "258", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5346", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "259", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5352", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "260", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5358", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "261", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5364", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "262", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5370", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "263", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5376", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "264", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5382", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "265", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5388", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "266", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5394", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "267", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5400", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "268", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5406", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "269", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5412", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "270", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5418", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "271", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5424", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "272", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5430", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "273", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5436", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "274", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5442", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "275", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5448", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "276", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5454", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "277", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5460", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "278", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5466", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "279", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5472", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "280", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5478", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "281", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5484", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "282", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5490", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "283", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5496", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "284", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5502", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "285", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5508", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "286", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5514", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "287", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5520", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "288", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5526", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "289", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5532", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "290", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5538", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "291", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5544", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "292", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5550", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "293", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5556", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "294", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5562", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "295", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5568", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "296", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5574", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "297", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5580", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "298", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5586", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "299", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5592", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "300", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5598", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "301", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5604", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "302", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5610", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "303", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5616", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "304", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5622", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "305", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5628", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "306", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5634", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "307", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5640", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "308", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5646", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "309", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5652", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "310", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5658", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "311", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5664", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "312", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5670", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "313", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5676", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "314", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5682", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "315", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5688", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "316", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5694", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "317", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5700", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "318", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5706", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "319", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5712", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "320", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5718", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "321", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5724", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "322", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5730", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "323", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5736", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "324", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5742", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "325", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5748", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "326", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5754", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "327", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5760", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "328", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5766", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "329", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5772", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "330", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5778", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "331", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5784", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "332", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5790", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "333", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5796", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "334", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5802", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "335", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5808", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "336", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5814", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "337", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5820", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "338", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5826", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "339", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5832", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "340", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5838", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "341", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5844", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "342", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5850", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "343", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5856", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "344", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5862", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "345", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5868", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "346", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5874", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "347", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5880", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "348", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5886", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "349", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5892", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "350", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5898", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "351", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5904", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "352", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5910", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "353", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5916", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "354", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5922", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "355", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5928", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "356", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5934", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "357", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5940", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "358", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5946", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "359", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5952", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "360", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5958", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "361", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5964", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "362", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5970", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "363", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5976", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "364", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5982", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "365", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5988", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "366", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_5994", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "367", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_6000", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "368", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_6006", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "369", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_6012", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "370", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_6018", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "371", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_6024", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "372", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_6030", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "373", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_6036", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "374", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_6042", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "375", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_6048", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "376", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_6054", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "377", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_6060", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "378", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_6066", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "379", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_6072", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "380", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_6078", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "381", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_6084", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "382", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_6090", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "383", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_6096", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "384", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_6102", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "385", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_6108", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "386", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_6114", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "387", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_6120", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "388", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_6126", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "389", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_6132", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "390", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_6138", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "391", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_6144", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "392", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_6150", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "393", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_6156", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "394", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.grp_compute_engine_64_fu_6162", "Parent" : "106",
		"CDFG" : "compute_engine_64",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "b_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "w_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "395", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.FracNet_T_mux_336cvx_U3", "Parent" : "106"},
	{"ID" : "396", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.FracNet_T_mux_336cvx_U4", "Parent" : "106"},
	{"ID" : "397", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.FracNet_T_mux_647cwx_U5", "Parent" : "106"},
	{"ID" : "398", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.FracNet_T_mux_647cwx_U6", "Parent" : "106"},
	{"ID" : "399", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.FracNet_T_mux_647cwx_U7", "Parent" : "106"},
	{"ID" : "400", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.FracNet_T_mux_647cwx_U8", "Parent" : "106"},
	{"ID" : "401", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.FracNet_T_mux_647cwx_U9", "Parent" : "106"},
	{"ID" : "402", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.FracNet_T_mux_647cwx_U10", "Parent" : "106"},
	{"ID" : "403", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.FracNet_T_mux_647cwx_U11", "Parent" : "106"},
	{"ID" : "404", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.FracNet_T_mux_647cwx_U12", "Parent" : "106"},
	{"ID" : "405", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.FracNet_T_mux_647cwx_U13", "Parent" : "106"},
	{"ID" : "406", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.FracNet_T_mux_647cwx_U14", "Parent" : "106"},
	{"ID" : "407", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.FracNet_T_mux_647cwx_U15", "Parent" : "106"},
	{"ID" : "408", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.FracNet_T_mux_647cwx_U16", "Parent" : "106"},
	{"ID" : "409", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.FracNet_T_mux_647cwx_U17", "Parent" : "106"},
	{"ID" : "410", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.FracNet_T_mux_647cwx_U18", "Parent" : "106"},
	{"ID" : "411", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.FracNet_T_mux_647cwx_U19", "Parent" : "106"},
	{"ID" : "412", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.FracNet_T_mux_647cwx_U20", "Parent" : "106"},
	{"ID" : "413", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.FracNet_T_mul_mulcxx_U21", "Parent" : "106"},
	{"ID" : "414", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.FracNet_T_mul_mulcxx_U22", "Parent" : "106"},
	{"ID" : "415", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.FracNet_T_mul_mulcxx_U23", "Parent" : "106"},
	{"ID" : "416", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.FracNet_T_mul_mulcxx_U24", "Parent" : "106"},
	{"ID" : "417", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.FracNet_T_mul_mulcxx_U25", "Parent" : "106"},
	{"ID" : "418", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.FracNet_T_mul_mulcxx_U26", "Parent" : "106"},
	{"ID" : "419", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.FracNet_T_mul_mulcxx_U27", "Parent" : "106"},
	{"ID" : "420", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.FracNet_T_mul_mulcxx_U28", "Parent" : "106"},
	{"ID" : "421", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.FracNet_T_mul_mulcxx_U29", "Parent" : "106"},
	{"ID" : "422", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.FracNet_T_mul_mulcxx_U30", "Parent" : "106"},
	{"ID" : "423", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.FracNet_T_mul_mulcxx_U31", "Parent" : "106"},
	{"ID" : "424", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.FracNet_T_mul_mulcxx_U32", "Parent" : "106"},
	{"ID" : "425", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.FracNet_T_mul_mulcxx_U33", "Parent" : "106"},
	{"ID" : "426", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.FracNet_T_mul_mulcxx_U34", "Parent" : "106"},
	{"ID" : "427", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.FracNet_T_mul_mulcxx_U35", "Parent" : "106"},
	{"ID" : "428", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_binary_conv3x3_tile_fu_5968.FracNet_T_mul_mulcxx_U36", "Parent" : "106"},
	{"ID" : "429", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_quant_and_pack_fu_7897", "Parent" : "0", "Child" : ["430", "431", "432", "433", "434", "435", "436", "437", "438", "439", "440", "441", "442", "443", "444", "445", "446", "447", "448", "449", "450", "451", "452", "453", "454", "455", "456", "457", "458", "459", "460", "461"],
		"CDFG" : "quant_and_pack",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "261", "EstimateLatencyMax" : "4101",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "prior_outputs_0_0_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "prior_outputs_0_1_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "prior_outputs_0_2_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "prior_outputs_0_3_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "prior_outputs_0_4_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "prior_outputs_0_5_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "prior_outputs_0_6_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "prior_outputs_0_7_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "prior_outputs_0_8_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "prior_outputs_0_9_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "prior_outputs_0_10_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "prior_outputs_0_11_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "prior_outputs_0_12_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "prior_outputs_0_13_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "prior_outputs_0_14_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "prior_outputs_0_15_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "prior_outputs_1_0_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "prior_outputs_1_1_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "prior_outputs_1_2_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "prior_outputs_1_3_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "prior_outputs_1_4_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "prior_outputs_1_5_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "prior_outputs_1_6_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "prior_outputs_1_7_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "prior_outputs_1_8_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "prior_outputs_1_9_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "prior_outputs_1_10_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "prior_outputs_1_11_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "prior_outputs_1_12_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "prior_outputs_1_13_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "prior_outputs_1_14_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "prior_outputs_1_15_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "prior_outputs_2_0_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "prior_outputs_2_1_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "prior_outputs_2_2_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "prior_outputs_2_3_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "prior_outputs_2_4_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "prior_outputs_2_5_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "prior_outputs_2_6_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "prior_outputs_2_7_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "prior_outputs_2_8_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "prior_outputs_2_9_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "prior_outputs_2_10_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "prior_outputs_2_11_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "prior_outputs_2_12_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "prior_outputs_2_13_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "prior_outputs_2_14_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "prior_outputs_2_15_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "prior_outputs_3_0_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "prior_outputs_3_1_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "prior_outputs_3_2_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "prior_outputs_3_3_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "prior_outputs_3_4_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "prior_outputs_3_5_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "prior_outputs_3_6_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "prior_outputs_3_7_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "prior_outputs_3_8_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "prior_outputs_3_9_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "prior_outputs_3_10_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "prior_outputs_3_11_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "prior_outputs_3_12_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "prior_outputs_3_13_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "prior_outputs_3_14_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "prior_outputs_3_15_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "msb_buffer_0_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "msb_buffer_1_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "H_fmap", "Type" : "None", "Direction" : "I"},
			{"Name" : "in_channels", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "430", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_quant_and_pack_fu_7897.grp_to2bit_fu_1371", "Parent" : "429",
		"CDFG" : "to2bit",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "x_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "431", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_quant_and_pack_fu_7897.grp_to2bit_fu_1377", "Parent" : "429",
		"CDFG" : "to2bit",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "x_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "432", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_quant_and_pack_fu_7897.grp_to2bit_fu_1383", "Parent" : "429",
		"CDFG" : "to2bit",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "x_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "433", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_quant_and_pack_fu_7897.grp_to2bit_fu_1389", "Parent" : "429",
		"CDFG" : "to2bit",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "x_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "434", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_quant_and_pack_fu_7897.grp_to2bit_fu_1395", "Parent" : "429",
		"CDFG" : "to2bit",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "x_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "435", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_quant_and_pack_fu_7897.grp_to2bit_fu_1401", "Parent" : "429",
		"CDFG" : "to2bit",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "x_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "436", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_quant_and_pack_fu_7897.grp_to2bit_fu_1407", "Parent" : "429",
		"CDFG" : "to2bit",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "x_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "437", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_quant_and_pack_fu_7897.grp_to2bit_fu_1413", "Parent" : "429",
		"CDFG" : "to2bit",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "x_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "438", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_quant_and_pack_fu_7897.grp_to2bit_fu_1419", "Parent" : "429",
		"CDFG" : "to2bit",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "x_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "439", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_quant_and_pack_fu_7897.grp_to2bit_fu_1425", "Parent" : "429",
		"CDFG" : "to2bit",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "x_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "440", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_quant_and_pack_fu_7897.grp_to2bit_fu_1431", "Parent" : "429",
		"CDFG" : "to2bit",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "x_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "441", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_quant_and_pack_fu_7897.grp_to2bit_fu_1437", "Parent" : "429",
		"CDFG" : "to2bit",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "x_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "442", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_quant_and_pack_fu_7897.grp_to2bit_fu_1443", "Parent" : "429",
		"CDFG" : "to2bit",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "x_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "443", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_quant_and_pack_fu_7897.grp_to2bit_fu_1449", "Parent" : "429",
		"CDFG" : "to2bit",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "x_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "444", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_quant_and_pack_fu_7897.grp_to2bit_fu_1455", "Parent" : "429",
		"CDFG" : "to2bit",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "x_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "445", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_quant_and_pack_fu_7897.grp_to2bit_fu_1461", "Parent" : "429",
		"CDFG" : "to2bit",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "x_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "446", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_quant_and_pack_fu_7897.FracNet_T_mux_42_cyx_U320", "Parent" : "429"},
	{"ID" : "447", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_quant_and_pack_fu_7897.FracNet_T_mux_42_cyx_U321", "Parent" : "429"},
	{"ID" : "448", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_quant_and_pack_fu_7897.FracNet_T_mux_42_cyx_U322", "Parent" : "429"},
	{"ID" : "449", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_quant_and_pack_fu_7897.FracNet_T_mux_42_cyx_U323", "Parent" : "429"},
	{"ID" : "450", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_quant_and_pack_fu_7897.FracNet_T_mux_42_cyx_U324", "Parent" : "429"},
	{"ID" : "451", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_quant_and_pack_fu_7897.FracNet_T_mux_42_cyx_U325", "Parent" : "429"},
	{"ID" : "452", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_quant_and_pack_fu_7897.FracNet_T_mux_42_cyx_U326", "Parent" : "429"},
	{"ID" : "453", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_quant_and_pack_fu_7897.FracNet_T_mux_42_cyx_U327", "Parent" : "429"},
	{"ID" : "454", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_quant_and_pack_fu_7897.FracNet_T_mux_42_cyx_U328", "Parent" : "429"},
	{"ID" : "455", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_quant_and_pack_fu_7897.FracNet_T_mux_42_cyx_U329", "Parent" : "429"},
	{"ID" : "456", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_quant_and_pack_fu_7897.FracNet_T_mux_42_cyx_U330", "Parent" : "429"},
	{"ID" : "457", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_quant_and_pack_fu_7897.FracNet_T_mux_42_cyx_U331", "Parent" : "429"},
	{"ID" : "458", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_quant_and_pack_fu_7897.FracNet_T_mux_42_cyx_U332", "Parent" : "429"},
	{"ID" : "459", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_quant_and_pack_fu_7897.FracNet_T_mux_42_cyx_U333", "Parent" : "429"},
	{"ID" : "460", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_quant_and_pack_fu_7897.FracNet_T_mux_42_cyx_U334", "Parent" : "429"},
	{"ID" : "461", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_quant_and_pack_fu_7897.FracNet_T_mux_42_cyx_U335", "Parent" : "429"},
	{"ID" : "462", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106", "Parent" : "0", "Child" : ["463", "464", "465", "466", "467", "468", "469", "470", "471", "472", "473", "474", "475", "476", "477", "478", "479", "480", "481", "482", "483", "484", "485", "486", "487", "488", "489", "490", "491", "492", "493", "494", "495", "496", "497", "498", "499", "500", "501", "502", "503", "504", "505", "506", "507", "508", "509", "510", "511", "512", "513", "514", "515", "516", "517", "518", "519", "520", "521", "522", "523", "524", "525", "526", "527", "528", "529", "530", "531", "532", "533", "534", "535", "536", "537", "538", "539", "540", "541", "542", "543", "544", "545", "546", "547", "548", "549", "550", "551", "552", "553", "554", "555", "556", "557", "558", "559", "560", "561", "562", "563", "564", "565", "566", "567", "568", "569", "570", "571", "572", "573", "574", "575", "576", "577", "578", "579", "580", "581", "582", "583", "584", "585", "586", "587", "588", "589", "590", "591", "592", "593", "594", "595", "596", "597", "598", "599", "600", "601", "602", "603", "604", "605", "606", "607", "608", "609", "610", "611", "612", "613", "614", "615", "616", "617", "618", "619", "620", "621", "622", "623", "624", "625", "626", "627", "628", "629", "630", "631", "632", "633", "634", "635", "636", "637", "638", "639", "640", "641", "642", "643", "644", "645", "646", "647", "648", "649", "650", "651", "652", "653", "654", "655", "656", "657", "658", "659", "660", "661", "662", "663", "664", "665", "666", "667", "668", "669", "670", "671", "672"],
		"CDFG" : "bn_relu_shortcut",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "82", "EstimateLatencyMax" : "1042",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "residual_0_0_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "residual_0_1_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "residual_0_2_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "residual_0_3_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "residual_0_4_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "residual_0_5_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "residual_0_6_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "residual_0_7_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "residual_0_8_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "residual_0_9_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "residual_0_10_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "residual_0_11_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "residual_0_12_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "residual_0_13_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "residual_0_14_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "residual_0_15_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "residual_1_0_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "residual_1_1_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "residual_1_2_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "residual_1_3_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "residual_1_4_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "residual_1_5_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "residual_1_6_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "residual_1_7_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "residual_1_8_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "residual_1_9_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "residual_1_10_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "residual_1_11_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "residual_1_12_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "residual_1_13_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "residual_1_14_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "residual_1_15_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "residual_2_0_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "residual_2_1_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "residual_2_2_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "residual_2_3_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "residual_2_4_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "residual_2_5_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "residual_2_6_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "residual_2_7_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "residual_2_8_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "residual_2_9_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "residual_2_10_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "residual_2_11_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "residual_2_12_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "residual_2_13_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "residual_2_14_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "residual_2_15_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "residual_3_0_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "residual_3_1_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "residual_3_2_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "residual_3_3_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "residual_3_4_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "residual_3_5_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "residual_3_6_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "residual_3_7_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "residual_3_8_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "residual_3_9_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "residual_3_10_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "residual_3_11_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "residual_3_12_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "residual_3_13_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "residual_3_14_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "residual_3_15_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "block_t0_0_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "block_t0_1_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "block_t0_2_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "block_t0_3_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "block_t0_4_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "block_t0_5_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "block_t0_6_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "block_t0_7_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "block_t0_8_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "block_t0_9_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "block_t0_10_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "block_t0_11_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "block_t0_12_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "block_t0_13_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "block_t0_14_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "block_t0_15_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "block_t1_0_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "block_t1_1_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "block_t1_2_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "block_t1_3_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "block_t1_4_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "block_t1_5_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "block_t1_6_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "block_t1_7_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "block_t1_8_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "block_t1_9_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "block_t1_10_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "block_t1_11_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "block_t1_12_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "block_t1_13_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "block_t1_14_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "block_t1_15_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "bn_weight_0_0_0_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_0_1_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_0_2_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_0_3_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_1_0_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_1_1_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_1_2_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_1_3_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_2_0_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_2_1_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_2_2_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_2_3_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_3_0_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_3_1_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_3_2_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_3_3_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_4_0_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_4_1_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_4_2_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_4_3_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_5_0_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_5_1_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_5_2_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_5_3_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_6_0_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_6_1_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_6_2_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_6_3_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_7_0_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_7_1_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_7_2_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_7_3_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_8_0_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_8_1_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_8_2_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_8_3_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_9_0_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_9_1_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_9_2_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_9_3_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_10_0_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_10_1_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_10_2_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_10_3_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_11_0_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_11_1_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_11_2_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_11_3_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_12_0_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_12_1_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_12_2_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_12_3_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_13_0_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_13_1_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_13_2_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_13_3_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_14_0_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_14_1_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_14_2_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_14_3_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_15_0_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_15_1_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_15_2_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_15_3_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_0_V_offset", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_0_0_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_0_1_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_0_2_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_0_3_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_1_0_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_1_1_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_1_2_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_1_3_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_2_0_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_2_1_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_2_2_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_2_3_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_3_0_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_3_1_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_3_2_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_3_3_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_4_0_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_4_1_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_4_2_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_4_3_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_5_0_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_5_1_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_5_2_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_5_3_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_6_0_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_6_1_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_6_2_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_6_3_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_7_0_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_7_1_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_7_2_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_7_3_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_8_0_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_8_1_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_8_2_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_8_3_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_9_0_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_9_1_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_9_2_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_9_3_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_10_0_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_10_1_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_10_2_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_10_3_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_11_0_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_11_1_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_11_2_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_11_3_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_12_0_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_12_1_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_12_2_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_12_3_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_13_0_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_13_1_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_13_2_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_13_3_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_14_0_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_14_1_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_14_2_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_14_3_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_15_0_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_15_1_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_15_2_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_15_3_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_weight_1_V_offset", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_0_0_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_0_1_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_0_2_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_0_3_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_1_0_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_1_1_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_1_2_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_1_3_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_2_0_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_2_1_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_2_2_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_2_3_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_3_0_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_3_1_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_3_2_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_3_3_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_4_0_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_4_1_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_4_2_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_4_3_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_5_0_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_5_1_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_5_2_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_5_3_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_6_0_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_6_1_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_6_2_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_6_3_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_7_0_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_7_1_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_7_2_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_7_3_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_8_0_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_8_1_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_8_2_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_8_3_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_9_0_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_9_1_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_9_2_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_9_3_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_10_0_V_r", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_10_1_V_r", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_10_2_V_r", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_10_3_V_r", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_11_0_V_r", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_11_1_V_r", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_11_2_V_r", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_11_3_V_r", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_12_0_V_r", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_12_1_V_r", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_12_2_V_r", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_12_3_V_r", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_13_0_V_r", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_13_1_V_r", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_13_2_V_r", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_13_3_V_r", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_14_0_V_r", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_14_1_V_r", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_14_2_V_r", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_14_3_V_r", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_15_0_V_r", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_15_1_V_r", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_15_2_V_r", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_15_3_V_r", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_0_V_offset", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_0_0_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_0_1_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_0_2_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_0_3_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_1_0_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_1_1_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_1_2_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_1_3_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_2_0_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_2_1_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_2_2_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_2_3_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_3_0_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_3_1_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_3_2_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_3_3_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_4_0_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_4_1_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_4_2_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_4_3_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_5_0_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_5_1_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_5_2_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_5_3_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_6_0_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_6_1_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_6_2_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_6_3_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_7_0_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_7_1_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_7_2_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_7_3_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_8_0_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_8_1_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_8_2_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_8_3_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_9_0_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_9_1_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_9_2_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_9_3_V_re", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_10_0_V_r", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_10_1_V_r", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_10_2_V_r", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_10_3_V_r", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_11_0_V_r", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_11_1_V_r", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_11_2_V_r", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_11_3_V_r", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_12_0_V_r", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_12_1_V_r", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_12_2_V_r", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_12_3_V_r", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_13_0_V_r", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_13_1_V_r", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_13_2_V_r", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_13_3_V_r", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_14_0_V_r", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_14_1_V_r", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_14_2_V_r", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_14_3_V_r", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_15_0_V_r", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_15_1_V_r", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_15_2_V_r", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_15_3_V_r", "Type" : "None", "Direction" : "I"},
			{"Name" : "bn_bias_1_V_offset", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_0_0_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_0_1_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_0_2_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_0_3_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_1_0_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_1_1_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_1_2_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_1_3_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_2_0_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_2_1_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_2_2_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_2_3_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_3_0_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_3_1_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_3_2_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_3_3_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_4_0_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_4_1_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_4_2_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_4_3_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_5_0_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_5_1_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_5_2_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_5_3_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_6_0_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_6_1_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_6_2_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_6_3_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_7_0_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_7_1_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_7_2_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_7_3_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_8_0_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_8_1_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_8_2_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_8_3_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_9_0_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_9_1_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_9_2_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_9_3_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_10_0_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_10_1_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_10_2_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_10_3_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_11_0_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_11_1_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_11_2_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_11_3_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_12_0_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_12_1_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_12_2_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_12_3_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_13_0_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_13_1_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_13_2_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_13_3_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_14_0_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_14_1_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_14_2_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_14_3_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_15_0_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_15_1_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_15_2_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_15_3_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_x_bias_V_offset", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_0_0_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_0_1_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_0_2_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_0_3_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_1_0_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_1_1_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_1_2_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_1_3_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_2_0_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_2_1_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_2_2_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_2_3_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_3_0_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_3_1_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_3_2_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_3_3_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_4_0_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_4_1_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_4_2_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_4_3_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_5_0_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_5_1_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_5_2_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_5_3_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_6_0_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_6_1_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_6_2_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_6_3_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_7_0_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_7_1_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_7_2_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_7_3_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_8_0_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_8_1_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_8_2_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_8_3_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_9_0_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_9_1_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_9_2_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_9_3_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_10_0_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_10_1_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_10_2_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_10_3_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_11_0_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_11_1_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_11_2_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_11_3_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_12_0_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_12_1_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_12_2_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_12_3_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_13_0_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_13_1_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_13_2_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_13_3_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_14_0_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_14_1_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_14_2_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_14_3_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_15_0_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_15_1_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_15_2_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_15_3_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_y_bias_V_offset", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_0_0_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_0_1_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_0_2_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_0_3_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_1_0_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_1_1_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_1_2_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_1_3_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_2_0_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_2_1_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_2_2_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_2_3_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_3_0_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_3_1_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_3_2_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_3_3_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_4_0_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_4_1_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_4_2_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_4_3_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_5_0_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_5_1_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_5_2_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_5_3_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_6_0_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_6_1_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_6_2_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_6_3_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_7_0_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_7_1_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_7_2_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_7_3_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_8_0_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_8_1_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_8_2_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_8_3_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_9_0_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_9_1_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_9_2_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_9_3_V_s", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_10_0_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_10_1_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_10_2_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_10_3_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_11_0_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_11_1_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_11_2_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_11_3_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_12_0_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_12_1_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_12_2_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_12_3_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_13_0_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_13_1_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_13_2_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_13_3_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_14_0_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_14_1_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_14_2_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_14_3_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_15_0_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_15_1_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_15_2_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_15_3_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "relu_weight_V_offset", "Type" : "None", "Direction" : "I"},
			{"Name" : "stride", "Type" : "None", "Direction" : "I"},
			{"Name" : "channel_tile", "Type" : "None", "Direction" : "I"},
			{"Name" : "H_fmap", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "463", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_cyx_U405", "Parent" : "462"},
	{"ID" : "464", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_cyx_U406", "Parent" : "462"},
	{"ID" : "465", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_cyx_U407", "Parent" : "462"},
	{"ID" : "466", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_cyx_U408", "Parent" : "462"},
	{"ID" : "467", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_cyx_U409", "Parent" : "462"},
	{"ID" : "468", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_cyx_U410", "Parent" : "462"},
	{"ID" : "469", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_cyx_U411", "Parent" : "462"},
	{"ID" : "470", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_cyx_U412", "Parent" : "462"},
	{"ID" : "471", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_cyx_U413", "Parent" : "462"},
	{"ID" : "472", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_cyx_U414", "Parent" : "462"},
	{"ID" : "473", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_cyx_U415", "Parent" : "462"},
	{"ID" : "474", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_cyx_U416", "Parent" : "462"},
	{"ID" : "475", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_cyx_U417", "Parent" : "462"},
	{"ID" : "476", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_cyx_U418", "Parent" : "462"},
	{"ID" : "477", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_cyx_U419", "Parent" : "462"},
	{"ID" : "478", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_cyx_U420", "Parent" : "462"},
	{"ID" : "479", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U421", "Parent" : "462"},
	{"ID" : "480", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U422", "Parent" : "462"},
	{"ID" : "481", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U423", "Parent" : "462"},
	{"ID" : "482", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U424", "Parent" : "462"},
	{"ID" : "483", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U425", "Parent" : "462"},
	{"ID" : "484", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U426", "Parent" : "462"},
	{"ID" : "485", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U427", "Parent" : "462"},
	{"ID" : "486", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U428", "Parent" : "462"},
	{"ID" : "487", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U429", "Parent" : "462"},
	{"ID" : "488", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U430", "Parent" : "462"},
	{"ID" : "489", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U431", "Parent" : "462"},
	{"ID" : "490", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U432", "Parent" : "462"},
	{"ID" : "491", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U433", "Parent" : "462"},
	{"ID" : "492", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U434", "Parent" : "462"},
	{"ID" : "493", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U435", "Parent" : "462"},
	{"ID" : "494", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U436", "Parent" : "462"},
	{"ID" : "495", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U437", "Parent" : "462"},
	{"ID" : "496", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U438", "Parent" : "462"},
	{"ID" : "497", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U439", "Parent" : "462"},
	{"ID" : "498", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U440", "Parent" : "462"},
	{"ID" : "499", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U441", "Parent" : "462"},
	{"ID" : "500", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U442", "Parent" : "462"},
	{"ID" : "501", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U443", "Parent" : "462"},
	{"ID" : "502", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U444", "Parent" : "462"},
	{"ID" : "503", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U445", "Parent" : "462"},
	{"ID" : "504", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U446", "Parent" : "462"},
	{"ID" : "505", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U447", "Parent" : "462"},
	{"ID" : "506", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U448", "Parent" : "462"},
	{"ID" : "507", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U449", "Parent" : "462"},
	{"ID" : "508", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U450", "Parent" : "462"},
	{"ID" : "509", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U451", "Parent" : "462"},
	{"ID" : "510", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U452", "Parent" : "462"},
	{"ID" : "511", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U453", "Parent" : "462"},
	{"ID" : "512", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U454", "Parent" : "462"},
	{"ID" : "513", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U455", "Parent" : "462"},
	{"ID" : "514", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U456", "Parent" : "462"},
	{"ID" : "515", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U457", "Parent" : "462"},
	{"ID" : "516", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U458", "Parent" : "462"},
	{"ID" : "517", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U459", "Parent" : "462"},
	{"ID" : "518", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U460", "Parent" : "462"},
	{"ID" : "519", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U461", "Parent" : "462"},
	{"ID" : "520", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U462", "Parent" : "462"},
	{"ID" : "521", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U463", "Parent" : "462"},
	{"ID" : "522", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U464", "Parent" : "462"},
	{"ID" : "523", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U465", "Parent" : "462"},
	{"ID" : "524", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U466", "Parent" : "462"},
	{"ID" : "525", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U467", "Parent" : "462"},
	{"ID" : "526", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U468", "Parent" : "462"},
	{"ID" : "527", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U469", "Parent" : "462"},
	{"ID" : "528", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U470", "Parent" : "462"},
	{"ID" : "529", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U471", "Parent" : "462"},
	{"ID" : "530", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U472", "Parent" : "462"},
	{"ID" : "531", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U473", "Parent" : "462"},
	{"ID" : "532", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U474", "Parent" : "462"},
	{"ID" : "533", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U475", "Parent" : "462"},
	{"ID" : "534", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U476", "Parent" : "462"},
	{"ID" : "535", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U477", "Parent" : "462"},
	{"ID" : "536", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U478", "Parent" : "462"},
	{"ID" : "537", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U479", "Parent" : "462"},
	{"ID" : "538", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U480", "Parent" : "462"},
	{"ID" : "539", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U481", "Parent" : "462"},
	{"ID" : "540", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U482", "Parent" : "462"},
	{"ID" : "541", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U483", "Parent" : "462"},
	{"ID" : "542", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U484", "Parent" : "462"},
	{"ID" : "543", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U485", "Parent" : "462"},
	{"ID" : "544", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U486", "Parent" : "462"},
	{"ID" : "545", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U487", "Parent" : "462"},
	{"ID" : "546", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U488", "Parent" : "462"},
	{"ID" : "547", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U489", "Parent" : "462"},
	{"ID" : "548", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U490", "Parent" : "462"},
	{"ID" : "549", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U491", "Parent" : "462"},
	{"ID" : "550", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U492", "Parent" : "462"},
	{"ID" : "551", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U493", "Parent" : "462"},
	{"ID" : "552", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U494", "Parent" : "462"},
	{"ID" : "553", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U495", "Parent" : "462"},
	{"ID" : "554", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U496", "Parent" : "462"},
	{"ID" : "555", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U497", "Parent" : "462"},
	{"ID" : "556", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U498", "Parent" : "462"},
	{"ID" : "557", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U499", "Parent" : "462"},
	{"ID" : "558", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U500", "Parent" : "462"},
	{"ID" : "559", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U501", "Parent" : "462"},
	{"ID" : "560", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U502", "Parent" : "462"},
	{"ID" : "561", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U503", "Parent" : "462"},
	{"ID" : "562", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U504", "Parent" : "462"},
	{"ID" : "563", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U505", "Parent" : "462"},
	{"ID" : "564", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U506", "Parent" : "462"},
	{"ID" : "565", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U507", "Parent" : "462"},
	{"ID" : "566", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U508", "Parent" : "462"},
	{"ID" : "567", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U509", "Parent" : "462"},
	{"ID" : "568", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U510", "Parent" : "462"},
	{"ID" : "569", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U511", "Parent" : "462"},
	{"ID" : "570", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U512", "Parent" : "462"},
	{"ID" : "571", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U513", "Parent" : "462"},
	{"ID" : "572", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U514", "Parent" : "462"},
	{"ID" : "573", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U515", "Parent" : "462"},
	{"ID" : "574", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U516", "Parent" : "462"},
	{"ID" : "575", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U517", "Parent" : "462"},
	{"ID" : "576", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U518", "Parent" : "462"},
	{"ID" : "577", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U519", "Parent" : "462"},
	{"ID" : "578", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U520", "Parent" : "462"},
	{"ID" : "579", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U521", "Parent" : "462"},
	{"ID" : "580", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U522", "Parent" : "462"},
	{"ID" : "581", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U523", "Parent" : "462"},
	{"ID" : "582", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U524", "Parent" : "462"},
	{"ID" : "583", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U525", "Parent" : "462"},
	{"ID" : "584", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U526", "Parent" : "462"},
	{"ID" : "585", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U527", "Parent" : "462"},
	{"ID" : "586", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U528", "Parent" : "462"},
	{"ID" : "587", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U529", "Parent" : "462"},
	{"ID" : "588", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U530", "Parent" : "462"},
	{"ID" : "589", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U531", "Parent" : "462"},
	{"ID" : "590", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mux_42_czy_U532", "Parent" : "462"},
	{"ID" : "591", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mac_mulcAy_U533", "Parent" : "462"},
	{"ID" : "592", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mac_mulcBy_U534", "Parent" : "462"},
	{"ID" : "593", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mul_mulcCy_U535", "Parent" : "462"},
	{"ID" : "594", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mul_mulcCy_U536", "Parent" : "462"},
	{"ID" : "595", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mul_mulcCy_U537", "Parent" : "462"},
	{"ID" : "596", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mul_mulcCy_U538", "Parent" : "462"},
	{"ID" : "597", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mul_mulcCy_U539", "Parent" : "462"},
	{"ID" : "598", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mul_mulcCy_U540", "Parent" : "462"},
	{"ID" : "599", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mul_mulcCy_U541", "Parent" : "462"},
	{"ID" : "600", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mul_mulcCy_U542", "Parent" : "462"},
	{"ID" : "601", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mul_mulcCy_U543", "Parent" : "462"},
	{"ID" : "602", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mul_mulcCy_U544", "Parent" : "462"},
	{"ID" : "603", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mul_mulcCy_U545", "Parent" : "462"},
	{"ID" : "604", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mul_mulcCy_U546", "Parent" : "462"},
	{"ID" : "605", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mul_mulcCy_U547", "Parent" : "462"},
	{"ID" : "606", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mul_mulcCy_U548", "Parent" : "462"},
	{"ID" : "607", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mul_mulcCy_U549", "Parent" : "462"},
	{"ID" : "608", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mul_mulcCy_U550", "Parent" : "462"},
	{"ID" : "609", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mul_mulcDy_U551", "Parent" : "462"},
	{"ID" : "610", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mul_mulcDy_U552", "Parent" : "462"},
	{"ID" : "611", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mul_mulcDy_U553", "Parent" : "462"},
	{"ID" : "612", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mul_mulcDy_U554", "Parent" : "462"},
	{"ID" : "613", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mul_mulcDy_U555", "Parent" : "462"},
	{"ID" : "614", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mul_mulcDy_U556", "Parent" : "462"},
	{"ID" : "615", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mul_mulcDy_U557", "Parent" : "462"},
	{"ID" : "616", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mul_mulcDy_U558", "Parent" : "462"},
	{"ID" : "617", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mul_mulcDy_U559", "Parent" : "462"},
	{"ID" : "618", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mul_mulcDy_U560", "Parent" : "462"},
	{"ID" : "619", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mul_mulcDy_U561", "Parent" : "462"},
	{"ID" : "620", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mul_mulcDy_U562", "Parent" : "462"},
	{"ID" : "621", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mul_mulcDy_U563", "Parent" : "462"},
	{"ID" : "622", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mul_mulcDy_U564", "Parent" : "462"},
	{"ID" : "623", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mul_mulcDy_U565", "Parent" : "462"},
	{"ID" : "624", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mul_mulcDy_U566", "Parent" : "462"},
	{"ID" : "625", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mac_mulcEy_U567", "Parent" : "462"},
	{"ID" : "626", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mac_mulcEy_U568", "Parent" : "462"},
	{"ID" : "627", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mac_mulcEy_U569", "Parent" : "462"},
	{"ID" : "628", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mac_mulcEy_U570", "Parent" : "462"},
	{"ID" : "629", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mac_mulcEy_U571", "Parent" : "462"},
	{"ID" : "630", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mac_mulcEy_U572", "Parent" : "462"},
	{"ID" : "631", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mac_mulcEy_U573", "Parent" : "462"},
	{"ID" : "632", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mac_mulcEy_U574", "Parent" : "462"},
	{"ID" : "633", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mac_mulcEy_U575", "Parent" : "462"},
	{"ID" : "634", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mac_mulcEy_U576", "Parent" : "462"},
	{"ID" : "635", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mac_mulcEy_U577", "Parent" : "462"},
	{"ID" : "636", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mac_mulcEy_U578", "Parent" : "462"},
	{"ID" : "637", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mac_mulcEy_U579", "Parent" : "462"},
	{"ID" : "638", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mac_mulcEy_U580", "Parent" : "462"},
	{"ID" : "639", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mac_mulcEy_U581", "Parent" : "462"},
	{"ID" : "640", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mac_mulcEy_U582", "Parent" : "462"},
	{"ID" : "641", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mul_mulcFz_U583", "Parent" : "462"},
	{"ID" : "642", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mul_mulcFz_U584", "Parent" : "462"},
	{"ID" : "643", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mul_mulcFz_U585", "Parent" : "462"},
	{"ID" : "644", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mul_mulcFz_U586", "Parent" : "462"},
	{"ID" : "645", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mul_mulcFz_U587", "Parent" : "462"},
	{"ID" : "646", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mul_mulcFz_U588", "Parent" : "462"},
	{"ID" : "647", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mul_mulcFz_U589", "Parent" : "462"},
	{"ID" : "648", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mul_mulcFz_U590", "Parent" : "462"},
	{"ID" : "649", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mul_mulcFz_U591", "Parent" : "462"},
	{"ID" : "650", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mul_mulcFz_U592", "Parent" : "462"},
	{"ID" : "651", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mul_mulcFz_U593", "Parent" : "462"},
	{"ID" : "652", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mul_mulcFz_U594", "Parent" : "462"},
	{"ID" : "653", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mul_mulcFz_U595", "Parent" : "462"},
	{"ID" : "654", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mul_mulcFz_U596", "Parent" : "462"},
	{"ID" : "655", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mul_mulcFz_U597", "Parent" : "462"},
	{"ID" : "656", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mul_mulcFz_U598", "Parent" : "462"},
	{"ID" : "657", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mac_mulcEy_U599", "Parent" : "462"},
	{"ID" : "658", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mac_mulcGz_U600", "Parent" : "462"},
	{"ID" : "659", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mac_mulcGz_U601", "Parent" : "462"},
	{"ID" : "660", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mac_mulcGz_U602", "Parent" : "462"},
	{"ID" : "661", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mac_mulcGz_U603", "Parent" : "462"},
	{"ID" : "662", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mac_mulcGz_U604", "Parent" : "462"},
	{"ID" : "663", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mac_mulcGz_U605", "Parent" : "462"},
	{"ID" : "664", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mac_mulcGz_U606", "Parent" : "462"},
	{"ID" : "665", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mac_mulcGz_U607", "Parent" : "462"},
	{"ID" : "666", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mac_mulcGz_U608", "Parent" : "462"},
	{"ID" : "667", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mac_mulcGz_U609", "Parent" : "462"},
	{"ID" : "668", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mac_mulcGz_U610", "Parent" : "462"},
	{"ID" : "669", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mac_mulcGz_U611", "Parent" : "462"},
	{"ID" : "670", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mac_mulcGz_U612", "Parent" : "462"},
	{"ID" : "671", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mac_mulcGz_U613", "Parent" : "462"},
	{"ID" : "672", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_relu_shortcut_fu_8106.FracNet_T_mac_mulcGz_U614", "Parent" : "462"},
	{"ID" : "673", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_avgpool_concat_fu_18852", "Parent" : "0", "Child" : ["674", "675", "676", "677", "678", "679", "680", "681", "682", "683", "684", "685", "686", "687", "688", "689", "690", "691", "692", "693", "694", "695", "696", "697", "698", "699", "700", "701", "702", "703", "704", "705"],
		"CDFG" : "avgpool_concat",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "590", "EstimateLatencyMax" : "2841",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "outputs_0_0_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "outputs_0_1_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "outputs_0_2_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "outputs_0_3_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "outputs_0_4_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "outputs_0_5_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "outputs_0_6_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "outputs_0_7_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "outputs_0_8_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "outputs_0_9_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "outputs_0_10_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "outputs_0_11_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "outputs_0_12_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "outputs_0_13_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "outputs_0_14_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "outputs_0_15_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "outputs_1_0_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "outputs_1_1_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "outputs_1_2_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "outputs_1_3_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "outputs_1_4_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "outputs_1_5_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "outputs_1_6_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "outputs_1_7_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "outputs_1_8_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "outputs_1_9_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "outputs_1_10_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "outputs_1_11_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "outputs_1_12_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "outputs_1_13_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "outputs_1_14_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "outputs_1_15_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "outputs_2_0_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "outputs_2_1_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "outputs_2_2_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "outputs_2_3_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "outputs_2_4_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "outputs_2_5_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "outputs_2_6_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "outputs_2_7_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "outputs_2_8_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "outputs_2_9_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "outputs_2_10_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "outputs_2_11_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "outputs_2_12_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "outputs_2_13_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "outputs_2_14_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "outputs_2_15_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "outputs_3_0_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "outputs_3_1_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "outputs_3_2_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "outputs_3_3_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "outputs_3_4_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "outputs_3_5_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "outputs_3_6_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "outputs_3_7_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "outputs_3_8_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "outputs_3_9_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "outputs_3_10_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "outputs_3_11_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "outputs_3_12_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "outputs_3_13_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "outputs_3_14_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "outputs_3_15_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "H_fmap", "Type" : "None", "Direction" : "I"},
			{"Name" : "in_channels", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "674", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_avgpool_concat_fu_18852.out_tmp_0_V_U", "Parent" : "673"},
	{"ID" : "675", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_avgpool_concat_fu_18852.out_tmp_1_V_U", "Parent" : "673"},
	{"ID" : "676", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_avgpool_concat_fu_18852.out_tmp_2_V_U", "Parent" : "673"},
	{"ID" : "677", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_avgpool_concat_fu_18852.out_tmp_3_V_U", "Parent" : "673"},
	{"ID" : "678", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_avgpool_concat_fu_18852.out_tmp_4_V_U", "Parent" : "673"},
	{"ID" : "679", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_avgpool_concat_fu_18852.out_tmp_5_V_U", "Parent" : "673"},
	{"ID" : "680", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_avgpool_concat_fu_18852.out_tmp_6_V_U", "Parent" : "673"},
	{"ID" : "681", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_avgpool_concat_fu_18852.out_tmp_7_V_U", "Parent" : "673"},
	{"ID" : "682", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_avgpool_concat_fu_18852.out_tmp_8_V_U", "Parent" : "673"},
	{"ID" : "683", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_avgpool_concat_fu_18852.out_tmp_9_V_U", "Parent" : "673"},
	{"ID" : "684", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_avgpool_concat_fu_18852.out_tmp_10_V_U", "Parent" : "673"},
	{"ID" : "685", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_avgpool_concat_fu_18852.out_tmp_11_V_U", "Parent" : "673"},
	{"ID" : "686", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_avgpool_concat_fu_18852.out_tmp_12_V_U", "Parent" : "673"},
	{"ID" : "687", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_avgpool_concat_fu_18852.out_tmp_13_V_U", "Parent" : "673"},
	{"ID" : "688", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_avgpool_concat_fu_18852.out_tmp_14_V_U", "Parent" : "673"},
	{"ID" : "689", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_avgpool_concat_fu_18852.out_tmp_15_V_U", "Parent" : "673"},
	{"ID" : "690", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_avgpool_concat_fu_18852.FracNet_T_mux_42_cyx_U1179", "Parent" : "673"},
	{"ID" : "691", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_avgpool_concat_fu_18852.FracNet_T_mux_42_cyx_U1180", "Parent" : "673"},
	{"ID" : "692", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_avgpool_concat_fu_18852.FracNet_T_mux_42_cyx_U1181", "Parent" : "673"},
	{"ID" : "693", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_avgpool_concat_fu_18852.FracNet_T_mux_42_cyx_U1182", "Parent" : "673"},
	{"ID" : "694", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_avgpool_concat_fu_18852.FracNet_T_mux_42_cyx_U1183", "Parent" : "673"},
	{"ID" : "695", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_avgpool_concat_fu_18852.FracNet_T_mux_42_cyx_U1184", "Parent" : "673"},
	{"ID" : "696", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_avgpool_concat_fu_18852.FracNet_T_mux_42_cyx_U1185", "Parent" : "673"},
	{"ID" : "697", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_avgpool_concat_fu_18852.FracNet_T_mux_42_cyx_U1186", "Parent" : "673"},
	{"ID" : "698", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_avgpool_concat_fu_18852.FracNet_T_mux_42_cyx_U1187", "Parent" : "673"},
	{"ID" : "699", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_avgpool_concat_fu_18852.FracNet_T_mux_42_cyx_U1188", "Parent" : "673"},
	{"ID" : "700", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_avgpool_concat_fu_18852.FracNet_T_mux_42_cyx_U1189", "Parent" : "673"},
	{"ID" : "701", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_avgpool_concat_fu_18852.FracNet_T_mux_42_cyx_U1190", "Parent" : "673"},
	{"ID" : "702", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_avgpool_concat_fu_18852.FracNet_T_mux_42_cyx_U1191", "Parent" : "673"},
	{"ID" : "703", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_avgpool_concat_fu_18852.FracNet_T_mux_42_cyx_U1192", "Parent" : "673"},
	{"ID" : "704", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_avgpool_concat_fu_18852.FracNet_T_mux_42_cyx_U1193", "Parent" : "673"},
	{"ID" : "705", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_avgpool_concat_fu_18852.FracNet_T_mux_42_cyx_U1194", "Parent" : "673"},
	{"ID" : "706", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_avgpool_8x8_fu_18926", "Parent" : "0", "Child" : ["707", "708", "709", "710", "711", "712", "713", "714", "715", "716", "717", "718", "719", "720", "721", "722", "723"],
		"CDFG" : "avgpool_8x8",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "425", "EstimateLatencyMax" : "425",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "inputs_0_0_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inputs_0_1_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inputs_0_2_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inputs_0_3_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inputs_0_4_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inputs_0_5_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inputs_0_6_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inputs_0_7_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inputs_0_8_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inputs_0_9_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inputs_0_10_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inputs_0_11_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inputs_0_12_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inputs_0_13_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inputs_0_14_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inputs_0_15_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inputs_1_0_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inputs_1_1_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inputs_1_2_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inputs_1_3_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inputs_1_4_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inputs_1_5_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inputs_1_6_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inputs_1_7_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inputs_1_8_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inputs_1_9_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inputs_1_10_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inputs_1_11_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inputs_1_12_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inputs_1_13_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inputs_1_14_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inputs_1_15_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inputs_2_0_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inputs_2_1_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inputs_2_2_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inputs_2_3_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inputs_2_4_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inputs_2_5_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inputs_2_6_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inputs_2_7_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inputs_2_8_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inputs_2_9_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inputs_2_10_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inputs_2_11_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inputs_2_12_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inputs_2_13_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inputs_2_14_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inputs_2_15_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inputs_3_0_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inputs_3_1_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inputs_3_2_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inputs_3_3_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inputs_3_4_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inputs_3_5_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inputs_3_6_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inputs_3_7_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inputs_3_8_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inputs_3_9_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inputs_3_10_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inputs_3_11_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inputs_3_12_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inputs_3_13_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inputs_3_14_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inputs_3_15_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "outputs_V", "Type" : "Memory", "Direction" : "O"}]},
	{"ID" : "707", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_avgpool_8x8_fu_18926.FracNet_T_mux_42_cyx_U1262", "Parent" : "706"},
	{"ID" : "708", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_avgpool_8x8_fu_18926.FracNet_T_mux_42_cyx_U1263", "Parent" : "706"},
	{"ID" : "709", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_avgpool_8x8_fu_18926.FracNet_T_mux_42_cyx_U1264", "Parent" : "706"},
	{"ID" : "710", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_avgpool_8x8_fu_18926.FracNet_T_mux_42_cyx_U1265", "Parent" : "706"},
	{"ID" : "711", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_avgpool_8x8_fu_18926.FracNet_T_mux_42_cyx_U1266", "Parent" : "706"},
	{"ID" : "712", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_avgpool_8x8_fu_18926.FracNet_T_mux_42_cyx_U1267", "Parent" : "706"},
	{"ID" : "713", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_avgpool_8x8_fu_18926.FracNet_T_mux_42_cyx_U1268", "Parent" : "706"},
	{"ID" : "714", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_avgpool_8x8_fu_18926.FracNet_T_mux_42_cyx_U1269", "Parent" : "706"},
	{"ID" : "715", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_avgpool_8x8_fu_18926.FracNet_T_mux_42_cyx_U1270", "Parent" : "706"},
	{"ID" : "716", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_avgpool_8x8_fu_18926.FracNet_T_mux_42_cyx_U1271", "Parent" : "706"},
	{"ID" : "717", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_avgpool_8x8_fu_18926.FracNet_T_mux_42_cyx_U1272", "Parent" : "706"},
	{"ID" : "718", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_avgpool_8x8_fu_18926.FracNet_T_mux_42_cyx_U1273", "Parent" : "706"},
	{"ID" : "719", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_avgpool_8x8_fu_18926.FracNet_T_mux_42_cyx_U1274", "Parent" : "706"},
	{"ID" : "720", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_avgpool_8x8_fu_18926.FracNet_T_mux_42_cyx_U1275", "Parent" : "706"},
	{"ID" : "721", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_avgpool_8x8_fu_18926.FracNet_T_mux_42_cyx_U1276", "Parent" : "706"},
	{"ID" : "722", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_avgpool_8x8_fu_18926.FracNet_T_mux_42_cyx_U1277", "Parent" : "706"},
	{"ID" : "723", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_avgpool_8x8_fu_18926.FracNet_T_mux_164cXB_U1278", "Parent" : "706"},
	{"ID" : "724", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_bn1_fu_18996", "Parent" : "0",
		"CDFG" : "bn1",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "1030", "EstimateLatencyMax" : "1030",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "out_buf_0_0_V", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "out_buf_0_1_V", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "out_buf_0_2_V", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "out_buf_0_3_V", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "out_buf_0_4_V", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "out_buf_0_5_V", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "out_buf_0_6_V", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "out_buf_0_7_V", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "out_buf_0_8_V", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "out_buf_0_9_V", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "out_buf_0_10_V", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "out_buf_0_11_V", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "out_buf_0_12_V", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "out_buf_0_13_V", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "out_buf_0_14_V", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "out_buf_0_15_V", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "block_t0_0_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "block_t0_1_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "block_t0_2_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "block_t0_3_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "block_t0_4_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "block_t0_5_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "block_t0_6_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "block_t0_7_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "block_t0_8_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "block_t0_9_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "block_t0_10_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "block_t0_11_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "block_t0_12_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "block_t0_13_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "block_t0_14_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "block_t0_15_V", "Type" : "Memory", "Direction" : "I"}]},
	{"ID" : "725", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_matmul_fu_19032", "Parent" : "0", "Child" : ["726", "727", "728", "729", "730", "731", "732", "733", "734", "735", "736", "737", "738", "739", "740", "741", "742", "743", "744", "745", "746", "747"],
		"CDFG" : "matmul",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "94", "EstimateLatencyMax" : "94",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "outputs_V", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "pool_out_buf", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "linear_weight_fix_V_s", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "linear_weight_fix_V_1", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "linear_weight_fix_V_2", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "linear_weight_fix_V_3", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "linear_weight_fix_V_4", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "linear_weight_fix_V_5", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "linear_weight_fix_V_6", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "linear_weight_fix_V_7", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "linear_weight_fix_V_8", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "linear_weight_fix_V_9", "Type" : "Memory", "Direction" : "I"}]},
	{"ID" : "726", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_matmul_fu_19032.linear_weight_fix_V_s_U", "Parent" : "725"},
	{"ID" : "727", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_matmul_fu_19032.linear_weight_fix_V_1_U", "Parent" : "725"},
	{"ID" : "728", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_matmul_fu_19032.linear_weight_fix_V_2_U", "Parent" : "725"},
	{"ID" : "729", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_matmul_fu_19032.linear_weight_fix_V_3_U", "Parent" : "725"},
	{"ID" : "730", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_matmul_fu_19032.linear_weight_fix_V_4_U", "Parent" : "725"},
	{"ID" : "731", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_matmul_fu_19032.linear_weight_fix_V_5_U", "Parent" : "725"},
	{"ID" : "732", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_matmul_fu_19032.linear_weight_fix_V_6_U", "Parent" : "725"},
	{"ID" : "733", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_matmul_fu_19032.linear_weight_fix_V_7_U", "Parent" : "725"},
	{"ID" : "734", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_matmul_fu_19032.linear_weight_fix_V_8_U", "Parent" : "725"},
	{"ID" : "735", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_matmul_fu_19032.linear_weight_fix_V_9_U", "Parent" : "725"},
	{"ID" : "736", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_matmul_fu_19032.FracNet_T_mux_164c8D_U1345", "Parent" : "725"},
	{"ID" : "737", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_matmul_fu_19032.FracNet_T_mux_104c9D_U1346", "Parent" : "725"},
	{"ID" : "738", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_matmul_fu_19032.FracNet_T_mul_muldaE_U1347", "Parent" : "725"},
	{"ID" : "739", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_matmul_fu_19032.FracNet_T_mul_muldaE_U1348", "Parent" : "725"},
	{"ID" : "740", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_matmul_fu_19032.FracNet_T_mul_muldaE_U1349", "Parent" : "725"},
	{"ID" : "741", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_matmul_fu_19032.FracNet_T_mul_muldbE_U1350", "Parent" : "725"},
	{"ID" : "742", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_matmul_fu_19032.FracNet_T_mul_muldaE_U1351", "Parent" : "725"},
	{"ID" : "743", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_matmul_fu_19032.FracNet_T_mul_muldaE_U1352", "Parent" : "725"},
	{"ID" : "744", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_matmul_fu_19032.FracNet_T_mul_muldaE_U1353", "Parent" : "725"},
	{"ID" : "745", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_matmul_fu_19032.FracNet_T_mul_muldaE_U1354", "Parent" : "725"},
	{"ID" : "746", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_matmul_fu_19032.FracNet_T_mul_muldaE_U1355", "Parent" : "725"},
	{"ID" : "747", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_matmul_fu_19032.FracNet_T_mul_muldaE_U1356", "Parent" : "725"},
	{"ID" : "748", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.FracNet_T_fcmp_32ePU_U1373", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
	FracNet_T {
		IMG {Type I LastRead 12 FirstWrite -1}
		RESULT {Type O LastRead 98 FirstWrite 104}
		image_V {Type I LastRead 0 FirstWrite -1}
		output_r {Type I LastRead 0 FirstWrite -1}
		conv_weight_all_V_0_8 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_0_7 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_0_6 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_0_5 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_0_4 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_0_3 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_0_2 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_0_1 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_0_s {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_1_8 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_1_7 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_1_6 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_1_5 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_1_4 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_1_3 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_1_2 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_1_1 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_1_s {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_2_8 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_2_7 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_2_6 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_2_5 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_2_4 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_2_3 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_2_2 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_2_1 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_2_s {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_3_8 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_3_7 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_3_6 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_3_5 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_3_4 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_3_3 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_3_2 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_3_1 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_3_s {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_4_8 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_4_7 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_4_6 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_4_5 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_4_4 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_4_3 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_4_2 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_4_1 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_4_s {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_5_8 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_5_7 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_5_6 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_5_5 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_5_4 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_5_3 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_5_2 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_5_1 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_5_s {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_6_8 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_6_7 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_6_6 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_6_5 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_6_4 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_6_3 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_6_2 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_6_1 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_6_s {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_7_8 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_7_7 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_7_6 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_7_5 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_7_4 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_7_3 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_7_2 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_7_1 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_7_s {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_8_8 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_8_7 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_8_6 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_8_5 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_8_4 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_8_3 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_8_2 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_8_1 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_8_s {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_9_8 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_9_7 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_9_6 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_9_5 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_9_4 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_9_3 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_9_2 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_9_1 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_9_s {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_10_8 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_10_7 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_10_6 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_10_5 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_10_4 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_10_3 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_10_2 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_10_1 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_10 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_11_8 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_11_7 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_11_6 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_11_5 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_11_4 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_11_3 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_11_2 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_11_1 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_11 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_12_8 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_12_7 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_12_6 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_12_5 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_12_4 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_12_3 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_12_2 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_12_1 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_12 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_13_8 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_13_7 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_13_6 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_13_5 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_13_4 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_13_3 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_13_2 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_13_1 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_13 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_14_8 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_14_7 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_14_6 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_14_5 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_14_4 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_14_3 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_14_2 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_14_1 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_14 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_15_8 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_15_7 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_15_6 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_15_5 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_15_4 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_15_3 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_15_2 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_15_1 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_15 {Type I LastRead -1 FirstWrite -1}
		gate_mask {Type I LastRead -1 FirstWrite -1}
		pool_out_buf {Type IO LastRead -1 FirstWrite -1}
		linear_weight_fix_V_s {Type I LastRead -1 FirstWrite -1}
		linear_weight_fix_V_1 {Type I LastRead -1 FirstWrite -1}
		linear_weight_fix_V_2 {Type I LastRead -1 FirstWrite -1}
		linear_weight_fix_V_3 {Type I LastRead -1 FirstWrite -1}
		linear_weight_fix_V_4 {Type I LastRead -1 FirstWrite -1}
		linear_weight_fix_V_5 {Type I LastRead -1 FirstWrite -1}
		linear_weight_fix_V_6 {Type I LastRead -1 FirstWrite -1}
		linear_weight_fix_V_7 {Type I LastRead -1 FirstWrite -1}
		linear_weight_fix_V_8 {Type I LastRead -1 FirstWrite -1}
		linear_weight_fix_V_9 {Type I LastRead -1 FirstWrite -1}}
	binary_conv3x3_tile {
		msb_inputs_V {Type I LastRead 3 FirstWrite -1}
		weights_V_offset {Type I LastRead 0 FirstWrite -1}
		msb_outputs_0_V {Type IO LastRead 3 FirstWrite 15}
		msb_outputs_1_V {Type IO LastRead 3 FirstWrite 15}
		msb_outputs_2_V {Type IO LastRead 3 FirstWrite 15}
		msb_outputs_3_V {Type IO LastRead 3 FirstWrite 15}
		msb_outputs_4_V {Type IO LastRead 3 FirstWrite 15}
		msb_outputs_5_V {Type IO LastRead 3 FirstWrite 15}
		msb_outputs_6_V {Type IO LastRead 3 FirstWrite 15}
		msb_outputs_7_V {Type IO LastRead 3 FirstWrite 15}
		msb_outputs_8_V {Type IO LastRead 3 FirstWrite 15}
		msb_outputs_9_V {Type IO LastRead 3 FirstWrite 15}
		msb_outputs_10_V {Type IO LastRead 3 FirstWrite 15}
		msb_outputs_11_V {Type IO LastRead 3 FirstWrite 15}
		msb_outputs_12_V {Type IO LastRead 3 FirstWrite 15}
		msb_outputs_13_V {Type IO LastRead 3 FirstWrite 15}
		msb_outputs_14_V {Type IO LastRead 3 FirstWrite 15}
		msb_outputs_15_V {Type IO LastRead 3 FirstWrite 15}
		comparator_0_V {Type I LastRead 3 FirstWrite -1}
		comparator_1_V {Type I LastRead 3 FirstWrite -1}
		comparator_2_V {Type I LastRead 3 FirstWrite -1}
		comparator_3_V {Type I LastRead 3 FirstWrite -1}
		comparator_4_V {Type I LastRead 3 FirstWrite -1}
		comparator_5_V {Type I LastRead 3 FirstWrite -1}
		comparator_6_V {Type I LastRead 3 FirstWrite -1}
		comparator_7_V {Type I LastRead 3 FirstWrite -1}
		comparator_8_V {Type I LastRead 3 FirstWrite -1}
		comparator_9_V {Type I LastRead 3 FirstWrite -1}
		comparator_10_V {Type I LastRead 3 FirstWrite -1}
		comparator_11_V {Type I LastRead 3 FirstWrite -1}
		comparator_12_V {Type I LastRead 3 FirstWrite -1}
		comparator_13_V {Type I LastRead 3 FirstWrite -1}
		comparator_14_V {Type I LastRead 3 FirstWrite -1}
		comparator_15_V {Type I LastRead 3 FirstWrite -1}
		p_read {Type I LastRead 1 FirstWrite -1}
		p_read1 {Type I LastRead 1 FirstWrite -1}
		p_read2 {Type I LastRead 1 FirstWrite -1}
		p_read3 {Type I LastRead 1 FirstWrite -1}
		p_read4 {Type I LastRead 1 FirstWrite -1}
		p_read5 {Type I LastRead 1 FirstWrite -1}
		p_read6 {Type I LastRead 1 FirstWrite -1}
		p_read7 {Type I LastRead 1 FirstWrite -1}
		p_read8 {Type I LastRead 1 FirstWrite -1}
		p_read9 {Type I LastRead 1 FirstWrite -1}
		p_read10 {Type I LastRead 1 FirstWrite -1}
		p_read11 {Type I LastRead 1 FirstWrite -1}
		p_read12 {Type I LastRead 1 FirstWrite -1}
		p_read13 {Type I LastRead 1 FirstWrite -1}
		p_read14 {Type I LastRead 1 FirstWrite -1}
		p_read15 {Type I LastRead 1 FirstWrite -1}
		p_read16 {Type I LastRead 1 FirstWrite -1}
		p_read17 {Type I LastRead 1 FirstWrite -1}
		p_read18 {Type I LastRead 1 FirstWrite -1}
		p_read19 {Type I LastRead 1 FirstWrite -1}
		p_read20 {Type I LastRead 1 FirstWrite -1}
		p_read21 {Type I LastRead 1 FirstWrite -1}
		p_read22 {Type I LastRead 1 FirstWrite -1}
		p_read23 {Type I LastRead 1 FirstWrite -1}
		p_read24 {Type I LastRead 1 FirstWrite -1}
		p_read25 {Type I LastRead 1 FirstWrite -1}
		p_read26 {Type I LastRead 1 FirstWrite -1}
		p_read27 {Type I LastRead 1 FirstWrite -1}
		p_read28 {Type I LastRead 1 FirstWrite -1}
		p_read29 {Type I LastRead 1 FirstWrite -1}
		p_read30 {Type I LastRead 1 FirstWrite -1}
		p_read31 {Type I LastRead 1 FirstWrite -1}
		p_read32 {Type I LastRead 1 FirstWrite -1}
		p_read33 {Type I LastRead 1 FirstWrite -1}
		p_read34 {Type I LastRead 1 FirstWrite -1}
		p_read35 {Type I LastRead 1 FirstWrite -1}
		p_read36 {Type I LastRead 1 FirstWrite -1}
		p_read37 {Type I LastRead 1 FirstWrite -1}
		p_read38 {Type I LastRead 1 FirstWrite -1}
		p_read39 {Type I LastRead 1 FirstWrite -1}
		p_read40 {Type I LastRead 1 FirstWrite -1}
		p_read41 {Type I LastRead 1 FirstWrite -1}
		p_read42 {Type I LastRead 1 FirstWrite -1}
		p_read43 {Type I LastRead 1 FirstWrite -1}
		p_read44 {Type I LastRead 1 FirstWrite -1}
		p_read45 {Type I LastRead 1 FirstWrite -1}
		p_read46 {Type I LastRead 1 FirstWrite -1}
		p_read47 {Type I LastRead 1 FirstWrite -1}
		p_read48 {Type I LastRead 1 FirstWrite -1}
		p_read49 {Type I LastRead 1 FirstWrite -1}
		p_read50 {Type I LastRead 1 FirstWrite -1}
		p_read51 {Type I LastRead 1 FirstWrite -1}
		p_read52 {Type I LastRead 1 FirstWrite -1}
		p_read53 {Type I LastRead 1 FirstWrite -1}
		p_read54 {Type I LastRead 1 FirstWrite -1}
		p_read55 {Type I LastRead 1 FirstWrite -1}
		p_read56 {Type I LastRead 1 FirstWrite -1}
		p_read57 {Type I LastRead 1 FirstWrite -1}
		p_read58 {Type I LastRead 1 FirstWrite -1}
		p_read59 {Type I LastRead 1 FirstWrite -1}
		p_read60 {Type I LastRead 1 FirstWrite -1}
		p_read61 {Type I LastRead 1 FirstWrite -1}
		p_read62 {Type I LastRead 1 FirstWrite -1}
		p_read63 {Type I LastRead 1 FirstWrite -1}
		threshold_V_offset {Type I LastRead 1 FirstWrite -1}
		switch_on {Type I LastRead 1 FirstWrite -1}
		c_in {Type I LastRead 1 FirstWrite -1}
		in_channels {Type I LastRead 1 FirstWrite -1}
		H_fmap_out {Type I LastRead 1 FirstWrite -1}
		conv_weight_all_V_0_8 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_0_7 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_0_6 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_0_5 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_0_4 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_0_3 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_0_2 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_0_1 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_0_s {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_1_8 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_1_7 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_1_6 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_1_5 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_1_4 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_1_3 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_1_2 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_1_1 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_1_s {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_2_8 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_2_7 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_2_6 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_2_5 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_2_4 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_2_3 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_2_2 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_2_1 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_2_s {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_3_8 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_3_7 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_3_6 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_3_5 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_3_4 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_3_3 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_3_2 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_3_1 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_3_s {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_4_8 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_4_7 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_4_6 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_4_5 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_4_4 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_4_3 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_4_2 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_4_1 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_4_s {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_5_8 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_5_7 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_5_6 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_5_5 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_5_4 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_5_3 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_5_2 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_5_1 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_5_s {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_6_8 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_6_7 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_6_6 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_6_5 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_6_4 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_6_3 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_6_2 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_6_1 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_6_s {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_7_8 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_7_7 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_7_6 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_7_5 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_7_4 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_7_3 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_7_2 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_7_1 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_7_s {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_8_8 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_8_7 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_8_6 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_8_5 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_8_4 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_8_3 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_8_2 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_8_1 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_8_s {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_9_8 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_9_7 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_9_6 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_9_5 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_9_4 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_9_3 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_9_2 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_9_1 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_9_s {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_10_8 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_10_7 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_10_6 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_10_5 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_10_4 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_10_3 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_10_2 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_10_1 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_10 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_11_8 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_11_7 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_11_6 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_11_5 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_11_4 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_11_3 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_11_2 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_11_1 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_11 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_12_8 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_12_7 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_12_6 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_12_5 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_12_4 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_12_3 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_12_2 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_12_1 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_12 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_13_8 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_13_7 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_13_6 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_13_5 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_13_4 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_13_3 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_13_2 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_13_1 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_13 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_14_8 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_14_7 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_14_6 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_14_5 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_14_4 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_14_3 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_14_2 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_14_1 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_14 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_15_8 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_15_7 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_15_6 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_15_5 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_15_4 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_15_3 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_15_2 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_15_1 {Type I LastRead -1 FirstWrite -1}
		conv_weight_all_V_15 {Type I LastRead -1 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	compute_engine_64 {
		b_V {Type I LastRead 0 FirstWrite -1}
		w_V {Type I LastRead 0 FirstWrite -1}}
	quant_and_pack {
		prior_outputs_0_0_V {Type I LastRead 2 FirstWrite -1}
		prior_outputs_0_1_V {Type I LastRead 2 FirstWrite -1}
		prior_outputs_0_2_V {Type I LastRead 2 FirstWrite -1}
		prior_outputs_0_3_V {Type I LastRead 2 FirstWrite -1}
		prior_outputs_0_4_V {Type I LastRead 2 FirstWrite -1}
		prior_outputs_0_5_V {Type I LastRead 2 FirstWrite -1}
		prior_outputs_0_6_V {Type I LastRead 2 FirstWrite -1}
		prior_outputs_0_7_V {Type I LastRead 2 FirstWrite -1}
		prior_outputs_0_8_V {Type I LastRead 2 FirstWrite -1}
		prior_outputs_0_9_V {Type I LastRead 2 FirstWrite -1}
		prior_outputs_0_10_V {Type I LastRead 2 FirstWrite -1}
		prior_outputs_0_11_V {Type I LastRead 2 FirstWrite -1}
		prior_outputs_0_12_V {Type I LastRead 2 FirstWrite -1}
		prior_outputs_0_13_V {Type I LastRead 2 FirstWrite -1}
		prior_outputs_0_14_V {Type I LastRead 2 FirstWrite -1}
		prior_outputs_0_15_V {Type I LastRead 2 FirstWrite -1}
		prior_outputs_1_0_V {Type I LastRead 2 FirstWrite -1}
		prior_outputs_1_1_V {Type I LastRead 2 FirstWrite -1}
		prior_outputs_1_2_V {Type I LastRead 2 FirstWrite -1}
		prior_outputs_1_3_V {Type I LastRead 2 FirstWrite -1}
		prior_outputs_1_4_V {Type I LastRead 2 FirstWrite -1}
		prior_outputs_1_5_V {Type I LastRead 2 FirstWrite -1}
		prior_outputs_1_6_V {Type I LastRead 2 FirstWrite -1}
		prior_outputs_1_7_V {Type I LastRead 2 FirstWrite -1}
		prior_outputs_1_8_V {Type I LastRead 2 FirstWrite -1}
		prior_outputs_1_9_V {Type I LastRead 2 FirstWrite -1}
		prior_outputs_1_10_V {Type I LastRead 2 FirstWrite -1}
		prior_outputs_1_11_V {Type I LastRead 2 FirstWrite -1}
		prior_outputs_1_12_V {Type I LastRead 2 FirstWrite -1}
		prior_outputs_1_13_V {Type I LastRead 2 FirstWrite -1}
		prior_outputs_1_14_V {Type I LastRead 2 FirstWrite -1}
		prior_outputs_1_15_V {Type I LastRead 2 FirstWrite -1}
		prior_outputs_2_0_V {Type I LastRead 2 FirstWrite -1}
		prior_outputs_2_1_V {Type I LastRead 2 FirstWrite -1}
		prior_outputs_2_2_V {Type I LastRead 2 FirstWrite -1}
		prior_outputs_2_3_V {Type I LastRead 2 FirstWrite -1}
		prior_outputs_2_4_V {Type I LastRead 2 FirstWrite -1}
		prior_outputs_2_5_V {Type I LastRead 2 FirstWrite -1}
		prior_outputs_2_6_V {Type I LastRead 2 FirstWrite -1}
		prior_outputs_2_7_V {Type I LastRead 2 FirstWrite -1}
		prior_outputs_2_8_V {Type I LastRead 2 FirstWrite -1}
		prior_outputs_2_9_V {Type I LastRead 2 FirstWrite -1}
		prior_outputs_2_10_V {Type I LastRead 2 FirstWrite -1}
		prior_outputs_2_11_V {Type I LastRead 2 FirstWrite -1}
		prior_outputs_2_12_V {Type I LastRead 2 FirstWrite -1}
		prior_outputs_2_13_V {Type I LastRead 2 FirstWrite -1}
		prior_outputs_2_14_V {Type I LastRead 2 FirstWrite -1}
		prior_outputs_2_15_V {Type I LastRead 2 FirstWrite -1}
		prior_outputs_3_0_V {Type I LastRead 2 FirstWrite -1}
		prior_outputs_3_1_V {Type I LastRead 2 FirstWrite -1}
		prior_outputs_3_2_V {Type I LastRead 2 FirstWrite -1}
		prior_outputs_3_3_V {Type I LastRead 2 FirstWrite -1}
		prior_outputs_3_4_V {Type I LastRead 2 FirstWrite -1}
		prior_outputs_3_5_V {Type I LastRead 2 FirstWrite -1}
		prior_outputs_3_6_V {Type I LastRead 2 FirstWrite -1}
		prior_outputs_3_7_V {Type I LastRead 2 FirstWrite -1}
		prior_outputs_3_8_V {Type I LastRead 2 FirstWrite -1}
		prior_outputs_3_9_V {Type I LastRead 2 FirstWrite -1}
		prior_outputs_3_10_V {Type I LastRead 2 FirstWrite -1}
		prior_outputs_3_11_V {Type I LastRead 2 FirstWrite -1}
		prior_outputs_3_12_V {Type I LastRead 2 FirstWrite -1}
		prior_outputs_3_13_V {Type I LastRead 2 FirstWrite -1}
		prior_outputs_3_14_V {Type I LastRead 2 FirstWrite -1}
		prior_outputs_3_15_V {Type I LastRead 2 FirstWrite -1}
		msb_buffer_0_V {Type IO LastRead 4 FirstWrite 5}
		msb_buffer_1_V {Type IO LastRead 4 FirstWrite 5}
		H_fmap {Type I LastRead 0 FirstWrite -1}
		in_channels {Type I LastRead 0 FirstWrite -1}}
	to2bit {
		x_V {Type I LastRead 0 FirstWrite -1}}
	to2bit {
		x_V {Type I LastRead 0 FirstWrite -1}}
	to2bit {
		x_V {Type I LastRead 0 FirstWrite -1}}
	to2bit {
		x_V {Type I LastRead 0 FirstWrite -1}}
	to2bit {
		x_V {Type I LastRead 0 FirstWrite -1}}
	to2bit {
		x_V {Type I LastRead 0 FirstWrite -1}}
	to2bit {
		x_V {Type I LastRead 0 FirstWrite -1}}
	to2bit {
		x_V {Type I LastRead 0 FirstWrite -1}}
	to2bit {
		x_V {Type I LastRead 0 FirstWrite -1}}
	to2bit {
		x_V {Type I LastRead 0 FirstWrite -1}}
	to2bit {
		x_V {Type I LastRead 0 FirstWrite -1}}
	to2bit {
		x_V {Type I LastRead 0 FirstWrite -1}}
	to2bit {
		x_V {Type I LastRead 0 FirstWrite -1}}
	to2bit {
		x_V {Type I LastRead 0 FirstWrite -1}}
	to2bit {
		x_V {Type I LastRead 0 FirstWrite -1}}
	to2bit {
		x_V {Type I LastRead 0 FirstWrite -1}}
	bn_relu_shortcut {
		residual_0_0_V {Type IO LastRead 2 FirstWrite 18}
		residual_0_1_V {Type IO LastRead 2 FirstWrite 18}
		residual_0_2_V {Type IO LastRead 2 FirstWrite 18}
		residual_0_3_V {Type IO LastRead 2 FirstWrite 18}
		residual_0_4_V {Type IO LastRead 2 FirstWrite 18}
		residual_0_5_V {Type IO LastRead 2 FirstWrite 18}
		residual_0_6_V {Type IO LastRead 2 FirstWrite 18}
		residual_0_7_V {Type IO LastRead 2 FirstWrite 18}
		residual_0_8_V {Type IO LastRead 2 FirstWrite 18}
		residual_0_9_V {Type IO LastRead 2 FirstWrite 18}
		residual_0_10_V {Type IO LastRead 2 FirstWrite 18}
		residual_0_11_V {Type IO LastRead 2 FirstWrite 18}
		residual_0_12_V {Type IO LastRead 2 FirstWrite 18}
		residual_0_13_V {Type IO LastRead 2 FirstWrite 18}
		residual_0_14_V {Type IO LastRead 2 FirstWrite 18}
		residual_0_15_V {Type IO LastRead 2 FirstWrite 18}
		residual_1_0_V {Type IO LastRead 2 FirstWrite 18}
		residual_1_1_V {Type IO LastRead 2 FirstWrite 18}
		residual_1_2_V {Type IO LastRead 2 FirstWrite 18}
		residual_1_3_V {Type IO LastRead 2 FirstWrite 18}
		residual_1_4_V {Type IO LastRead 2 FirstWrite 18}
		residual_1_5_V {Type IO LastRead 2 FirstWrite 18}
		residual_1_6_V {Type IO LastRead 2 FirstWrite 18}
		residual_1_7_V {Type IO LastRead 2 FirstWrite 18}
		residual_1_8_V {Type IO LastRead 2 FirstWrite 18}
		residual_1_9_V {Type IO LastRead 2 FirstWrite 18}
		residual_1_10_V {Type IO LastRead 2 FirstWrite 18}
		residual_1_11_V {Type IO LastRead 2 FirstWrite 18}
		residual_1_12_V {Type IO LastRead 2 FirstWrite 18}
		residual_1_13_V {Type IO LastRead 2 FirstWrite 18}
		residual_1_14_V {Type IO LastRead 2 FirstWrite 18}
		residual_1_15_V {Type IO LastRead 2 FirstWrite 18}
		residual_2_0_V {Type IO LastRead 2 FirstWrite 18}
		residual_2_1_V {Type IO LastRead 2 FirstWrite 18}
		residual_2_2_V {Type IO LastRead 2 FirstWrite 18}
		residual_2_3_V {Type IO LastRead 2 FirstWrite 18}
		residual_2_4_V {Type IO LastRead 2 FirstWrite 18}
		residual_2_5_V {Type IO LastRead 2 FirstWrite 18}
		residual_2_6_V {Type IO LastRead 2 FirstWrite 18}
		residual_2_7_V {Type IO LastRead 2 FirstWrite 18}
		residual_2_8_V {Type IO LastRead 2 FirstWrite 18}
		residual_2_9_V {Type IO LastRead 2 FirstWrite 18}
		residual_2_10_V {Type IO LastRead 2 FirstWrite 18}
		residual_2_11_V {Type IO LastRead 2 FirstWrite 18}
		residual_2_12_V {Type IO LastRead 2 FirstWrite 18}
		residual_2_13_V {Type IO LastRead 2 FirstWrite 18}
		residual_2_14_V {Type IO LastRead 2 FirstWrite 18}
		residual_2_15_V {Type IO LastRead 2 FirstWrite 18}
		residual_3_0_V {Type IO LastRead 2 FirstWrite 18}
		residual_3_1_V {Type IO LastRead 2 FirstWrite 18}
		residual_3_2_V {Type IO LastRead 2 FirstWrite 18}
		residual_3_3_V {Type IO LastRead 2 FirstWrite 18}
		residual_3_4_V {Type IO LastRead 2 FirstWrite 18}
		residual_3_5_V {Type IO LastRead 2 FirstWrite 18}
		residual_3_6_V {Type IO LastRead 2 FirstWrite 18}
		residual_3_7_V {Type IO LastRead 2 FirstWrite 18}
		residual_3_8_V {Type IO LastRead 2 FirstWrite 18}
		residual_3_9_V {Type IO LastRead 2 FirstWrite 18}
		residual_3_10_V {Type IO LastRead 2 FirstWrite 18}
		residual_3_11_V {Type IO LastRead 2 FirstWrite 18}
		residual_3_12_V {Type IO LastRead 2 FirstWrite 18}
		residual_3_13_V {Type IO LastRead 2 FirstWrite 18}
		residual_3_14_V {Type IO LastRead 2 FirstWrite 18}
		residual_3_15_V {Type IO LastRead 2 FirstWrite 18}
		block_t0_0_V {Type I LastRead 3 FirstWrite -1}
		block_t0_1_V {Type I LastRead 3 FirstWrite -1}
		block_t0_2_V {Type I LastRead 3 FirstWrite -1}
		block_t0_3_V {Type I LastRead 3 FirstWrite -1}
		block_t0_4_V {Type I LastRead 3 FirstWrite -1}
		block_t0_5_V {Type I LastRead 3 FirstWrite -1}
		block_t0_6_V {Type I LastRead 3 FirstWrite -1}
		block_t0_7_V {Type I LastRead 3 FirstWrite -1}
		block_t0_8_V {Type I LastRead 3 FirstWrite -1}
		block_t0_9_V {Type I LastRead 3 FirstWrite -1}
		block_t0_10_V {Type I LastRead 3 FirstWrite -1}
		block_t0_11_V {Type I LastRead 3 FirstWrite -1}
		block_t0_12_V {Type I LastRead 3 FirstWrite -1}
		block_t0_13_V {Type I LastRead 3 FirstWrite -1}
		block_t0_14_V {Type I LastRead 3 FirstWrite -1}
		block_t0_15_V {Type I LastRead 3 FirstWrite -1}
		block_t1_0_V {Type I LastRead 3 FirstWrite -1}
		block_t1_1_V {Type I LastRead 3 FirstWrite -1}
		block_t1_2_V {Type I LastRead 3 FirstWrite -1}
		block_t1_3_V {Type I LastRead 3 FirstWrite -1}
		block_t1_4_V {Type I LastRead 3 FirstWrite -1}
		block_t1_5_V {Type I LastRead 3 FirstWrite -1}
		block_t1_6_V {Type I LastRead 3 FirstWrite -1}
		block_t1_7_V {Type I LastRead 3 FirstWrite -1}
		block_t1_8_V {Type I LastRead 3 FirstWrite -1}
		block_t1_9_V {Type I LastRead 3 FirstWrite -1}
		block_t1_10_V {Type I LastRead 3 FirstWrite -1}
		block_t1_11_V {Type I LastRead 3 FirstWrite -1}
		block_t1_12_V {Type I LastRead 3 FirstWrite -1}
		block_t1_13_V {Type I LastRead 3 FirstWrite -1}
		block_t1_14_V {Type I LastRead 3 FirstWrite -1}
		block_t1_15_V {Type I LastRead 3 FirstWrite -1}
		bn_weight_0_0_0_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_0_1_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_0_2_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_0_3_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_1_0_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_1_1_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_1_2_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_1_3_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_2_0_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_2_1_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_2_2_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_2_3_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_3_0_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_3_1_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_3_2_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_3_3_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_4_0_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_4_1_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_4_2_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_4_3_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_5_0_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_5_1_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_5_2_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_5_3_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_6_0_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_6_1_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_6_2_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_6_3_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_7_0_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_7_1_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_7_2_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_7_3_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_8_0_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_8_1_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_8_2_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_8_3_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_9_0_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_9_1_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_9_2_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_9_3_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_10_0_V_read {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_10_1_V_read {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_10_2_V_read {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_10_3_V_read {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_11_0_V_read {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_11_1_V_read {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_11_2_V_read {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_11_3_V_read {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_12_0_V_read {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_12_1_V_read {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_12_2_V_read {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_12_3_V_read {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_13_0_V_read {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_13_1_V_read {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_13_2_V_read {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_13_3_V_read {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_14_0_V_read {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_14_1_V_read {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_14_2_V_read {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_14_3_V_read {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_15_0_V_read {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_15_1_V_read {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_15_2_V_read {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_15_3_V_read {Type I LastRead 0 FirstWrite -1}
		bn_weight_0_V_offset {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_0_0_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_0_1_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_0_2_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_0_3_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_1_0_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_1_1_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_1_2_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_1_3_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_2_0_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_2_1_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_2_2_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_2_3_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_3_0_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_3_1_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_3_2_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_3_3_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_4_0_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_4_1_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_4_2_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_4_3_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_5_0_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_5_1_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_5_2_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_5_3_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_6_0_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_6_1_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_6_2_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_6_3_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_7_0_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_7_1_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_7_2_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_7_3_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_8_0_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_8_1_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_8_2_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_8_3_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_9_0_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_9_1_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_9_2_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_9_3_V_s {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_10_0_V_read {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_10_1_V_read {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_10_2_V_read {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_10_3_V_read {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_11_0_V_read {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_11_1_V_read {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_11_2_V_read {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_11_3_V_read {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_12_0_V_read {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_12_1_V_read {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_12_2_V_read {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_12_3_V_read {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_13_0_V_read {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_13_1_V_read {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_13_2_V_read {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_13_3_V_read {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_14_0_V_read {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_14_1_V_read {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_14_2_V_read {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_14_3_V_read {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_15_0_V_read {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_15_1_V_read {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_15_2_V_read {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_15_3_V_read {Type I LastRead 0 FirstWrite -1}
		bn_weight_1_V_offset {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_0_0_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_0_1_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_0_2_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_0_3_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_1_0_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_1_1_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_1_2_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_1_3_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_2_0_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_2_1_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_2_2_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_2_3_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_3_0_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_3_1_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_3_2_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_3_3_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_4_0_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_4_1_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_4_2_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_4_3_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_5_0_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_5_1_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_5_2_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_5_3_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_6_0_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_6_1_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_6_2_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_6_3_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_7_0_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_7_1_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_7_2_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_7_3_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_8_0_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_8_1_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_8_2_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_8_3_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_9_0_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_9_1_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_9_2_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_9_3_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_10_0_V_r {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_10_1_V_r {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_10_2_V_r {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_10_3_V_r {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_11_0_V_r {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_11_1_V_r {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_11_2_V_r {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_11_3_V_r {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_12_0_V_r {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_12_1_V_r {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_12_2_V_r {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_12_3_V_r {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_13_0_V_r {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_13_1_V_r {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_13_2_V_r {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_13_3_V_r {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_14_0_V_r {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_14_1_V_r {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_14_2_V_r {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_14_3_V_r {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_15_0_V_r {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_15_1_V_r {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_15_2_V_r {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_15_3_V_r {Type I LastRead 0 FirstWrite -1}
		bn_bias_0_V_offset {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_0_0_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_0_1_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_0_2_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_0_3_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_1_0_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_1_1_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_1_2_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_1_3_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_2_0_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_2_1_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_2_2_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_2_3_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_3_0_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_3_1_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_3_2_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_3_3_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_4_0_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_4_1_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_4_2_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_4_3_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_5_0_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_5_1_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_5_2_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_5_3_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_6_0_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_6_1_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_6_2_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_6_3_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_7_0_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_7_1_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_7_2_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_7_3_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_8_0_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_8_1_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_8_2_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_8_3_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_9_0_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_9_1_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_9_2_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_9_3_V_re {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_10_0_V_r {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_10_1_V_r {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_10_2_V_r {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_10_3_V_r {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_11_0_V_r {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_11_1_V_r {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_11_2_V_r {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_11_3_V_r {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_12_0_V_r {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_12_1_V_r {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_12_2_V_r {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_12_3_V_r {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_13_0_V_r {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_13_1_V_r {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_13_2_V_r {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_13_3_V_r {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_14_0_V_r {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_14_1_V_r {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_14_2_V_r {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_14_3_V_r {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_15_0_V_r {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_15_1_V_r {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_15_2_V_r {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_15_3_V_r {Type I LastRead 0 FirstWrite -1}
		bn_bias_1_V_offset {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_0_0_V_s {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_0_1_V_s {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_0_2_V_s {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_0_3_V_s {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_1_0_V_s {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_1_1_V_s {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_1_2_V_s {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_1_3_V_s {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_2_0_V_s {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_2_1_V_s {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_2_2_V_s {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_2_3_V_s {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_3_0_V_s {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_3_1_V_s {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_3_2_V_s {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_3_3_V_s {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_4_0_V_s {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_4_1_V_s {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_4_2_V_s {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_4_3_V_s {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_5_0_V_s {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_5_1_V_s {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_5_2_V_s {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_5_3_V_s {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_6_0_V_s {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_6_1_V_s {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_6_2_V_s {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_6_3_V_s {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_7_0_V_s {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_7_1_V_s {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_7_2_V_s {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_7_3_V_s {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_8_0_V_s {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_8_1_V_s {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_8_2_V_s {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_8_3_V_s {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_9_0_V_s {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_9_1_V_s {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_9_2_V_s {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_9_3_V_s {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_10_0_V_read {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_10_1_V_read {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_10_2_V_read {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_10_3_V_read {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_11_0_V_read {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_11_1_V_read {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_11_2_V_read {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_11_3_V_read {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_12_0_V_read {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_12_1_V_read {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_12_2_V_read {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_12_3_V_read {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_13_0_V_read {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_13_1_V_read {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_13_2_V_read {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_13_3_V_read {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_14_0_V_read {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_14_1_V_read {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_14_2_V_read {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_14_3_V_read {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_15_0_V_read {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_15_1_V_read {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_15_2_V_read {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_15_3_V_read {Type I LastRead 0 FirstWrite -1}
		relu_x_bias_V_offset {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_0_0_V_s {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_0_1_V_s {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_0_2_V_s {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_0_3_V_s {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_1_0_V_s {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_1_1_V_s {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_1_2_V_s {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_1_3_V_s {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_2_0_V_s {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_2_1_V_s {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_2_2_V_s {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_2_3_V_s {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_3_0_V_s {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_3_1_V_s {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_3_2_V_s {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_3_3_V_s {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_4_0_V_s {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_4_1_V_s {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_4_2_V_s {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_4_3_V_s {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_5_0_V_s {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_5_1_V_s {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_5_2_V_s {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_5_3_V_s {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_6_0_V_s {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_6_1_V_s {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_6_2_V_s {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_6_3_V_s {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_7_0_V_s {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_7_1_V_s {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_7_2_V_s {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_7_3_V_s {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_8_0_V_s {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_8_1_V_s {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_8_2_V_s {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_8_3_V_s {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_9_0_V_s {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_9_1_V_s {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_9_2_V_s {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_9_3_V_s {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_10_0_V_read {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_10_1_V_read {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_10_2_V_read {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_10_3_V_read {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_11_0_V_read {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_11_1_V_read {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_11_2_V_read {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_11_3_V_read {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_12_0_V_read {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_12_1_V_read {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_12_2_V_read {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_12_3_V_read {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_13_0_V_read {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_13_1_V_read {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_13_2_V_read {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_13_3_V_read {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_14_0_V_read {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_14_1_V_read {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_14_2_V_read {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_14_3_V_read {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_15_0_V_read {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_15_1_V_read {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_15_2_V_read {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_15_3_V_read {Type I LastRead 0 FirstWrite -1}
		relu_y_bias_V_offset {Type I LastRead 0 FirstWrite -1}
		relu_weight_0_0_V_s {Type I LastRead 0 FirstWrite -1}
		relu_weight_0_1_V_s {Type I LastRead 0 FirstWrite -1}
		relu_weight_0_2_V_s {Type I LastRead 0 FirstWrite -1}
		relu_weight_0_3_V_s {Type I LastRead 0 FirstWrite -1}
		relu_weight_1_0_V_s {Type I LastRead 0 FirstWrite -1}
		relu_weight_1_1_V_s {Type I LastRead 0 FirstWrite -1}
		relu_weight_1_2_V_s {Type I LastRead 0 FirstWrite -1}
		relu_weight_1_3_V_s {Type I LastRead 0 FirstWrite -1}
		relu_weight_2_0_V_s {Type I LastRead 0 FirstWrite -1}
		relu_weight_2_1_V_s {Type I LastRead 0 FirstWrite -1}
		relu_weight_2_2_V_s {Type I LastRead 0 FirstWrite -1}
		relu_weight_2_3_V_s {Type I LastRead 0 FirstWrite -1}
		relu_weight_3_0_V_s {Type I LastRead 0 FirstWrite -1}
		relu_weight_3_1_V_s {Type I LastRead 0 FirstWrite -1}
		relu_weight_3_2_V_s {Type I LastRead 0 FirstWrite -1}
		relu_weight_3_3_V_s {Type I LastRead 0 FirstWrite -1}
		relu_weight_4_0_V_s {Type I LastRead 0 FirstWrite -1}
		relu_weight_4_1_V_s {Type I LastRead 0 FirstWrite -1}
		relu_weight_4_2_V_s {Type I LastRead 0 FirstWrite -1}
		relu_weight_4_3_V_s {Type I LastRead 0 FirstWrite -1}
		relu_weight_5_0_V_s {Type I LastRead 0 FirstWrite -1}
		relu_weight_5_1_V_s {Type I LastRead 0 FirstWrite -1}
		relu_weight_5_2_V_s {Type I LastRead 0 FirstWrite -1}
		relu_weight_5_3_V_s {Type I LastRead 0 FirstWrite -1}
		relu_weight_6_0_V_s {Type I LastRead 0 FirstWrite -1}
		relu_weight_6_1_V_s {Type I LastRead 0 FirstWrite -1}
		relu_weight_6_2_V_s {Type I LastRead 0 FirstWrite -1}
		relu_weight_6_3_V_s {Type I LastRead 0 FirstWrite -1}
		relu_weight_7_0_V_s {Type I LastRead 0 FirstWrite -1}
		relu_weight_7_1_V_s {Type I LastRead 0 FirstWrite -1}
		relu_weight_7_2_V_s {Type I LastRead 0 FirstWrite -1}
		relu_weight_7_3_V_s {Type I LastRead 0 FirstWrite -1}
		relu_weight_8_0_V_s {Type I LastRead 0 FirstWrite -1}
		relu_weight_8_1_V_s {Type I LastRead 0 FirstWrite -1}
		relu_weight_8_2_V_s {Type I LastRead 0 FirstWrite -1}
		relu_weight_8_3_V_s {Type I LastRead 0 FirstWrite -1}
		relu_weight_9_0_V_s {Type I LastRead 0 FirstWrite -1}
		relu_weight_9_1_V_s {Type I LastRead 0 FirstWrite -1}
		relu_weight_9_2_V_s {Type I LastRead 0 FirstWrite -1}
		relu_weight_9_3_V_s {Type I LastRead 0 FirstWrite -1}
		relu_weight_10_0_V_read {Type I LastRead 0 FirstWrite -1}
		relu_weight_10_1_V_read {Type I LastRead 0 FirstWrite -1}
		relu_weight_10_2_V_read {Type I LastRead 0 FirstWrite -1}
		relu_weight_10_3_V_read {Type I LastRead 0 FirstWrite -1}
		relu_weight_11_0_V_read {Type I LastRead 0 FirstWrite -1}
		relu_weight_11_1_V_read {Type I LastRead 0 FirstWrite -1}
		relu_weight_11_2_V_read {Type I LastRead 0 FirstWrite -1}
		relu_weight_11_3_V_read {Type I LastRead 0 FirstWrite -1}
		relu_weight_12_0_V_read {Type I LastRead 0 FirstWrite -1}
		relu_weight_12_1_V_read {Type I LastRead 0 FirstWrite -1}
		relu_weight_12_2_V_read {Type I LastRead 0 FirstWrite -1}
		relu_weight_12_3_V_read {Type I LastRead 0 FirstWrite -1}
		relu_weight_13_0_V_read {Type I LastRead 0 FirstWrite -1}
		relu_weight_13_1_V_read {Type I LastRead 0 FirstWrite -1}
		relu_weight_13_2_V_read {Type I LastRead 0 FirstWrite -1}
		relu_weight_13_3_V_read {Type I LastRead 0 FirstWrite -1}
		relu_weight_14_0_V_read {Type I LastRead 0 FirstWrite -1}
		relu_weight_14_1_V_read {Type I LastRead 0 FirstWrite -1}
		relu_weight_14_2_V_read {Type I LastRead 0 FirstWrite -1}
		relu_weight_14_3_V_read {Type I LastRead 0 FirstWrite -1}
		relu_weight_15_0_V_read {Type I LastRead 0 FirstWrite -1}
		relu_weight_15_1_V_read {Type I LastRead 0 FirstWrite -1}
		relu_weight_15_2_V_read {Type I LastRead 0 FirstWrite -1}
		relu_weight_15_3_V_read {Type I LastRead 0 FirstWrite -1}
		relu_weight_V_offset {Type I LastRead 0 FirstWrite -1}
		stride {Type I LastRead 0 FirstWrite -1}
		channel_tile {Type I LastRead 0 FirstWrite -1}
		H_fmap {Type I LastRead 0 FirstWrite -1}}
	avgpool_concat {
		outputs_0_0_V {Type IO LastRead 6 FirstWrite 8}
		outputs_0_1_V {Type IO LastRead 6 FirstWrite 8}
		outputs_0_2_V {Type IO LastRead 6 FirstWrite 8}
		outputs_0_3_V {Type IO LastRead 6 FirstWrite 8}
		outputs_0_4_V {Type IO LastRead 6 FirstWrite 8}
		outputs_0_5_V {Type IO LastRead 6 FirstWrite 8}
		outputs_0_6_V {Type IO LastRead 6 FirstWrite 8}
		outputs_0_7_V {Type IO LastRead 6 FirstWrite 8}
		outputs_0_8_V {Type IO LastRead 6 FirstWrite 8}
		outputs_0_9_V {Type IO LastRead 6 FirstWrite 8}
		outputs_0_10_V {Type IO LastRead 6 FirstWrite 8}
		outputs_0_11_V {Type IO LastRead 6 FirstWrite 8}
		outputs_0_12_V {Type IO LastRead 6 FirstWrite 8}
		outputs_0_13_V {Type IO LastRead 6 FirstWrite 8}
		outputs_0_14_V {Type IO LastRead 6 FirstWrite 8}
		outputs_0_15_V {Type IO LastRead 6 FirstWrite 8}
		outputs_1_0_V {Type IO LastRead 6 FirstWrite 8}
		outputs_1_1_V {Type IO LastRead 6 FirstWrite 8}
		outputs_1_2_V {Type IO LastRead 6 FirstWrite 8}
		outputs_1_3_V {Type IO LastRead 6 FirstWrite 8}
		outputs_1_4_V {Type IO LastRead 6 FirstWrite 8}
		outputs_1_5_V {Type IO LastRead 6 FirstWrite 8}
		outputs_1_6_V {Type IO LastRead 6 FirstWrite 8}
		outputs_1_7_V {Type IO LastRead 6 FirstWrite 8}
		outputs_1_8_V {Type IO LastRead 6 FirstWrite 8}
		outputs_1_9_V {Type IO LastRead 6 FirstWrite 8}
		outputs_1_10_V {Type IO LastRead 6 FirstWrite 8}
		outputs_1_11_V {Type IO LastRead 6 FirstWrite 8}
		outputs_1_12_V {Type IO LastRead 6 FirstWrite 8}
		outputs_1_13_V {Type IO LastRead 6 FirstWrite 8}
		outputs_1_14_V {Type IO LastRead 6 FirstWrite 8}
		outputs_1_15_V {Type IO LastRead 6 FirstWrite 8}
		outputs_2_0_V {Type IO LastRead 6 FirstWrite 8}
		outputs_2_1_V {Type IO LastRead 6 FirstWrite 8}
		outputs_2_2_V {Type IO LastRead 6 FirstWrite 8}
		outputs_2_3_V {Type IO LastRead 6 FirstWrite 8}
		outputs_2_4_V {Type IO LastRead 6 FirstWrite 8}
		outputs_2_5_V {Type IO LastRead 6 FirstWrite 8}
		outputs_2_6_V {Type IO LastRead 6 FirstWrite 8}
		outputs_2_7_V {Type IO LastRead 6 FirstWrite 8}
		outputs_2_8_V {Type IO LastRead 6 FirstWrite 8}
		outputs_2_9_V {Type IO LastRead 6 FirstWrite 8}
		outputs_2_10_V {Type IO LastRead 6 FirstWrite 8}
		outputs_2_11_V {Type IO LastRead 6 FirstWrite 8}
		outputs_2_12_V {Type IO LastRead 6 FirstWrite 8}
		outputs_2_13_V {Type IO LastRead 6 FirstWrite 8}
		outputs_2_14_V {Type IO LastRead 6 FirstWrite 8}
		outputs_2_15_V {Type IO LastRead 6 FirstWrite 8}
		outputs_3_0_V {Type IO LastRead 6 FirstWrite 8}
		outputs_3_1_V {Type IO LastRead 6 FirstWrite 8}
		outputs_3_2_V {Type IO LastRead 6 FirstWrite 8}
		outputs_3_3_V {Type IO LastRead 6 FirstWrite 8}
		outputs_3_4_V {Type IO LastRead 6 FirstWrite 8}
		outputs_3_5_V {Type IO LastRead 6 FirstWrite 8}
		outputs_3_6_V {Type IO LastRead 6 FirstWrite 8}
		outputs_3_7_V {Type IO LastRead 6 FirstWrite 8}
		outputs_3_8_V {Type IO LastRead 6 FirstWrite 8}
		outputs_3_9_V {Type IO LastRead 6 FirstWrite 8}
		outputs_3_10_V {Type IO LastRead 6 FirstWrite 8}
		outputs_3_11_V {Type IO LastRead 6 FirstWrite 8}
		outputs_3_12_V {Type IO LastRead 6 FirstWrite 8}
		outputs_3_13_V {Type IO LastRead 6 FirstWrite 8}
		outputs_3_14_V {Type IO LastRead 6 FirstWrite 8}
		outputs_3_15_V {Type IO LastRead 6 FirstWrite 8}
		H_fmap {Type I LastRead 0 FirstWrite -1}
		in_channels {Type I LastRead 0 FirstWrite -1}}
	avgpool_8x8 {
		inputs_0_0_V {Type I LastRead 5 FirstWrite -1}
		inputs_0_1_V {Type I LastRead 5 FirstWrite -1}
		inputs_0_2_V {Type I LastRead 5 FirstWrite -1}
		inputs_0_3_V {Type I LastRead 5 FirstWrite -1}
		inputs_0_4_V {Type I LastRead 5 FirstWrite -1}
		inputs_0_5_V {Type I LastRead 5 FirstWrite -1}
		inputs_0_6_V {Type I LastRead 5 FirstWrite -1}
		inputs_0_7_V {Type I LastRead 5 FirstWrite -1}
		inputs_0_8_V {Type I LastRead 5 FirstWrite -1}
		inputs_0_9_V {Type I LastRead 5 FirstWrite -1}
		inputs_0_10_V {Type I LastRead 5 FirstWrite -1}
		inputs_0_11_V {Type I LastRead 5 FirstWrite -1}
		inputs_0_12_V {Type I LastRead 5 FirstWrite -1}
		inputs_0_13_V {Type I LastRead 5 FirstWrite -1}
		inputs_0_14_V {Type I LastRead 5 FirstWrite -1}
		inputs_0_15_V {Type I LastRead 5 FirstWrite -1}
		inputs_1_0_V {Type I LastRead 5 FirstWrite -1}
		inputs_1_1_V {Type I LastRead 5 FirstWrite -1}
		inputs_1_2_V {Type I LastRead 5 FirstWrite -1}
		inputs_1_3_V {Type I LastRead 5 FirstWrite -1}
		inputs_1_4_V {Type I LastRead 5 FirstWrite -1}
		inputs_1_5_V {Type I LastRead 5 FirstWrite -1}
		inputs_1_6_V {Type I LastRead 5 FirstWrite -1}
		inputs_1_7_V {Type I LastRead 5 FirstWrite -1}
		inputs_1_8_V {Type I LastRead 5 FirstWrite -1}
		inputs_1_9_V {Type I LastRead 5 FirstWrite -1}
		inputs_1_10_V {Type I LastRead 5 FirstWrite -1}
		inputs_1_11_V {Type I LastRead 5 FirstWrite -1}
		inputs_1_12_V {Type I LastRead 5 FirstWrite -1}
		inputs_1_13_V {Type I LastRead 5 FirstWrite -1}
		inputs_1_14_V {Type I LastRead 5 FirstWrite -1}
		inputs_1_15_V {Type I LastRead 5 FirstWrite -1}
		inputs_2_0_V {Type I LastRead 5 FirstWrite -1}
		inputs_2_1_V {Type I LastRead 5 FirstWrite -1}
		inputs_2_2_V {Type I LastRead 5 FirstWrite -1}
		inputs_2_3_V {Type I LastRead 5 FirstWrite -1}
		inputs_2_4_V {Type I LastRead 5 FirstWrite -1}
		inputs_2_5_V {Type I LastRead 5 FirstWrite -1}
		inputs_2_6_V {Type I LastRead 5 FirstWrite -1}
		inputs_2_7_V {Type I LastRead 5 FirstWrite -1}
		inputs_2_8_V {Type I LastRead 5 FirstWrite -1}
		inputs_2_9_V {Type I LastRead 5 FirstWrite -1}
		inputs_2_10_V {Type I LastRead 5 FirstWrite -1}
		inputs_2_11_V {Type I LastRead 5 FirstWrite -1}
		inputs_2_12_V {Type I LastRead 5 FirstWrite -1}
		inputs_2_13_V {Type I LastRead 5 FirstWrite -1}
		inputs_2_14_V {Type I LastRead 5 FirstWrite -1}
		inputs_2_15_V {Type I LastRead 5 FirstWrite -1}
		inputs_3_0_V {Type I LastRead 5 FirstWrite -1}
		inputs_3_1_V {Type I LastRead 5 FirstWrite -1}
		inputs_3_2_V {Type I LastRead 5 FirstWrite -1}
		inputs_3_3_V {Type I LastRead 5 FirstWrite -1}
		inputs_3_4_V {Type I LastRead 5 FirstWrite -1}
		inputs_3_5_V {Type I LastRead 5 FirstWrite -1}
		inputs_3_6_V {Type I LastRead 5 FirstWrite -1}
		inputs_3_7_V {Type I LastRead 5 FirstWrite -1}
		inputs_3_8_V {Type I LastRead 5 FirstWrite -1}
		inputs_3_9_V {Type I LastRead 5 FirstWrite -1}
		inputs_3_10_V {Type I LastRead 5 FirstWrite -1}
		inputs_3_11_V {Type I LastRead 5 FirstWrite -1}
		inputs_3_12_V {Type I LastRead 5 FirstWrite -1}
		inputs_3_13_V {Type I LastRead 5 FirstWrite -1}
		inputs_3_14_V {Type I LastRead 5 FirstWrite -1}
		inputs_3_15_V {Type I LastRead 5 FirstWrite -1}
		outputs_V {Type O LastRead -1 FirstWrite 10}}
	bn1 {
		out_buf_0_0_V {Type O LastRead -1 FirstWrite 5}
		out_buf_0_1_V {Type O LastRead -1 FirstWrite 4}
		out_buf_0_2_V {Type O LastRead -1 FirstWrite 4}
		out_buf_0_3_V {Type O LastRead -1 FirstWrite 5}
		out_buf_0_4_V {Type O LastRead -1 FirstWrite 5}
		out_buf_0_5_V {Type O LastRead -1 FirstWrite 5}
		out_buf_0_6_V {Type O LastRead -1 FirstWrite 4}
		out_buf_0_7_V {Type O LastRead -1 FirstWrite 5}
		out_buf_0_8_V {Type O LastRead -1 FirstWrite 6}
		out_buf_0_9_V {Type O LastRead -1 FirstWrite 4}
		out_buf_0_10_V {Type O LastRead -1 FirstWrite 5}
		out_buf_0_11_V {Type O LastRead -1 FirstWrite 5}
		out_buf_0_12_V {Type O LastRead -1 FirstWrite 4}
		out_buf_0_13_V {Type O LastRead -1 FirstWrite 5}
		out_buf_0_14_V {Type O LastRead -1 FirstWrite 5}
		out_buf_0_15_V {Type O LastRead -1 FirstWrite 5}
		block_t0_0_V {Type I LastRead 2 FirstWrite -1}
		block_t0_1_V {Type I LastRead 2 FirstWrite -1}
		block_t0_2_V {Type I LastRead 2 FirstWrite -1}
		block_t0_3_V {Type I LastRead 2 FirstWrite -1}
		block_t0_4_V {Type I LastRead 2 FirstWrite -1}
		block_t0_5_V {Type I LastRead 2 FirstWrite -1}
		block_t0_6_V {Type I LastRead 2 FirstWrite -1}
		block_t0_7_V {Type I LastRead 2 FirstWrite -1}
		block_t0_8_V {Type I LastRead 2 FirstWrite -1}
		block_t0_9_V {Type I LastRead 2 FirstWrite -1}
		block_t0_10_V {Type I LastRead 2 FirstWrite -1}
		block_t0_11_V {Type I LastRead 2 FirstWrite -1}
		block_t0_12_V {Type I LastRead 2 FirstWrite -1}
		block_t0_13_V {Type I LastRead 2 FirstWrite -1}
		block_t0_14_V {Type I LastRead 2 FirstWrite -1}
		block_t0_15_V {Type I LastRead 2 FirstWrite -1}}
	matmul {
		outputs_V {Type O LastRead -1 FirstWrite 5}
		pool_out_buf {Type I LastRead 3 FirstWrite -1}
		linear_weight_fix_V_s {Type I LastRead -1 FirstWrite -1}
		linear_weight_fix_V_1 {Type I LastRead -1 FirstWrite -1}
		linear_weight_fix_V_2 {Type I LastRead -1 FirstWrite -1}
		linear_weight_fix_V_3 {Type I LastRead -1 FirstWrite -1}
		linear_weight_fix_V_4 {Type I LastRead -1 FirstWrite -1}
		linear_weight_fix_V_5 {Type I LastRead -1 FirstWrite -1}
		linear_weight_fix_V_6 {Type I LastRead -1 FirstWrite -1}
		linear_weight_fix_V_7 {Type I LastRead -1 FirstWrite -1}
		linear_weight_fix_V_8 {Type I LastRead -1 FirstWrite -1}
		linear_weight_fix_V_9 {Type I LastRead -1 FirstWrite -1}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "28120", "Max" : "232782"}
	, {"Name" : "Interval", "Min" : "28121", "Max" : "232783"}
]}

set PipelineEnableSignalInfo {[
	{"Pipeline" : "0", "EnableSignal" : "ap_enable_pp0"}
]}

set Spec2ImplPortList { 
	IMG { m_axi {  { m_axi_IMG_AWVALID VALID 1 1 }  { m_axi_IMG_AWREADY READY 0 1 }  { m_axi_IMG_AWADDR ADDR 1 32 }  { m_axi_IMG_AWID ID 1 1 }  { m_axi_IMG_AWLEN LEN 1 8 }  { m_axi_IMG_AWSIZE SIZE 1 3 }  { m_axi_IMG_AWBURST BURST 1 2 }  { m_axi_IMG_AWLOCK LOCK 1 2 }  { m_axi_IMG_AWCACHE CACHE 1 4 }  { m_axi_IMG_AWPROT PROT 1 3 }  { m_axi_IMG_AWQOS QOS 1 4 }  { m_axi_IMG_AWREGION REGION 1 4 }  { m_axi_IMG_AWUSER USER 1 1 }  { m_axi_IMG_WVALID VALID 1 1 }  { m_axi_IMG_WREADY READY 0 1 }  { m_axi_IMG_WDATA DATA 1 64 }  { m_axi_IMG_WSTRB STRB 1 8 }  { m_axi_IMG_WLAST LAST 1 1 }  { m_axi_IMG_WID ID 1 1 }  { m_axi_IMG_WUSER USER 1 1 }  { m_axi_IMG_ARVALID VALID 1 1 }  { m_axi_IMG_ARREADY READY 0 1 }  { m_axi_IMG_ARADDR ADDR 1 32 }  { m_axi_IMG_ARID ID 1 1 }  { m_axi_IMG_ARLEN LEN 1 8 }  { m_axi_IMG_ARSIZE SIZE 1 3 }  { m_axi_IMG_ARBURST BURST 1 2 }  { m_axi_IMG_ARLOCK LOCK 1 2 }  { m_axi_IMG_ARCACHE CACHE 1 4 }  { m_axi_IMG_ARPROT PROT 1 3 }  { m_axi_IMG_ARQOS QOS 1 4 }  { m_axi_IMG_ARREGION REGION 1 4 }  { m_axi_IMG_ARUSER USER 1 1 }  { m_axi_IMG_RVALID VALID 0 1 }  { m_axi_IMG_RREADY READY 1 1 }  { m_axi_IMG_RDATA DATA 0 64 }  { m_axi_IMG_RLAST LAST 0 1 }  { m_axi_IMG_RID ID 0 1 }  { m_axi_IMG_RUSER USER 0 1 }  { m_axi_IMG_RRESP RESP 0 2 }  { m_axi_IMG_BVALID VALID 0 1 }  { m_axi_IMG_BREADY READY 1 1 }  { m_axi_IMG_BRESP RESP 0 2 }  { m_axi_IMG_BID ID 0 1 }  { m_axi_IMG_BUSER USER 0 1 } } }
	RESULT { m_axi {  { m_axi_RESULT_AWVALID VALID 1 1 }  { m_axi_RESULT_AWREADY READY 0 1 }  { m_axi_RESULT_AWADDR ADDR 1 32 }  { m_axi_RESULT_AWID ID 1 1 }  { m_axi_RESULT_AWLEN LEN 1 8 }  { m_axi_RESULT_AWSIZE SIZE 1 3 }  { m_axi_RESULT_AWBURST BURST 1 2 }  { m_axi_RESULT_AWLOCK LOCK 1 2 }  { m_axi_RESULT_AWCACHE CACHE 1 4 }  { m_axi_RESULT_AWPROT PROT 1 3 }  { m_axi_RESULT_AWQOS QOS 1 4 }  { m_axi_RESULT_AWREGION REGION 1 4 }  { m_axi_RESULT_AWUSER USER 1 1 }  { m_axi_RESULT_WVALID VALID 1 1 }  { m_axi_RESULT_WREADY READY 0 1 }  { m_axi_RESULT_WDATA DATA 1 32 }  { m_axi_RESULT_WSTRB STRB 1 4 }  { m_axi_RESULT_WLAST LAST 1 1 }  { m_axi_RESULT_WID ID 1 1 }  { m_axi_RESULT_WUSER USER 1 1 }  { m_axi_RESULT_ARVALID VALID 1 1 }  { m_axi_RESULT_ARREADY READY 0 1 }  { m_axi_RESULT_ARADDR ADDR 1 32 }  { m_axi_RESULT_ARID ID 1 1 }  { m_axi_RESULT_ARLEN LEN 1 8 }  { m_axi_RESULT_ARSIZE SIZE 1 3 }  { m_axi_RESULT_ARBURST BURST 1 2 }  { m_axi_RESULT_ARLOCK LOCK 1 2 }  { m_axi_RESULT_ARCACHE CACHE 1 4 }  { m_axi_RESULT_ARPROT PROT 1 3 }  { m_axi_RESULT_ARQOS QOS 1 4 }  { m_axi_RESULT_ARREGION REGION 1 4 }  { m_axi_RESULT_ARUSER USER 1 1 }  { m_axi_RESULT_RVALID VALID 0 1 }  { m_axi_RESULT_RREADY READY 1 1 }  { m_axi_RESULT_RDATA DATA 0 32 }  { m_axi_RESULT_RLAST LAST 0 1 }  { m_axi_RESULT_RID ID 0 1 }  { m_axi_RESULT_RUSER USER 0 1 }  { m_axi_RESULT_RRESP RESP 0 2 }  { m_axi_RESULT_BVALID VALID 0 1 }  { m_axi_RESULT_BREADY READY 1 1 }  { m_axi_RESULT_BRESP RESP 0 2 }  { m_axi_RESULT_BID ID 0 1 }  { m_axi_RESULT_BUSER USER 0 1 } } }
}

set busDeadlockParameterList { 
	{ IMG { NUM_READ_OUTSTANDING 16 NUM_WRITE_OUTSTANDING 16 MAX_READ_BURST_LENGTH 16 MAX_WRITE_BURST_LENGTH 16 } } \
	{ RESULT { NUM_READ_OUTSTANDING 16 NUM_WRITE_OUTSTANDING 16 MAX_READ_BURST_LENGTH 16 MAX_WRITE_BURST_LENGTH 16 } } \
}

# RTL port scheduling information:
set fifoSchedulingInfoList { 
}

# RTL bus port read request latency information:
set busReadReqLatencyList { 
	{ IMG 1 }
	{ RESULT 1 }
}

# RTL bus port write response latency information:
set busWriteResLatencyList { 
	{ IMG 1 }
	{ RESULT 1 }
}

# RTL array port load latency information:
set memoryLoadLatencyList { 
}
