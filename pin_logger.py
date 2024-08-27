#!/usr/bin/python3
from subprocess import Popen, PIPE, STDOUT

import os
import time
import subprocess

import config


class cd:
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


def cmd(commandline):
    with cd(project_dir):
        print(commandline)
        status, output = subprocess.getstatusoutput(commandline)
        # print(output)
        return status, output


def run(prog_path):
    with cd(project_dir):
        # print(prog_path)
        proc = subprocess.Popen(prog_path, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = proc.communicate()  # stderr: summary, stdout:  each statement
        return stdout, stderr


project_dir = './'

pin_home = config.pin_home

mypintool_dir = config.mypintool_dir

trace_log_cmd = pin_home + "pin -t " + \
                       mypintool_dir + "obj-intel64/TraceLogger.so -o {} -func_start {} -func_end {} -- {} {}"

kernel_log_cmd = pin_home + "pin -t " + \
                       mypintool_dir + "obj-intel64/TraceLogger_kernel.so -o {} -func_start {} -func_end {} -- {} {}"

oram_log_cmd = pin_home + "pin -t " + \
                       mypintool_dir + "obj-intel64/ORAMTrace.so -o {} -func_start {} -func_end {} -- {} {}"

write_log_cmd = pin_home + "pin -t " + \
                       mypintool_dir + "obj-intel64/WriteLogger.so -o {} -func_start {} -func_end {} -- {} {}"

ciphertext_log_cmd = pin_home + "pin -t " + \
                       mypintool_dir + "obj-intel64/CiphertextLogger.so -o {} -func_start {} -func_end {} -- {} {}"
inst_log_cmd = pin_home + "pin -t " + \
                       mypintool_dir + "obj-intel64/InstLogger.so -o {} -func_start {} -func_end {} -- {} {}"

obfus_log_cmd = pin_home + "pin -t " + \
                       mypintool_dir + "obj-intel64/ObfusSim.so -o {} -func_start {} -func_end {} -insert_point {} -- {} {}"

compile_tool_cmd = "make obj-intel64/{}.so TARGET=intel64"
tools_list = ["TraceLogger", # log all memory access address in a function (but only [12:7] bits)
                             # 64 byte each cache line -> low 6 bit (block offset)
                             # 64 l1 cache sets -> 12-7 bits (set index)
                             # print every 100 memory accesses
              "TraceLogger_kernel",
              "ORAMTrace",
              "WriteLogger",
              "CiphertextLogger",
              "InstLogger",
              "ObfusSim",
              ]


def compile_all_tools():
    global project_dir
    for tool_name in tools_list:
        print("copying {} source code to MyPinTool dir...".format(tool_name))
        status, output = cmd("cp pin_tool/{}.cpp {}".format(tool_name, mypintool_dir))
        if status != 0:
            print(output)

    project_dir_backup = project_dir
    project_dir = mypintool_dir
    for tool_name in tools_list:
        print("compiling {}...".format(tool_name))
        status, output = cmd(compile_tool_cmd.format(tool_name))
        if status != 0:
            print(output)
    project_dir = project_dir_backup


def trace_log(dnn_exe, dnn_input, func_start, func_end, log_path):
    global project_dir
    project_dir_backup = project_dir
    project_dir = mypintool_dir
    # ------- set project_dir before instrumentation

    dnn_exe = os.path.abspath(dnn_exe)
    dnn_input = os.path.abspath(dnn_input)
    log_path = os.path.abspath(log_path)
    status, output = cmd(trace_log_cmd.format(log_path, func_start, func_end, dnn_exe, dnn_input))

    # ------- end reset project_dir
    project_dir = project_dir_backup


def oram_log(dnn_exe, dnn_input, func_start, func_end, log_path):
    global project_dir
    project_dir_backup = project_dir
    project_dir = mypintool_dir
    # ------- set project_dir before instrumentation

    dnn_exe = os.path.abspath(dnn_exe)
    dnn_input = os.path.abspath(dnn_input)
    log_path = os.path.abspath(log_path)
    status, output = cmd(oram_log_cmd.format(log_path, func_start, func_end, dnn_exe, dnn_input))

    # ------- end reset project_dir
    project_dir = project_dir_backup


def obfus_log(dnn_exe, dnn_input, func_start, func_end, log_path, insert_point):
    global project_dir
    project_dir_backup = project_dir
    project_dir = mypintool_dir
    # ------- set project_dir before instrumentation

    dnn_exe = os.path.abspath(dnn_exe)
    dnn_input = os.path.abspath(dnn_input)
    log_path = os.path.abspath(log_path)
    status, output = cmd(obfus_log_cmd.format(log_path, func_start, func_end, insert_point, dnn_exe, dnn_input))
    print(output)
    # ------- end reset project_dir
    project_dir = project_dir_backup


def kernel_log(dnn_exe, dnn_input, func_start, func_end, log_path):
    global project_dir
    project_dir_backup = project_dir
    project_dir = mypintool_dir
    # ------- set project_dir before instrumentation

    dnn_exe = os.path.abspath(dnn_exe)
    dnn_input = os.path.abspath(dnn_input)
    log_path = os.path.abspath(log_path)
    status, output = cmd(kernel_log_cmd.format(log_path, func_start, func_end, dnn_exe, dnn_input))

    # ------- end reset project_dir
    project_dir = project_dir_backup


def write_log(dnn_exe, dnn_input, func_start, func_end, log_path):
    status, output = cmd(write_log_cmd.format(log_path, func_start, func_end, dnn_exe, dnn_input))


def inst_log(dnn_exe, dnn_input, func_start, func_end, log_path):
    global project_dir
    project_dir_backup = project_dir
    project_dir = mypintool_dir
    # ------- set project_dir before instrumentation

    dnn_exe = os.path.abspath(dnn_exe)
    dnn_input = os.path.abspath(dnn_input)
    log_path = os.path.abspath(log_path)

    status, output = cmd(inst_log_cmd.format(log_path, func_start, func_end, dnn_exe, dnn_input))
    # ------- end reset project_dir
    project_dir = project_dir_backup


def ciphertext_log(dnn_exe, dnn_input, func_start, func_end, log_path):
    global project_dir
    project_dir_backup = project_dir
    project_dir = mypintool_dir
    # ------- set project_dir before instrumentation

    dnn_exe = os.path.abspath(dnn_exe)
    dnn_input = os.path.abspath(dnn_input)
    log_path = os.path.abspath(log_path)
    status, output = cmd(ciphertext_log_cmd.format(log_path, func_start, func_end, dnn_exe, dnn_input))
    # ------- end reset project_dir
    project_dir = project_dir_backup


if __name__ == '__main__':
    compile_all_tools()
    exit(0)


    dnn_exe = "/export/d2/zliudc/VirtualEnv/ONNX_Zoo/DNN_exe/TVM-0.12/resnet18-v1-7"
    dnn_input = "./examples/resnet18_tvm_O3/cat.bin"
    func_start = '0x406f80'  # tvmgen_default_fused_nn_contrib_conv2d_NCHWc_add
    func_end = '0x409406'
    insert_point = "0x40877e"
    obfus_log(dnn_exe, dnn_input, func_start, func_end, "./tmp-1.log", insert_point)
    exit(0)

    # test
    # dnn_exe = "./examples/resnet18-glow2022/resnet18_v1_7.out"
    # func_start = "0x4047b0"  # libjit_conv2d_f2
    # func_end = "0x404e1f"

    dnn_exe = "./examples/resnet18_tvm_O3/resnet18_tvm_O3"
    func_start = "0x436630"  # tvmgen_default_fused_nn_contrib_conv2d_NCHWc_add_nn_relu_4_compute_
    func_end = "0x439abf"

    # dnn_exe = "./examples/resnet18_tvm_O0/resnet18_tvm_O0"
    # # func_start = "0x41F110"  # tvmgen_default_fused_nn_conv2d_4
    # # func_end = "0x425248"
    # # func_start = "0x433EF0"  # tvmgen_default_fused_nn_relu
    # # func_end = "0x4344DB"
    # func_start = "0x401A70"  # whole
    # func_end = "0x436065"
    dnn_input = "./examples/resnet18_tvm_O3/cat.bin"

    log_path = "./tmp.log"
    # write_log(dnn_exe, dnn_input, func_start, func_end, log_path)
    ciphertext_log(dnn_exe, dnn_input, func_start, func_end, log_path)

    # =======

    # dnn_exe = "/export/d2/zliudc/VirtualEnv/ONNX_Zoo/DNN_exe/TVM-0.12/resnet18-v2-7"
    # dnn_input = "./examples/resnet18_tvm_O3/cat.bin"
    # func_start = '0x408880'  # tvmgen_default_fused_nn_contrib_conv2d_NCHWc
    # func_end = '0x409764'
    # inst_log(dnn_exe, dnn_input, func_start, func_end, "./tmp.log")

    dnn_exe = "/export/d2/zliudc/VirtualEnv/ONNX_Zoo/DNN_exe/TVM-0.12/resnet18-v1-7"
    dnn_input = "./examples/resnet18_tvm_O3/cat.bin"
    func_start = '0x406f80'  # tvmgen_default_fused_nn_contrib_conv2d_NCHWc_add
    func_end = '0x409406'
    inst_log(dnn_exe, dnn_input, func_start, func_end, "./tmp-1.log")

