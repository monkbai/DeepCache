import os
import onnx
import json
import subprocess


class cd:
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


def cmd(commandline, under_dir=""):
    if len(under_dir) == 0:
        under_dir = project_dir
    with cd(under_dir):
        print(commandline)
        status, output = subprocess.getstatusoutput(commandline)
        # print(output)
        return status, output


def run(prog_path, under_dir=""):
    if len(under_dir) == 0:
        under_dir = project_dir
    with cd(under_dir):
        # print(prog_path)
        proc = subprocess.Popen(prog_path, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = proc.communicate()  # stderr: summary, stdout:  each statement
        return stdout, stderr


project_dir = './'

def get_input_name(onnx_path: str):
    model = onnx.load(onnx_path)
    output = model.graph.output
    output =[node for node in model.graph.output]
    # print(output)
    # print(output[0].__dir__())
    # print(type(output[0]))

    input_all = model.graph.input
    input_initializer = model.graph.initializer
    input_initializer = [node.name for node in input_initializer]
    net_feed_input = []
    for v in input_all:
        if v.name not in input_initializer:
            net_feed_input.append(v)
    # print(net_feed_input)
    # print(input_initializer)
    
    # print(input_all[0].__dir__())
    # print(input_all[0].ListFields())
    # print(input_all[0].type.tensor_type.shape)
    # print(type(input_all[0].type.tensor_type.shape))
    dim_list = []
    for dim_value in net_feed_input[0].type.tensor_type.shape.dim:
        # print(dim_value.dim_value)
        if dim_value.dim_value == 0:
            dim_list.append(1)
        else:
            dim_list.append(dim_value.dim_value)

    dim_list2 = []
    for dim_value in output[0].type.tensor_type.shape.dim:
        # print(dim_value.dim_value)
        if dim_value.dim_value == 0:
            dim_list2.append(1)
        else:
            dim_list2.append(dim_value.dim_value)

    return net_feed_input[0].name, dim_list, output[0].name, dim_list2


def compile_all_onnx(s_time):
    import time
    time.sleep(s_time)

    onnx_dir = "../onnx_zoo/"
    rm = False
    
    files = os.listdir(onnx_dir)
    files.sort()
    for f in files:
        if f.endswith(".onnx") and "mnist" not in f:
            model_name = os.path.splitext(f)[0]
            f_path = os.path.join(os.path.abspath(onnx_dir), f)

            # copy
            if rm:
                status, output = cmd("rm -r ./{}/".format(model_name))   
            elif os.path.exists("./{}/".format(model_name)):
                continue
            
            status, output = cmd("cp -r ./template/ ./{}/".format(model_name))
            out, err = run("make clean", "./{}/".format(model_name))
            status, output = cmd("cp {} ./{}/".format(f_path, model_name))

            # modify compilation script
            input_name, input_shape, output_name, output_shape = get_input_name(f_path)
            script_path = "./{}/build_model.py".format(model_name)
            with open(script_path, 'r') as fr:
                txt = fr.read()
            txt = txt.replace("dshape = (1, 3, 224, 224)", "dshape = {}".format(tuple(input_shape)))
            txt = txt.replace('shape_dict = {"data": dshape}', 'shape_dict = {"%s": dshape}' % (input_name))
            txt = txt.replace("model_path = './vgg16-7.onnx'", "model_path = './{}'".format(f))
            with open(script_path, 'w') as fw:
                fw.write(txt)

            # modify main source code
            main_path = "./{}/demo_static.c".format(model_name)
            with open(main_path, 'r') as fr:
                txt = fr.read()
            txt = txt.replace('#define OUTPUT_LEN 1000', '#define OUTPUT_LEN {}'.format(output_shape[1]))
            txt = txt.replace('input.ndim = 4;', 'input.ndim = {};'.format(len(input_shape)))
            txt = txt.replace('tvm_runtime_set_input(handle, "data", &input);', 'tvm_runtime_set_input(handle, "%s", &input);' % (input_name))
            txt = txt.replace('int64_t shape[4] = {1, 3, 224, 224};', 'int64_t shape[%d] = {%d, %d, %d, %d};' % (len(input_shape), input_shape[0], input_shape[1], input_shape[2], input_shape[3]))
            if len(output_shape) != 2:
                txt = txt.replace('output.ndim = 2;', 'output.ndim = {};'.format(len(output_shape)))
                txt = txt.replace('int64_t out_shape[2] = {1, OUTPUT_LEN};', 'int64_t out_shape[%d] = {%d, %d, %d, %d};' % (len(output_shape), output_shape[0], output_shape[1], output_shape[2], output_shape[3]))
            with open(main_path, 'w') as fw:
                fw.write(txt)

            # build model
            out, err = run("/export/d2/zliudc/TOOLS/tvm.sh && make demo_static", "./{}/".format(model_name))
            # print(out)
            # print(err)

            # check DNN exe
            out, err = run("demo_static cat.bin", "./{}/build/".format(model_name))
            print(model_name, out)
            # break


def check_exe():
    exe_dir = "./"
    files = os.listdir(exe_dir)
    files.sort()

    count = 0
    all = 0
    for d in files:
        if "template" in d:
            continue
        d_path = os.path.join(exe_dir, d)
        if os.path.isdir(d_path):
            build_dir = os.path.join(d_path, "build/")
            # print(build_dir)
            all += 1
            status, output = cmd("./demo_static cat.bin", build_dir)
            if status == 0:
                print(d, 'exist')
                count += 1
            else:
                print(d, output)
                # if "template" not in d:
                #     status, output = cmd("rm -r {}".format(d_path), "./")
            # break
    print("{}/{}".format(count, all))


def mv_exe():
    exe_dir = "./"
    files = os.listdir(exe_dir)
    files.sort()

    count = 0
    all = 0
    for d in files:
        if "template" in d:
            continue
        d_path = os.path.join(exe_dir, d)
        if os.path.isdir(d_path):
            build_dir = os.path.join(d_path, "build/")
            # print(build_dir)
            all += 1
            status, output = cmd("./demo_static cat.bin", build_dir)
            if status == 0:
                print(d, 'exist')
                status, output = cmd("cp ./demo_static ../../../DNN_exe/TVM-0.12/{}".format(d), build_dir)
                count += 1
            else:
                print(d, output)
                # if "template" not in d:
                #     status, output = cmd("rm -r {}".format(d_path), "./")
            # break
    print("{}/{}".format(count, all))


def get_labels_false():
    """
    Only get the output shape
    """
    overall_labels = {}
    exe_dir = "./"
    files = os.listdir(exe_dir)
    files.sort()

    count = 0
    all = 0
    for d in files:
        if "template" in d:
            continue
        d_path = os.path.join(exe_dir, d)
        if os.path.isdir(d_path):
            json_path = os.path.join(d_path, "build/graph_c.json")
            
            with open(json_path) as f:
                tmp_dict = json.load(f)
            nodes = tmp_dict["nodes"]
            shapes = tmp_dict["attrs"]["shape"][1]
            
            for i in range(len(nodes)):
                if "tvmgen" in nodes[i]["name"]:
                    name = "{}+{}".format(d, nodes[i]["name"])
                    overall_labels[name] = shapes[i]
    
    return overall_labels


def get_labels():
    """
    Try to get the weights shape, which is more important
    """
    overall_labels = {}
    exe_dir = "./"
    files = os.listdir(exe_dir)
    files.sort()

    count = 0
    all = 0
    for d in files:
        if "template" in d:
            continue
        d_path = os.path.join(exe_dir, d)
        if os.path.isdir(d_path):
            json_path = os.path.join(d_path, "build/graph_c.json")
            
            with open(json_path) as f:
                tmp_dict = json.load(f)
            nodes = tmp_dict["nodes"]
            shapes = tmp_dict["attrs"]["shape"][1]
            
            for i in range(len(nodes)):
                shape_label = None
                if "tvmgen" in nodes[i]["name"] and "conv" in nodes[i]["name"]:
                    # find the weights index
                    input_ids = nodes[i]["inputs"]
                    for id in input_ids:
                        id = id[0]
                        if len(shapes[id]) == 6:
                            shape_label = shapes[id]
                            break
                elif "tvmgen" in nodes[i]["name"] and "dense" in nodes[i]["name"]:
                    # find the weights index
                    input_ids = nodes[i]["inputs"]
                    for id in input_ids:
                        id = id[0]
                        if len(shapes[id]) == 3:
                            shape_label = shapes[id]
                elif "tvmgen" in nodes[i]["name"]:
                    shape_label = shapes[i]
                    
                if shape_label:
                    name = "{}+{}".format(d, nodes[i]["name"])
                    overall_labels[name] = shape_label
    
    return overall_labels


def get_labels_new():
    """
    not only the weights shape, bug also the corresponding func_name
    """
    overall_labels = {}
    exe_dir = "./"
    files = os.listdir(exe_dir)
    files.sort()

    count = 0
    all = 0
    for d in files:
        if "template" in d:
            continue
        d_path = os.path.join(exe_dir, d)
        if os.path.isdir(d_path):
            json_path = os.path.join(d_path, "build/graph_c.json")
            
            with open(json_path) as f:
                tmp_dict = json.load(f)
            nodes = tmp_dict["nodes"]
            shapes = tmp_dict["attrs"]["shape"][1]
            
            for i in range(len(nodes)):
                shape_label = None
                if "tvmgen" in nodes[i]["name"] and "conv" in nodes[i]["name"]:
                    # find the weights index
                    input_ids = nodes[i]["inputs"]
                    for id in input_ids:
                        id = id[0]
                        if len(shapes[id]) == 6 or True:  # skip this check
                            shape_label = shapes[id]
                            break
                elif "tvmgen" in nodes[i]["name"] and "dense" in nodes[i]["name"]:
                    # find the weights index
                    input_ids = nodes[i]["inputs"]
                    for id in input_ids:
                        id = id[0]
                        if len(shapes[id]) == 3:
                            shape_label = shapes[id]
                elif "tvmgen" in nodes[i]["name"]:
                    shape_label = shapes[i]
                    
                if "attrs" in nodes[i]:
                  func_name = nodes[i]["attrs"]["func_name"]
                if shape_label:
                    name = "{}+{}".format(d, nodes[i]["name"])
                    func_name = "{}+{}".format(d, func_name)
                    overall_labels[name] = (func_name, shape_label)
    
    return overall_labels


if __name__ == '__main__':
    # print(get_input_name("../onnx_zoo/densenet-3.onnx"))
    # check_exe()
    # mv_exe()
    
    
    # labels = get_labels()
    # with open("labels.json", "w") as f:
    #     json.dump(labels, f, sort_keys=True, indent=4)
    # exit(0)

    labels = get_labels_new()
    with open("labels_new.json", "w") as f:
        json.dump(labels, f, sort_keys=True, indent=2)
    exit(0)


    # print(get_input_name("../onnx_zoo/vgg16-7.onnx"))
    import multiprocessing
    pool = multiprocessing.Pool(processes = 15)
    for i in range(15):
        pool.apply_async(func=compile_all_onnx, args=(i*5,))
    pool.close()
    pool.join()
    pool.terminate()
