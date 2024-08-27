import os
import json

import torch
import torch.nn as nn
import torch.functional as F
from tqdm import tqdm
from encoder.model.conv_encoder import RED

from encoder.utils.matrix_generator import generate_signature_matrix_node_fast, generate_train_test_data
from encoder.utils.data import load_data_multi, load_data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
batch_size = 32
model_path_prefix = "./encoder/checkpoints/"


def read_trace(trace_file: str):
    trace_vec = []
    with open(trace_file, 'r') as f:
        counter = 0
        while True:
            line = f.readline()
            if not line:
                break
            if not (line.startswith("0") or line.startswith("1")):
                continue
            # vec = line.strip().split()
            # vec = [int(c) for c in vec]
            # trace_vec.append(vec)
            counter += 1
    # trace_vec = np.array(trace_vec, dtype=np.float32)  # TODO: 64 float -> 4 float, hopefully it will preform better
    return counter


def preprocess_data(trace_file: str):
    assert trace_file.endswith(".log"), "the trace file path should end with .log"
    
    # will generate 3 *_win[x].py files
    generate_signature_matrix_node_fast(trace_file)
	
    # will generate *_multi5.py, 5 is the step length
    generate_train_test_data(trace_file[:-4])


def get_encoder():
    red = RED(3, 256, batch_size)
    return red


def get_dataloader(log_path: str):
    assert log_path.endswith(".log"), "log path should ends with '.log'"
    trace_file_prefix = log_path[:-4]
    dataloader = load_data_multi(trace_file_prefix, batch_size=batch_size, size=5000)
    return dataloader


def train(red, dataloader, model_path: str):
    red.to(device)
    # train stage
    # red.load_state_dict(torch.load("./checkpoints/model1.pth"))
    optimizer = torch.optim.Adam(red.parameters(), lr=0.0002)
    train_internal(dataloader, red, optimizer, epochs=1, device=device)
    print("Saving model to {} ...".format(model_path))
    torch.save(red.state_dict(), model_path)

    return red


def train_internal(dataLoader, model, optimizer, epochs, device):
    global batch_size

    model = model.to(device)
    print("------training on {}-------".format(device))
    for epoch in range(epochs):
        train_l_sum, n = 0.0, 0
        for x in tqdm(dataLoader):
            if x.shape[0] < batch_size:
                continue  # skip the last batch

            if batch_size > 1:
                shp = x.shape
                x_in = x.reshape([batch_size*5, shp[2], shp[3], shp[4]])
            else:
                x_in = x.squeeze()
            # print(x_in.shape)
            x_in = x_in.to(device)
            out = model(x_in)
            # print(out.shape)
            # input("debug")
            
            if batch_size == 1:
                target = x[-1].unsqueeze(0).to(device)
                # target = x
            else:
                target = x[:,-1].to(device)

            l = torch.mean((out - target) ** 2)
            train_l_sum += l
            # print(l.item())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            n += 1
            # print("[Epoch %d/%d][Batch %d/%d] [loss: %f]" % (epoch+1, epochs, n, len(dataLoader), l.item()))

        print("[Epoch %d/%d] [loss: %f]" % (epoch + 1, epochs, train_l_sum / n))


def test(dataLoader, model, max_len=-1):
    global batch_size

    print("------Testing-------")
    index = 30
    loss_list = []

    # reconstructed_data_path = "./data/matrix_data/reconstructed_data/"    
    # if max_len == -1:
    #     max_len = len(dataLoader.dataset)
    # else:
    #     max_len = min(max_len, len(dataLoader.dataset))

    model.to(device)
    with torch.no_grad():
        for x in tqdm(dataLoader):  # for idx in range(len(dataLoader.dataset)):  # for x in dataLoader:
            # x = dataLoader.dataset[idx]
            if x.shape[0] < batch_size:
                continue  # skip the last batch
            
            if batch_size > 1:
                shp = x.shape
                x_in = x.reshape([batch_size*5, shp[2], shp[3], shp[4]])
            else:
                x_in = x.squeeze()
            x_in = x_in.to(device)
            reconstructed_matrix = model(x_in)

            if batch_size == 1:
                target = x[-1].unsqueeze(0).to(device)
                # target = x
            else:
                target = x[:,-1].to(device)
            # target = x

            # compare reconstructed with original
            if batch_size == 1:
                loss = torch.mean((reconstructed_matrix - target) ** 2)
                loss_list.append([index, loss.item()])
                indx += 1
            else:
                tmp = (reconstructed_matrix - target)
                for idx in range(batch_size):
                    loss = torch.mean(tmp[idx] ** 2)
                    loss_list.append([index, loss.item()])
                    index += 1 
    return loss_list


def predict(red, dataloader, log_path):
    # test stage
    red.to(device)
    
    output_json_path = log_path[:log_path.rfind(".")] + ".json"
    if os.path.exists(output_json_path):
        with open(output_json_path, 'r') as f:
            output_dict = json.load(f)
            trace_len = output_dict["trace_len"]
            loss_list = output_dict["loss_list"]    
    else:
        trace_len = read_trace(log_path)
        loss_list = test(dataloader, red, max_len=5000)    
        output_dict = {}
        output_dict["trace_len"] = trace_len
        output_dict["loss_list"] = loss_list
        with open(output_json_path, 'w') as f:
            json.dump(output_dict, f, indent=2)

    return trace_len, loss_list


def get_seg_ids(loss_list: list):
    seg_ids = []
    
    loss_values = [x[1] for x in loss_list]  # idx, val
    avg_loss = sum(loss_values) / len(loss_values)
    # print("avg_loss", avg_loss)
    for idx, loss in loss_list:
        if loss > avg_loss * 2:
            seg_ids.append(idx)
    
    first_id = 1
    for idx in range(len(seg_ids)):
        sid = seg_ids[idx]
        if sid > 50 and idx > 0 and idx < len(seg_ids)-1 and seg_ids[idx]-seg_ids[idx-1] > 30 and seg_ids[idx+1]-seg_ids[idx] == 1:
            first_id = sid
            break
    return first_id


def get_loop_num_with_log(log_path: str, prefix="", load_pretrained=True):
    log_path = os.path.abspath(log_path)
    model_name = os.path.basename(log_path)
    model_name = prefix + "." + model_name[:model_name.rfind(".")] + ".net"
    model_name = os.path.join(model_path_prefix, model_name)

    # Step 1: train/load Convolutional RED model
    print("Start training model...")
    preprocess_data(log_path)
    dataloader = get_dataloader(log_path)
    model = get_encoder()
    
    if not os.path.exists(model_name) or not load_pretrained:
        # train the model
        model = train(model, dataloader, model_name)
    else:
        # load the model
        if device == "cpu":
            model.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load(model_name))
        model.to(device)
        model.eval()
    # input("debug")
    # Step 2: get segmentation points
    print("Segmenting trace...")
    trace_len, loss_list = predict(model, dataloader, log_path)
    first_id = get_seg_ids(loss_list)
    if first_id > (trace_len/2):
        first_id = 1
    return [trace_len, first_id]
