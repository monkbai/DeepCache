import re
import math
import os.path

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


def generate_cache_pic(trace_file, skip=500, count=110):
    cache_pic = []
    with open(trace_file, 'r') as f:
        counter = 0
        while counter < skip:
            line = f.readline()
            if not line:
                break
            counter += 1
        while len(cache_pic) < count:
            cur_pic = []
            eof = False
            for i in range(128):
                line = f.readline()
                if not line:
                    eof = True
                    break
                if not (line.startswith("0") or line.startswith("1")):
                    break
                vec = line.strip().split()
                vec = [int(c) for c in vec]
                cur_pic.append(vec)
            if len(cur_pic) < 128 or eof:
                if len(cache_pic) == 0:  # even shorter than 128 lines
                    return
                else:  # else we duplicate data
                    while len(cache_pic) < count:
                        cache_pic = cache_pic + cache_pic
                cache_pic = cache_pic[:count]
            else:
                cache_pic.append(cur_pic)
    # add noise
    cache_pic = np.array(cache_pic, dtype=np.int8)
    cache_pic = np.expand_dims(cache_pic, 1)
    # noise = np.random.rand(count, 1, 128, 64)
    # noise[noise < 0.7] = 0
    # noise[noise >= 0.7] = 1
    # noise = noise.astype(np.int8)
    # cache_pic = np.bitwise_or(cache_pic, noise)

    filename, file_extension = os.path.splitext(trace_file)
    npy_file = filename + ".npy"
    np.save(npy_file, cache_pic)


def generate_cache_pic_new(trace_file, skip=0, count=110, length=128):
    trace_vec = []
    with open(trace_file, 'r') as f:
        counter = 0
        while counter < skip:
            line = f.readline()
            if not line:
                break
            counter += 1
        while True:
            line = f.readline()
            if not line:
                    break
            if (line.startswith("stop") or line.startswith("#eof")):
                break
            vec = line.strip().split()
            vec = [int(c) for c in vec]
            trace_vec.append(vec)
    print(len(trace_vec))
    # decide how to generate numpy array according to the trace length
    cache_pic = []
    if len(trace_vec) <= length:
        return 
    elif len(trace_vec) >= length*55:
        # large stride
        for idx in range(0, len(trace_vec), length):
            if len(cache_pic) >= count:
                break
            if idx + length < len(trace_vec):
                cache_pic.append(trace_vec[idx:idx+length])
        if len(cache_pic) < count:  # 110
            # duplicate the data
            while len(cache_pic) < count:
                cache_pic = cache_pic + cache_pic
            cache_pic = cache_pic[:count]
    else:
        # small stride
        stride = math.ceil((len(trace_vec) - length) / count)
        for idx in range(0, len(trace_vec), stride):
            if len(cache_pic) >= count:
                break
            if idx + length < len(trace_vec):
                cache_pic.append(trace_vec[idx:idx+length])
        if len(cache_pic) < count:  # 110
            # duplicate the data
            while len(cache_pic) < count:
                cache_pic = cache_pic + cache_pic
            cache_pic = cache_pic[:count]

    # add noise <-- no need
    cache_pic = np.array(cache_pic, dtype=np.int8)
    cache_pic = np.expand_dims(cache_pic, 1)
    # noise = np.random.rand(count, 1, 128, 64)
    # noise[noise < 0.7] = 0
    # noise[noise >= 0.7] = 1
    # noise = noise.astype(np.int8)
    # cache_pic = np.bitwise_or(cache_pic, noise)

    filename, file_extension = os.path.splitext(trace_file)
    npy_file = filename + ".npy"
    np.save(npy_file, cache_pic)


class CachePicDataset(Dataset):
    def __init__(self, np_dir, train=True):
        cache_pics = None
        targets = []  # labels
        next_ids = 0

        id2name = {}
        name2id = {}
        name_id = 0

        files = os.listdir(np_dir)
        files.sort()
        for f in files:
            if not f.endswith('.npy'):
                continue
            cache_pic = np.load(os.path.join(np_dir, f))
            assert len(cache_pic) % 2 == 0
            if cache_pics is None:
                cache_pics = cache_pic
            else:
                cache_pics = np.concatenate((cache_pics, cache_pic), axis=0)

            for i in range(next_ids, next_ids+int(len(cache_pic)/2)):
                targets.append(name_id)
            id2name[name_id] = f
            name2id[f] = name_id
            name_id += 1
            next_ids += len(cache_pic)

        self.cache_pics = cache_pics
        self.targets = targets
        self.id2name = id2name
        self.name2id = name2id
        self.next_ids = next_ids
        self.name_id = name_id

        self.train = train

    def __len__(self):
        if self.train:
            return int(self.next_ids/2)
        else:
            return int(self.next_ids/10)

    def __getitem__(self, idx):
        if self.train:
            input1 = self.cache_pics[idx*2]
            input2 = self.cache_pics[idx*2+1]
            return input1.astype(np.float32), input2.astype(np.float32), self.targets[idx], idx
        else:
            input1 = self.cache_pics[idx*10]
            return input1.astype(np.float32), self.targets[idx*5], idx


class LargeCachePicDataset(Dataset):
    def __init__(self, root_dir, train=True, compiler='tvm', as_database=False, paral_ver=False):
        if compiler == 'tvm':
            test_names = ["resnet18-v1-7", "vgg16-7", "resnet18-v1-7-loop", "vgg16-7-loop"]
            # test_names = ["resnet18-v1-7-loop_1", "vgg16-7-loop_1"]
            # test_names = ["resnet18-v2-7", "vgg16-bn-7"]
        elif compiler == 'glow':
            test_names = ["resnet18_v1_7.out", "vgg16_7.out"]

        # if compiler == 'tvm':
        #     test_names = ["inception-v1-3", "shufflenet-3"]
        # elif compiler == 'glow':
        #     test_names = ["inception_v1_12.out", "shufflenet_9.out"]

        cache_pics = []
        targets = []  # labels
        next_ids = 0

        id2name = {}
        name2id = {}
        name_id = 0

        self.paral_ver = paral_ver
        if paral_ver:  # traces are collected in parallel
            for root, dirs, files in os.walk(root_dir, topdown = False):
                for model_name in dirs:
                    if train and model_name in test_names:
                        continue
                    elif not train and model_name not in test_names:
                        continue

                    if 'loop_' in model_name:  # tvm
                        continue
                    elif '.out_' in model_name:  # glow
                        continue
                    # print(model_name)
                    np_dir = os.path.join(root, model_name)
                    files = os.listdir(np_dir)
                    files.sort()
                    for f in files:
                        if not f.endswith('.npy') or "_win" in f or "_multi" in f or "_mat" in f:  # _win, _multi, and _mat are used by encoder
                            continue
                        if "conv" not in f and "dense" not in f and "pool" not in f and 'fc' not in f:
                            continue
                        # print(os.path.join(np_dir, f))  # debug
                        cache_pic = np.load(os.path.join(np_dir, f))
                        # print("cache_pic.shape", cache_pic.shape) # cache_pic.shape (110, 1, 128, 64)
                        
                        cache_pic = torch.from_numpy(cache_pic)
                        # DONE： shuffle snippets
                        rnd = torch.randperm(len(cache_pic))
                        cache_pic = cache_pic[rnd]
                        # DONE： interleave all related snippets
                        cache_pic_list = [cache_pic]
                        expected_view = [110, 1, 128, 64]
                        for idx in range(1, 5):
                            if compiler=='tvm':
                                new_model_name = model_name.replace('-loop', '-loop_{}'.format(idx))
                            else:
                                assert compiler == 'glow'
                                new_model_name = model_name.replace('.out', '.out_{}'.format(idx))
                            assert new_model_name != model_name, "{} {} {}".format(compiler, model_name, new_model_name)
                            new_np_dir = os.path.join(root, new_model_name)
                            if os.path.exists(new_np_dir) and os.path.exists(os.path.join(new_np_dir, f)):
                                new_cache_pic = np.load(os.path.join(new_np_dir, f))
                                new_cache_pic = torch.from_numpy(new_cache_pic)
                                # shuffle snippets
                                rnd = torch.randperm(len(new_cache_pic))
                                new_cache_pic = new_cache_pic[rnd]
                                
                                cache_pic_list.append(new_cache_pic)
                                expected_view[0] += 110
                        # print(len(cache_pic_list), expected_view)
                        tmp_cache_pic = torch.stack(cache_pic_list, dim=1)
                        cache_pic = tmp_cache_pic.view(expected_view)
                        cache_pic = cache_pic.numpy()
                        # print(type(cache_pic))
                        # print("cache_pic.shape", cache_pic.shape, "len", len(cache_pic)) # cache_pic.shape (110*x, 1, 128, 64)
                        # input("continue?")

                        assert (not train) or len(cache_pic) % 2 == 0
                        cache_pics.append(cache_pic)
                        
                        # get the label name of current inputs
                        tmp = os.path.basename(f).split(".")[1]  # rm prefix, e.g., '0010.' 
                        tmp = tmp[:tmp.find("-0x")]
                        name = "{}+{}".format(model_name.replace(".out", ""), tmp) # the label name

                        id2name[name_id] = name
                        name2id[name] = name_id
                        cur_id = name_id
                        name_id += 1
                        
                        # set the target id
                        for i in range(next_ids, next_ids+len(cache_pic)):
                            targets.append(cur_id)
                        
                        next_ids += len(cache_pic)  # next_ids --> length
        else:
            for root, dirs, files in os.walk(root_dir, topdown = False):
                for model_name in dirs:
                    if train and model_name in test_names:
                        continue
                    elif not train and model_name not in test_names:
                        continue

                    np_dir = os.path.join(root, model_name)
                    files = os.listdir(np_dir)
                    files.sort()
                    for f in files:
                        if not f.endswith('.npy') or "_win" in f or "_multi" in f or "_mat" in f:  # _win, _multi, and _mat are used by encoder
                            continue
                        # if "conv" not in f and "dense" not in f and "relu" not in f and "pool" not in f:
                        #     continue
                        if "conv" not in f and "dense" not in f and "pool" not in f and 'fc' not in f:
                            continue
                        # print(os.path.join(np_dir, f))  # debug
                        cache_pic = np.load(os.path.join(np_dir, f))
                        # print(cache_pic.shape)
                        
                        # shuffle snippets
                        cache_pic = torch.from_numpy(cache_pic)
                        rnd = torch.randperm(len(cache_pic))
                        cache_pic = cache_pic[rnd]
                        cache_pic = cache_pic.numpy()

                        assert (not train) or len(cache_pic) % 2 == 0
                        cache_pics.append(cache_pic)
                        
                        # get the label name of current inputs
                        tmp = os.path.basename(f).split(".")[1]  # rm prefix, e.g., '0010.' 
                        tmp = tmp[:tmp.find("-0x")]
                        # model_name = model_name.replace(".out", "")
                        if not as_database:
                            # handle '-loop' suffix
                            mat = re.search("-loop(_\d+)?", model_name)
                            tmp_model_name = model_name
                            if mat:
                                tmp_model_name = model_name.replace(mat.group(), '')
                            mat = re.search("\.out(_\d+)?", model_name)
                            if mat:
                                tmp_model_name = model_name.replace(mat.group(), '')
                            name = "{}+{}".format(tmp_model_name, tmp) # the label name
                        else:
                            # keep `.out`, wait until label checking to handle
                            name = "{}+{}".format(model_name, tmp) # the label name

                        if not as_database:
                            if name not in name2id:
                                # new name -> assign a new id
                                id2name[name_id] = name
                                name2id[name] = name_id
                                cur_id = name_id
                                name_id += 1
                            else:
                                # existing name/id
                                # print('existing name', name)
                                # print(model_name)
                                # input("continue?")
                                cur_id = name2id[name]
                        else:  # as database --> view each trace as independent
                            id2name[name_id] = name
                            name2id[name] = name_id
                            cur_id = name_id
                            name_id += 1
                        
                        # set the target id
                        for i in range(next_ids, next_ids+len(cache_pic)):
                            targets.append(cur_id)
                        
                        next_ids += len(cache_pic)  # next_ids --> length
        
        cache_pics = np.concatenate(cache_pics, axis=0)
        # print(cache_pics.shape)  # debug
        self.cache_pics = cache_pics
        self.targets = targets
        self.id2name = id2name
        self.name2id = name2id
        self.next_ids = next_ids
        self.name_id = name_id

        self.train = train

    def __len__(self):
        if self.train:
            return int(self.next_ids/2)
        else:
            return self.next_ids

    def __getitem__(self, idx):
        if self.train:
            input1 = self.cache_pics[idx*2]
            input2 = self.cache_pics[idx*2+1]
            return input1.astype(np.float32), input2.astype(np.float32), self.targets[idx*2], idx
        else:
            input1 = self.cache_pics[idx]
            return input1.astype(np.float32), self.targets[idx], idx


class LargeCacheMatDataset(Dataset):
    def pic2mat(self, np_arr):
        new_arrs = []
        # (110, 1, 128,64) --> (110, 1, 64, 64)
        # win_size = 128
        for idx in range(len(np_arr)):
            cur_arr = np_arr[idx]
            #print t
            matrix_t = np.zeros((64, 64), dtype=np.float32)
			
            for i in range(64):
                for j in range(i, 64):
                    #if np.var(data[i, t - win:t]) and np.var(data[j, t - win:t]):
                    # matrix_t[i][j] = np.inner(data[i, t - win:t], data[j, t - win:t])/(win) # rescale by win
                    tmp1 = cur_arr[0, :, i].sum()  # 128 0 or 1
                    tmp2 = cur_arr[0, :, j].sum()
                    # print(tmp1, tmp2)
                    # input("continue?")
                    matrix_t[i][j] = (tmp1 + tmp2)  # / (win)
                    matrix_t[j][i] = matrix_t[i][j]
            matrix_t.resize(1, 64, 64)
            # print(matrix_t)
            # input("continue?")
            new_arrs.append(matrix_t)
        return np.array(new_arrs)

    def __init__(self, root_dir, train=True, compiler='tvm'):
        if compiler == 'tvm':
            test_names = ["resnet18-v1-7", "vgg16-7"]
            # test_names = ["resnet18-v2-7", "vgg16-bn-7"]
        elif compiler == 'glow':
            test_names = ["resnet18_v1_7.out", "vgg16_7.out"]

        # if compiler == 'tvm':
        #     test_names = ["inception-v1-3", "shufflenet-3"]
        # elif compiler == 'glow':
        #     test_names = ["inception_v1_12.out", "shufflenet_9.out"]

        cache_pics = []
        targets = []  # labels
        next_ids = 0

        id2name = {}
        name2id = {}
        name_id = 0

        for root, dirs, files in os.walk(root_dir, topdown = False):
            for model_name in dirs:
                if train and model_name in test_names:
                    continue
                elif not train and model_name not in test_names:
                    continue

                np_dir = os.path.join(root, model_name)
                files = os.listdir(np_dir)
                files.sort()
                for f in files:
                    if not f.endswith('.npy') or "_win" in f or "_multi" in f or "_mat" in f:  # _win, _multi, and _mat are used by encoder
                        continue
                    # if "conv" not in f and "dense" not in f and "relu" not in f and "pool" not in f:
                    #     continue
                    if "conv" not in f and "dense" not in f and "pool" not in f and 'fc' not in f:
                        continue
                    # print(os.path.join(np_dir, f))  # debug
                    cache_pic = np.load(os.path.join(np_dir, f))
                    # print(cache_pic.shape)  # (110, 1, 128, 64)
                    # print(type(cache_pic))
                    cache_pic = self.pic2mat(cache_pic)
                    # print(cache_pic.shape)  # (110, 1, 128, 64)
                    # print(type(cache_pic))
                    # input("Continue?")
                    assert (not train) or len(cache_pic) % 2 == 0
                    cache_pics.append(cache_pic)
                    
                    for i in range(next_ids, next_ids+len(cache_pic)):
                        targets.append(name_id)

                    tmp = os.path.basename(f).split(".")[1]  # rm 0010. prefix
                    tmp = tmp[:tmp.find("-0x")]
                    model_name = model_name.replace(".out", "")
                    name = "{}+{}".format(model_name, tmp)
                    id2name[name_id] = name
                    name2id[name] = name_id
                    name_id += 1
                    next_ids += len(cache_pic)
        
        cache_pics = np.concatenate(cache_pics, axis=0)
        # print(cache_pics.shape)  # debug
        self.cache_pics = cache_pics
        self.targets = targets
        self.id2name = id2name
        self.name2id = name2id
        self.next_ids = next_ids
        self.name_id = name_id

        self.train = train

    def __len__(self):
        if self.train:
            return int(self.next_ids/2)
        else:
            return self.next_ids

    def __getitem__(self, idx):
        if self.train:
            input1 = self.cache_pics[idx*2]
            input2 = self.cache_pics[idx*2+1]
            return input1.astype(np.float32), input2.astype(np.float32), self.targets[idx*2], idx
        else:
            input1 = self.cache_pics[idx]
            return input1.astype(np.float32), self.targets[idx], idx


def preprocess_traces(log_dir: str, skip=500, count=110):
    files = os.listdir(log_dir)
    files.sort()
    for f in files:
        if f.endswith(".log"):  # and f.startswith("0x") 
            # if f.replace(".log", ".npy") in files:
            #     continue
            f_path = os.path.join(log_dir, f)
            # generate_cache_pic(f_path, skip, count)
            print(f_path, end=" ")
            generate_cache_pic_new(f_path, skip, count, length=128)  # length=256)


def preprocess_traces_dir(log_dir: str, skip=100):
    for root, dirs, files in os.walk(log_dir, topdown = False):
        for name in dirs:
            # print(os.path.join(root, name))
            preprocess_traces(os.path.join(root, name), skip=skip, count=110)



if __name__ == '__main__':
    # preprocess_traces_dir("/export/d2/zliudc/VirtualEnv/DeepCache/pin_dataset/pin_dataset_glow/")
    # preprocess_traces_dir("/export/ssd1/zliudc/VirtualEnv/DeepCache/cache_dataset/cache_dataset_tvm/")
    # preprocess_traces_dir("/export/ssd1/zliudc/VirtualEnv/DeepCache/cache_dataset/cache_dataset_glow/")
    # preprocess_traces_dir("/export/ssd1/zliudc/VirtualEnv/DeepCache/pin_dataset/pin_dataset_glow_100/")
    # preprocess_traces_dir("/export/d2/zliudc/VirtualEnv/DeepCache/mitigate/oram_traces/")
    preprocess_traces_dir("/export/d2/zliudc/VirtualEnv/DeepCache/mitigate/obfs_traces/")
    # ======= LLC experiments =======
    # preprocess_traces_dir("/export/ssd1/zliudc/VirtualEnv/DeepCache/cache_dataset/cache_dataset_llc_tvm/")
    
    # preprocess_traces_dir("/export/d2/zliudc/VirtualEnv/DeepCache/cache_dataset/LLC_dataset_tvm/")
    # preprocess_traces_dir("/export/d2/zliudc/VirtualEnv/DeepCache/cache_dataset/LLC_dataset_glow/")
    # preprocess_traces_dir("./cache_dataset/cache_dataset_tvm/")
    # preprocess_traces_dir("./cache_dataset/cache_dataset_glow/", skip=0)
    # preprocess_traces_dir("./cache_dataset/cache_dataset_tvm/", skip=0)
    exit(0)
    
    # preprocess_traces_dir("/export/d2/zliudc/VirtualEnv/DeepCache/pin_dataset/")
    data = LargeCachePicDataset("/export/d2/zliudc/VirtualEnv/DeepCache/pin_dataset/")
    # print(data.id2name)
    print(len(data))
    print(data.id2name[0])

    # data = CachePicDataset("/export/d2/zliudc/VirtualEnv/DeepCache/cache_log/")
    # print(len(data))
    
    # generate_cache_pic("/home/wasmith/Music/pinTrace/cache_log/0x403460-0x403986.log")
    # preprocess_traces("/export/d2/zliudc/VirtualEnv/DeepCache/cache_log/")
