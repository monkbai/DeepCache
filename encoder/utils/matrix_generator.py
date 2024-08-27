import numpy as np
import argparse
import pandas as pd
import os, sys
import math
import scipy
#import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy import spatial
import itertools as it
import string
import re



# parser = argparse.ArgumentParser(description = 'Signature Matrix Generator')
# parser.add_argument('--ts_type', type = str, default = "node",
# 				   help = 'type of time series: node or link')
# parser.add_argument('--step_max', type = int, default = 5,
# 				   help = 'maximum step in ConvLSTM')
# parser.add_argument('--gap_time', type = int, default = 1, # stride width...
# 				   help = 'gap time between each segment')
# # parser.add_argument('--gap_time', type = int, default = 10, # stride width...
# # 				   help = 'gap time between each segment')
# parser.add_argument('--win_size', type = int, default = [10, 20, 30],
# 				   help = 'window size of each segment')
# parser.add_argument('--min_time', type = int, default = 0,
# 				   help = 'minimum time point')
# parser.add_argument('--max_time', type = int, default = 5040,
# 				   help = 'maximum time point')
# parser.add_argument('--train_start_point',  type = int, default = 0,
# 						help = 'train start point')
# parser.add_argument('--train_end_point',  type = int, default = 8000,
# 						help = 'train end point')
# parser.add_argument('--test_start_point',  type = int, default = 8000,
# 						help = 'test start point')
# parser.add_argument('--test_end_point',  type = int, default = 20000,
# 						help = 'test end point')
# parser.add_argument('--raw_data_path', type = str, default = './data/synthetic_data_with_anomaly-s-1.csv',
# 				   help='path to load raw data')
# parser.add_argument('--save_data_path', type = str, default = './data/',
# 				   help='path to save data')

# args = parser.parse_args()
# print(args)

ts_type = "node"  # args.ts_type
step_max = 5  # args.step_max
min_time = 0  # args.min_time
max_time = 5040  # args.max_time
gap_time = 1  # args.gap_time
win_size = [10, 20, 30]  # args.win_size

train_start = 0  # args.train_start_point
train_end = 8000  # args.train_end_point
test_start = 8000 # args.test_start_point
test_end = 20000  # args.test_end_point

raw_data_path = './data/synthetic_data_with_anomaly-s-1.csv'  # args.raw_data_path
save_data_path = './data/'  # args.save_data_path

ts_colname="agg_time_interval"
agg_freq='5min'

matrix_data_path = save_data_path + "matrix_data/"
# if not os.path.exists(matrix_data_path):
# 	os.makedirs(matrix_data_path)


def generate_signature_matrix_node_old():
	data = np.array(pd.read_csv(raw_data_path, header = None), dtype=np.float64)
	sensor_n = data.shape[0]
	# min-max normalization
	max_value = np.max(data, axis=1)
	min_value = np.min(data, axis=1)
	data = (np.transpose(data) - min_value)/(max_value - min_value + 1e-6)
	data = np.transpose(data)

	#multi-scale signature matix generation
	for w in range(len(win_size)):
		matrix_all = []
		win = win_size[w]
		print ("generating signature with window " + str(win) + "...")
		for t in range(min_time, max_time, gap_time):
			#print t
			matrix_t = np.zeros((sensor_n, sensor_n))
			if t >= win_size[2]:
				for i in range(sensor_n):
					for j in range(i, sensor_n):
						#if np.var(data[i, t - win:t]) and np.var(data[j, t - win:t]):
						matrix_t[i][j] = np.inner(data[i, t - win:t], data[j, t - win:t])/(win) # rescale by win
						matrix_t[j][i] = matrix_t[i][j]
			matrix_all.append(matrix_t)
		path_temp = matrix_data_path + "matrix_win_" + str(win)
		np.save(path_temp, matrix_all)
		del matrix_all[:]

	print ("matrix generation finish!")


def generate_signature_matrix_node(trace_file: str):
	# TODO
	""" Optimize this function, it is too slow currently
	"""
	def read_trace(trace_file: str):
		trace_vec = []
		with open(trace_file, 'r') as f:
			counter = 0
			while counter < max_time+100:
				line = f.readline()
				if not line:
					break
				if not (line.startswith("0") or line.startswith("1")):
					continue
				vec = line.strip().split()
				vec = [int(c) for c in vec]
				trace_vec.append(vec)
				counter += 1
		trace_vec = np.array(trace_vec, dtype=np.float16)
		return trace_vec
	# trace_file = "/home/wasmith/Music/pinTrace/cache_log/0x402240-0x4029ed.log"
	data = read_trace(trace_file)
	sensor_n = data.shape[1]

	# multi-scale signature matrix generation
	for w in range(len(win_size)):
		matrix_all = []
		win = win_size[w]
		print ("generating signature with window " + str(win) + "...")
		for t in range(min_time, max_time+100, gap_time):
			#print t
			matrix_t = np.zeros((sensor_n, sensor_n), dtype=np.float32)
			if t >= win_size[2]:
				for i in range(sensor_n):
					for j in range(i, sensor_n):
						#if np.var(data[i, t - win:t]) and np.var(data[j, t - win:t]):
						# matrix_t[i][j] = np.inner(data[i, t - win:t], data[j, t - win:t])/(win) # rescale by win
						tmp1 = data[t - win:t, i].sum()
						tmp2 = data[t - win:t, j].sum()
						matrix_t[i][j] = (tmp1 + tmp2) / (win)
						matrix_t[j][i] = matrix_t[i][j]
			matrix_all.append(matrix_t)
		path_temp = trace_file[:-4] + "_win_" + str(win) + '.npy'
		np.save(path_temp, matrix_all)
		del matrix_all[:]

	print ("matrix generation finish!")


def generate_signature_matrix_node_fast(trace_file: str):
	def read_trace(trace_file: str):
		trace_vec = []
		with open(trace_file, 'r') as f:
			counter = 0
			while counter < max_time+100:
				line = f.readline()
				if not line:
					break
				if not (line.startswith("0") or line.startswith("1")):
					continue
				vec = line.strip().split()
				vec = [int(c) for c in vec]
				trace_vec.append(vec)
				counter += 1
		trace_vec = np.array(trace_vec, dtype=np.float16)
		# what if len(trace_vec) < max_time
		if len(trace_vec) < max_time + 100:
			trace_vec = np.repeat(trace_vec, np.ceil((max_time+100)/len(trace_vec)), axis=0)
			# print(trace_vec.shape)
			# input("debug") 
		return trace_vec
	# trace_file = "/home/wasmith/Music/pinTrace/cache_log/0x402240-0x4029ed.log"
	data = read_trace(trace_file)
	sensor_n = data.shape[1]


	# multi-scale signature matrix generation
	for w in range(len(win_size)):
		matrix_all = []
		win = win_size[w]
		
		path_temp = trace_file[:-4] + "_win_" + str(win) + '.npy'
		if os.path.exists(path_temp):
			continue
		# print ("generating signature with window " + str(win) + "...")

		# init temporary sum
		sum_tmp = np.zeros(sensor_n, dtype=np.float16)
		for i in range(sensor_n):
			sum_tmp[i] = data[min_time:min_time+win, i].sum()
		
		for t in range(min_time, max_time+100, gap_time):
			if t > win:
				# update temporary sum
				for i in range(sensor_n):
					sum_tmp[i] = sum_tmp[i] - data[t-win-1, i] + data[t-1, i]
					# assert sum_tmp[i] == data[t - win:t, i].sum(), "not equal, debug"
			
			matrix_t = np.zeros((sensor_n, sensor_n), dtype=np.float16)
			if t >= win_size[2]:
				for i in range(sensor_n):
					for j in range(i, sensor_n):
						
						tmp1 = sum_tmp[i]
						tmp2 = sum_tmp[j]
						matrix_t[i][j] = (tmp1 + tmp2) / (win)
						matrix_t[j][i] = matrix_t[i][j]
			matrix_all.append(matrix_t)
		path_temp = trace_file[:-4] + "_win_" + str(win) + '.npy'
		np.save(path_temp, matrix_all)
		del matrix_all[:]

	# print ("matrix generation finish!")


def generate_naive_matrix(trace_file: str):
	def read_trace(trace_file: str):
		trace_vec = []
		with open(trace_file, 'r') as f:
			counter = 0
			while counter < max_time+100:
				line = f.readline()
				if not line:
					break
				if not (line.startswith("0") or line.startswith("1")):
					continue
				vec = line.strip().split()
				vec = [int(c) for c in vec]
				trace_vec.append(vec)
				counter += 1
		trace_vec = np.array(trace_vec, dtype=np.int8)
		return trace_vec
	# trace_file = "/home/wasmith/Music/pinTrace/cache_log/0x402240-0x4029ed.log"
	data = read_trace(trace_file)
	sensor_n = data.shape[1]

	matrix_all = []
	for t in range(min_time, max_time+100, gap_time):
		if t + sensor_n >= len(data):
			break
		matrix_t = np.zeros((sensor_n, sensor_n), dtype=np.int8)
		matrix_t = np.copy(data[t:t+sensor_n])
		matrix_all.append(matrix_t)
	path_temp = trace_file[:-4] + "_mat_" + str(64) + '.npy'
	np.save(path_temp, matrix_all)

	data_all = matrix_all
	output_arr = []
	max_len = max(10000, len(data_all)-1)
	for data_id in range(step_max, max_len):
		step_multi_matrix = []  # len=5
		for step_id in range(step_max, 0, -1):
			multi_matrix = []  # len=3
			for i in range(len(win_size)):
				multi_matrix.append(data_all[data_id - step_id])
			step_multi_matrix.append(multi_matrix)

		output_arr.append(step_multi_matrix)
		
	output_arr = np.array(output_arr, dtype=np.int8)
	path_temp = trace_file[:-4] + '_multi5.npy'
	np.save(path_temp, output_arr)

	print ("matrix generation finish!")

def generate_train_test_data_old():
	#data sample generation
	print ("generating train/test data samples...")
	matrix_data_path = save_data_path + "matrix_data/"

	train_data_path = matrix_data_path + "train_data/"
	if not os.path.exists(train_data_path):
		os.makedirs(train_data_path)
	test_data_path = matrix_data_path + "test_data/"
	if not os.path.exists(test_data_path):
		os.makedirs(test_data_path)

	data_all = []
	# for value_col in value_colnames:
	for w in range(len(win_size)):
		#path_temp = matrix_data_path + "matrix_win_" + str(win_size[w]) + str(value_col) + ".npy"
		path_temp = matrix_data_path + "matrix_win_" + str(win_size[w]) + ".npy"
		data_all.append(np.load(path_temp))

	train_test_time = [[train_start, train_end], [test_start, test_end]]
	for i in range(len(train_test_time)):
		for data_id in range(int(train_test_time[i][0]/gap_time), int(train_test_time[i][1]/gap_time)):
			#print data_id
			step_multi_matrix = []
			for step_id in range(step_max, 0, -1):
				multi_matrix = []
				# for k in range(len(value_colnames)):
				for i in range(len(win_size)):
					multi_matrix.append(data_all[i][data_id - step_id])
				step_multi_matrix.append(multi_matrix)

			if data_id >= (train_start/gap_time + win_size[-1]/gap_time + step_max) and data_id < (train_end/gap_time): # remove start points with invalid value
				path_temp = os.path.join(train_data_path, 'train_data_' + str(data_id))
				np.save(path_temp, step_multi_matrix)
			elif data_id >= (test_start/gap_time) and data_id < (test_end/gap_time):
				path_temp = os.path.join(test_data_path, 'test_data_' + str(data_id))
				np.save(path_temp, step_multi_matrix)

			#print np.shape(step_multi_matrix)

			del step_multi_matrix[:]

	print ("train/test data generation finish!")


def generate_train_test_data(file_prefix: str):
	# print ("generating data samples...")
	# prefix = "/home/wasmith/Music/pinTrace/cache_log/0x402240-0x4029ed"
	prefix = file_prefix

	path_temp = prefix + '_multi5.npy'
	if os.path.exists(path_temp):
		return 

	data_vec = []
	for win in win_size:
		tmp_data = np.load(prefix + '_win_' + str(win) + '.npy')
		data_vec.append(tmp_data[win_size[2]:])
	data_all = np.array(data_vec, dtype=np.float16)

	output_arr = []
	max_len = min(10000, len(data_all[0]))
	for data_id in range(step_max, max_len):
		step_multi_matrix = []  # len=5
		for step_id in range(step_max, 0, -1):
			multi_matrix = []  # len=3
			for i in range(len(win_size)):
				multi_matrix.append(data_all[i][data_id - step_id])
			step_multi_matrix.append(multi_matrix)

		output_arr.append(step_multi_matrix)
		# del step_multi_matrix[:]
	
	output_arr = np.array(output_arr, dtype=np.float16)
	path_temp = prefix + '_multi5.npy'
	np.save(path_temp, output_arr)

	# print ("data generation finish!")


if __name__ == '__main__':
	# tmp_data = np.load("/export/d2/zliudc/VirtualEnv/DeepCache/pin_dataset/pin_dataset_glow/resnet18_v1_7.out/0036.libjit_convDKKC8_f_5-0x4090f0-0x409b08_win_10.npy")
	# print(tmp_data[30][0][25])

	# tmp_data2 = np.load("/export/d2/zliudc/VirtualEnv/DeepCache/pin_dataset/pin_dataset_glow/resnet18_v1_7.out/0036.libjit_convDKKC8_f_5-0x4090f0-0x409b08_win_10_backup.npy")
	# print(tmp_data2[30][0][25])
	# cmp = tmp_data - tmp_data2
	# for t in range(len(cmp)):
	# 	for i in range(64):
	# 		for j in range(64):
	# 			if -0.0001 > cmp[t][i][j] > 0.0001:
	# 				input("debug {} {} {} {}".format(t, i, j, cmp[t][i][j]))
	# # print(np.unique(cmp, return_counts=True))
	# exit(0)

	trace_file = "/export/d2/zliudc/VirtualEnv/DeepCache/pin_dataset/pin_dataset_glow/resnet18_v1_7.out/0036.libjit_convDKKC8_f_5-0x4090f0-0x409b08.log"
	# generate_naive_matrix(trace_file)
	# exit(0)
	# generate_signature_matrix_node_fast(trace_file)
	generate_train_test_data(trace_file[:-4])
	exit(0)
	generate_signature_matrix_node(trace_file)
	generate_train_test_data(trace_file[:-4])
	exit(0)

	'''need one more dimension to manage mulitple "features" for each node or link in each time point,
	this multiple features can be simply added as extra channels
	'''

	if ts_type == "node":
		generate_signature_matrix_node()

	generate_train_test_data()
