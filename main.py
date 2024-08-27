import os
import sys
import time
from embedding import Embed
from utils import check_loop_factor_red, check_loop_factor_red_LLC


def experiment_tvm_O3():
    # Part 1: === Embedding Model ===
    # using unsuperivsed learning to extrace embedding vectors for logged traces
    # A embedding mode for all data
    
    trace_data_dir = os.path.abspath("./pin_dataset/pin_dataset_tvm/")
    trainset, trainloader, testset, testloader = Embed.dataset(trace_data_dir, compiler='tvm')

    Embed.set_model_name(compiler='tvm', epoch_num=10, prefix='embedding')

    net, criterion, optimizer = Embed.build_model()
    print("Training...")
    localtime = time.asctime( time.localtime(time.time()) )
    print (localtime)
    Embed.train(net, criterion, optimizer, trainloader, testloader)  # training takes hours

    print("Getting Embeddings...")
    localtime = time.asctime( time.localtime(time.time()) )
    print (localtime)
    embedding_path = os.path.join(trace_data_dir, "embedding.json")
    embedding_path = Embed.generate_embedding_database(net, trainset, embedding_file=embedding_path) 

    print("Matching...")
    localtime = time.asctime( time.localtime(time.time()) )
    print (localtime)
    topk_labels_path = os.path.abspath("./embedding/distinct_labels_tvm.json")
    Embed.database_matching(net, trainset, testset, embedding_file=embedding_path, output_labels_path=topk_labels_path)

    # Part 2: === Recurrent Encoder-Decoder (RED) Network ===
    # For each trace, we train a encoder-decoder. 
    # To save time, we only train a model when we want to segment the trace 
    # (sort by the embedding similarity)
    print("Checking...")
    localtime = time.asctime( time.localtime(time.time()) )
    print (localtime)
    attr_labels = "./attr_labels_tvm.json"
    new_labels = os.path.join(trace_data_dir, "./final_labels_tvm.jsopn")
    check_loop_factor_red(topk_labels_path, attr_labels, trace_data_dir, new_labels, 'tvm')


def experiment_glow():
    # Part 1: === Embedding Model ===
    # using unsuperivsed learning to extrace embedding vectors for logged traces
    # A embedding mode for all data
    
    trace_data_dir = os.path.abspath("./pin_dataset/pin_dataset_glow_100/")
    trainset, trainloader, testset, testloader = Embed.dataset(trace_data_dir, compiler='glow')

    Embed.set_model_name(compiler='glow', epoch_num=40, prefix='embedding')

    net, criterion, optimizer = Embed.build_model()
    print("Training...")
    localtime = time.asctime( time.localtime(time.time()) )
    print (localtime)
    Embed.train(net, criterion, optimizer, trainloader, testloader)  # training takes hours

    print("Get Embeddings...")
    localtime = time.asctime( time.localtime(time.time()) )
    print (localtime)
    embedding_path = os.path.join(trace_data_dir, "embedding.json")
    embedding_path = Embed.generate_embedding_database(net, trainset, embedding_file=embedding_path) 

    print("Matching...")
    localtime = time.asctime( time.localtime(time.time()) )
    print (localtime)
    topk_labels_path = os.path.abspath("./embedding/distinct_labels_glow.json")
    Embed.database_matching(net, trainset, testset, embedding_file=embedding_path, output_labels_path=topk_labels_path)

    # Part 2: === Recurrent Encoder-Decoder (RED) Network ===
    # For each trace, we train a encoder-decoder. 
    # To save time, we only train a model when we want to segment the trace 
    # (sort by the embedding similarity)
    print("Checking...")
    localtime = time.asctime( time.localtime(time.time()) )
    print (localtime)
    attr_labels = "./attr_labels_glow.json"
    new_labels = os.path.join(trace_data_dir, "./final_labels_glow.jsopn")
    check_loop_factor_red(topk_labels_path, attr_labels, trace_data_dir, new_labels, 'glow')


def experiment_tvm_O3_cache(trace_dir="./cache_dataset/cache_dataset_tvm/"):
    # Part 1: === Embedding Model ===
    # using unsuperivsed learning to extrace embedding vectors for logged traces
    # A embedding mode for all data
    
    trace_data_dir = os.path.abspath(trace_dir)
    trainset, trainloader, testset, testloader = Embed.dataset(trace_data_dir, compiler='tvm')
    # input("after getting dataset...continue?")
    Embed.set_model_name(compiler='tvm_cache', epoch_num=70, prefix='embedding')

    net, criterion, optimizer = Embed.build_model()
    print("Training...")
    localtime = time.asctime( time.localtime(time.time()) )
    print (localtime)
    Embed.train(net, criterion, optimizer, trainloader, testloader)  # training takes hours

    print("Getting Embeddings...")
    localtime = time.asctime( time.localtime(time.time()) )
    print (localtime)
    embedding_path = os.path.join(trace_data_dir, "embedding.json")
    trainset, trainloader, testset, testloader = Embed.database(trace_data_dir, compiler='tvm')
    # embedding_path = Embed.generate_embedding_database(net, trainset, embedding_file=embedding_path) 

    print("Matching...")
    localtime = time.asctime( time.localtime(time.time()) )
    print (localtime)
    topk_labels_path = os.path.abspath("./embedding/distinct_labels_tvm_cache.json")
    Embed.database_matching(net, trainset, testset, embedding_file=embedding_path, output_labels_path=topk_labels_path, LLC=True, dis_thre_min=0.55)

    # Part 2: === Recurrent Encoder-Decoder (RED) Network ===
    # For each trace, we train a encoder-decoder. 
    # To save time, we only train a model when we want to segment the trace 
    # (sort by the embedding similarity)
    print("Checking...")
    localtime = time.asctime( time.localtime(time.time()) )
    print (localtime)
    attr_labels = "./attr_labels_tvm.json"
    new_labels = os.path.join(trace_data_dir, "./final_labels_tvm_cache.json")
    # check_loop_factor_red(topk_labels_path, attr_labels, trace_data_dir, new_labels, 'tvm')
    check_loop_factor_red_LLC(topk_labels_path, attr_labels, trace_data_dir, new_labels, 'tvm', topk=30)


def experiment_glow_cache(trace_dir="./cache_dataset/cache_dataset_glow/"):
    # Part 1: === Embedding Model ===
    # using unsuperivsed learning to extrace embedding vectors for logged traces
    # A embedding mode for all data
    
    trace_data_dir = os.path.abspath(trace_dir)
    trainset, trainloader, testset, testloader = Embed.dataset(trace_data_dir, compiler='glow')

    Embed.set_model_name(compiler='glow_cache', epoch_num=70, prefix='embedding')

    net, criterion, optimizer = Embed.build_model()
    print("Training...")
    localtime = time.asctime( time.localtime(time.time()) )
    print (localtime)
    Embed.train(net, criterion, optimizer, trainloader, testloader)  # training takes hours

    print("Get Embeddings...")
    localtime = time.asctime( time.localtime(time.time()) )
    print (localtime)
    embedding_path = os.path.join(trace_data_dir, "embedding.json")
    # trainset, trainloader, testset, testloader = Embed.database(trace_data_dir, compiler='glow')
    # embedding_path = Embed.generate_embedding_database(net, trainset, embedding_file=embedding_path) 

    print("Matching...")
    localtime = time.asctime( time.localtime(time.time()) )
    print (localtime)
    topk_labels_path = os.path.abspath("./embedding/distinct_labels_glow_cache.json")
    Embed.database_matching(net, trainset, testset, embedding_file=embedding_path, output_labels_path=topk_labels_path, LLC=True, dis_thre_min=0.55, dis_thre_max=0.95)  # dis_thre_max is useless

    # Part 2: === Recurrent Encoder-Decoder (RED) Network ===
    # For each trace, we train a encoder-decoder. 
    # To save time, we only train a model when we want to segment the trace 
    # (sort by the embedding similarity)
    print("Checking...")
    localtime = time.asctime( time.localtime(time.time()) )
    print (localtime)
    attr_labels = "./attr_labels_glow.json"
    new_labels = os.path.join(trace_data_dir, "./final_labels_glow.json")
    # check_loop_factor_red(topk_labels_path, attr_labels, trace_data_dir, new_labels, 'glow')
    check_loop_factor_red_LLC(topk_labels_path, attr_labels, trace_data_dir, new_labels, 'glow', topk=30, len_thre=0.1)


def experiment_oram_mitigate(trace_dir="./mitigate/oram_traces/"):
    # Part 1: === Embedding Model ===
    # using unsuperivsed learning to extrace embedding vectors for logged traces
    # A embedding mode for all data
    
    trace_data_dir = os.path.abspath(trace_dir)
    trainset, trainloader, testset, testloader = Embed.dataset(trace_data_dir, compiler='tvm')
    # input("after getting dataset...continue?")
    # Embed.set_model_name(compiler='tvm_oram', epoch_num=40, prefix='embedding')
    Embed.set_model_name(compiler='tvm_oram', epoch_num=20, prefix='embedding')

    net, criterion, optimizer = Embed.build_model()
    print("Training...")
    localtime = time.asctime( time.localtime(time.time()) )
    print (localtime)
    Embed.train(net, criterion, optimizer, trainloader, testloader)  # training takes hours

    print("Getting Embeddings...")
    localtime = time.asctime( time.localtime(time.time()) )
    print (localtime)
    embedding_path = os.path.join(trace_data_dir, "embedding.json")
    embedding_path = Embed.generate_embedding_database(net, trainset, embedding_file=embedding_path) 

    print("Matching...")
    localtime = time.asctime( time.localtime(time.time()) )
    print (localtime)
    topk_labels_path = os.path.abspath("./embedding/distinct_labels_tvm_oram.json")
    Embed.database_matching(net, trainset, testset, embedding_file=embedding_path, output_labels_path=topk_labels_path)

    # # Part 2: === Recurrent Encoder-Decoder (RED) Network ===
    # # For each trace, we train a encoder-decoder. 
    # # To save time, we only train a model when we want to segment the trace 
    # # (sort by the embedding similarity)
    # print("Checking...")
    # localtime = time.asctime( time.localtime(time.time()) )
    # print (localtime)
    # attr_labels = "./attr_labels_tvm.json"
    # new_labels = os.path.join(trace_data_dir, "./final_labels_tvm_cache.json")
    # check_loop_factor_red(topk_labels_path, attr_labels, trace_data_dir, new_labels, 'tvm')
    
def experiment_obfs_mitigate(trace_dir="./mitigate/obfs_traces/"):
    # Part 1: === Embedding Model ===
    # using unsuperivsed learning to extrace embedding vectors for logged traces
    # A embedding mode for all data
    
    trace_data_dir = os.path.abspath(trace_dir)
    trainset, trainloader, testset, testloader = Embed.dataset(trace_data_dir, compiler='glow')
    # input("after getting dataset...continue?")
    # Embed.set_model_name(compiler='glow_obfs', epoch_num=40, prefix='embedding')
    Embed.set_model_name(compiler='glow_obfs', epoch_num=20, prefix='embedding')

    net, criterion, optimizer = Embed.build_model()
    print("Training...")
    localtime = time.asctime( time.localtime(time.time()) )
    print (localtime)
    Embed.train(net, criterion, optimizer, trainloader, testloader)  # training takes hours

    print("Getting Embeddings...")
    localtime = time.asctime( time.localtime(time.time()) )
    print (localtime)
    embedding_path = os.path.join(trace_data_dir, "embedding.json")
    embedding_path = Embed.generate_embedding_database(net, trainset, embedding_file=embedding_path) 

    print("Matching...")
    localtime = time.asctime( time.localtime(time.time()) )
    print (localtime)
    topk_labels_path = os.path.abspath("./embedding/distinct_labels_glow_obfs.json")
    Embed.database_matching(net, trainset, testset, embedding_file=embedding_path, output_labels_path=topk_labels_path)

    # # Part 2: === Recurrent Encoder-Decoder (RED) Network ===
    # # For each trace, we train a encoder-decoder. 
    # # To save time, we only train a model when we want to segment the trace 
    # # (sort by the embedding similarity)
    # print("Checking...")
    # localtime = time.asctime( time.localtime(time.time()) )
    # print (localtime)
    # attr_labels = "./attr_labels_glow.json"
    # new_labels = os.path.join(trace_data_dir, "./final_labels_glow_obfs.json")
    # check_loop_factor_red(topk_labels_path, attr_labels, trace_data_dir, new_labels, 'glow')


# ======= LLC PP (Prime+Probe) Experiments =======
def experiment_tvm_O3_LLC(trace_dir="./cache_dataset/cache_dataset_llc_tvm/"):
    # Part 1: === Embedding Model ===
    # using unsuperivsed learning to extrace embedding vectors for logged traces
    # A embedding mode for all data
    
    trace_data_dir = os.path.abspath(trace_dir)
    trainset, trainloader, testset, testloader = Embed.dataset(trace_data_dir, compiler='tvm', paral_ver=True)
    # input("after getting dataset...continue?")
    Embed.set_model_name(compiler='tvm_cache_llc', epoch_num=50, prefix='embedding')

    net, criterion, optimizer = Embed.build_model()
    print("Training...")
    localtime = time.asctime( time.localtime(time.time()) )
    print (localtime)
    Embed.train(net, criterion, optimizer, trainloader, testloader)  # training takes hours

    print("\nGetting Embeddings...")
    localtime = time.asctime( time.localtime(time.time()) )
    print (localtime)
    embedding_path = os.path.join(trace_data_dir, "embedding.json")
    trainset, trainloader, testset, testloader = Embed.database(trace_data_dir, compiler='tvm')
    # embedding_path = Embed.generate_embedding_database(net, trainset, embedding_file=embedding_path) 

    print("Matching...")
    localtime = time.asctime( time.localtime(time.time()) )
    print (localtime)
    topk_labels_path = os.path.abspath("./embedding/distinct_labels_tvm_cache_llc.json")
    Embed.database_matching(net, trainset, testset, embedding_file=embedding_path, output_labels_path=topk_labels_path, LLC=True)

    # Part 2: === Recurrent Encoder-Decoder (RED) Network ===
    # For each trace, we train a encoder-decoder. 
    # To save time, we only train a model when we want to segment the trace 
    # (sort by the embedding similarity)
    print("Checking...")
    localtime = time.asctime( time.localtime(time.time()) )
    print (localtime)
    attr_labels = "/export/d2/zliudc/VirtualEnv/ONNX_Zoo_Loop/TVM-0.12/labels_new.json"
    new_labels = os.path.join(trace_data_dir, "./final_labels_tvm_cache_llc.json")
    check_loop_factor_red_LLC(topk_labels_path, attr_labels, trace_data_dir, new_labels, 'tvm', topk=20)


def experiment_glow_LLC(trace_dir="./cache_dataset/LLC_dataset_glow/"):
    # Part 1: === Embedding Model ===
    # using unsuperivsed learning to extrace embedding vectors for logged traces
    # A embedding mode for all data
    
    trace_data_dir = os.path.abspath(trace_dir)
    trainset, trainloader, testset, testloader = Embed.dataset(trace_data_dir, compiler='glow', paral_ver=True)
    # input("after getting dataset...continue?")
    Embed.set_model_name(compiler='glow_cache_llc', epoch_num=50, prefix='embedding')

    net, criterion, optimizer = Embed.build_model()
    print("Training...")
    localtime = time.asctime( time.localtime(time.time()) )
    print (localtime)
    Embed.train(net, criterion, optimizer, trainloader, testloader)  # training takes hours

    print("\nGetting Embeddings...")
    localtime = time.asctime( time.localtime(time.time()) )
    print (localtime)
    embedding_path = os.path.join(trace_data_dir, "embedding.json")
    trainset, trainloader, testset, testloader = Embed.database(trace_data_dir, compiler='glow')
    # embedding_path = Embed.generate_embedding_database(net, trainset, embedding_file=embedding_path) 

    print("Matching...")
    localtime = time.asctime( time.localtime(time.time()) )
    print (localtime)
    topk_labels_path = os.path.abspath("./embedding/distinct_labels_glow_cache_llc.json")
    Embed.database_matching(net, trainset, testset, embedding_file=embedding_path, output_labels_path=topk_labels_path, LLC=True)

    # Part 2: === Recurrent Encoder-Decoder (RED) Network ===
    # For each trace, we train a encoder-decoder. 
    # To save time, we only train a model when we want to segment the trace 
    # (sort by the embedding similarity)
    print("Checking...")
    localtime = time.asctime( time.localtime(time.time()) )
    print (localtime)
    attr_labels = "./labels_glow_HP.json"
    new_labels = os.path.join(trace_data_dir, "./final_labels_glow_cache_llc.json")
    check_loop_factor_red_LLC(topk_labels_path, attr_labels, trace_data_dir, new_labels, 'glow', topk=20)


if __name__ == '__main__':
    # # check results of tvm O3 cache
    # check_loop_factor_red("/export/ssd1/zliudc/VirtualEnv/DeepCache/embedding/distinct_labels_tvm_cache.json", 
    #                       "./attr_labels_tvm.json", 
    #                       "/export/ssd1/zliudc/VirtualEnv/DeepCache/cache_dataset/cache_dataset_tvm/", "./tmp_tvm_O3_cache.json", 'tvm')
    # # check results of tvm O3 cache
    # check_loop_factor_red("/export/ssd1/zliudc/VirtualEnv/DeepCache/embedding/distinct_labels_glow_cache.json", 
    #                       "./attr_labels_glow.json", 
    #                       "/export/ssd1/zliudc/VirtualEnv/DeepCache/cache_dataset/cache_dataset_glow/", "./tmp_tvm_O3_glow.json", 'glow')
    # exit(0)

    # experiment_tvm_O3_cache("./cache_dataset/cache_dataset_tvm/")
    # experiment_glow_cache("./cache_dataset/cache_dataset_glow/")
    # # experiment_oram_mitigate()

    # experiment_tvm_O3_LLC("/export/d2/zliudc/VirtualEnv/DeepCache/cache_dataset/LLC_dataset_tvm/")
    # experiment_glow_LLC("./cache_dataset/LLC_dataset_glow/")
    # exit(0)
    if sys.argv[1] == 'tvm':
        print("Start TVM experiment")
        # experiment_tvm_O3()
        experiment_tvm_O3_LLC("./cache_dataset/LLC_dataset_tvm/")
    if sys.argv[1] == 'glow':
        print("Start Glow experiment")
        # experiment_glow()
        experiment_glow_LLC("./cache_dataset/LLC_dataset_glow/")

    if sys.argv[1] == 'oram':
        print("Start ORAM experiment")
        experiment_oram_mitigate("./mitigate/oram_traces/")

    if sys.argv[1] == 'obfs':
        print("Start obfuscation experiment")
        experiment_obfs_mitigate("./mitigate/obfs_traces/")



