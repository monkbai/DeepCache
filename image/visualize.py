from PIL import Image

image_length = 128

block = 3

width = 64 * block
height = image_length * block

# trace_file = "/export/ssd1/zliudc/VirtualEnv/DeepCache/pin_dataset/pin_dataset_tvm/resnet18-v1-7/0049.tvmgen_default_fused_nn_contrib_conv2d_NCHWc_add_compute_-0x406f80-0x409406-.log"
# tvm
# trace_file = "/export/ssd1/zliudc/VirtualEnv/DeepCache/pin_dataset/pin_dataset_tvm/resnet18-v1-7/0072.tvmgen_default_fused_nn_contrib_conv2d_NCHWc_add_add_nn_relu_3_compute_-0x416ed0-0x417a80-.log"
# glow
# trace_file = "/export/ssd1/zliudc/VirtualEnv/DeepCache/pin_dataset/pin_dataset_glow/resnet18_v1_7.out/0034.libjit_convDKKC8_f_4-0x407c90-0x4085bb.log"


trace_file = "/export/ssd1/zliudc/VirtualEnv/DeepCache/cache_dataset/cache_dataset_glow/resnet18_v1_7.out/0036.libjit_convDKKC8_f_5-0x4090f0-0x409b08-.log"

trace_file = "/export/d2/zliudc/pin_dataset/pin_dataset_tvm/vgg16-7/0064.tvmgen_default_fused_nn_contrib_conv2d_NCHWc_add_nn_relu_4_compute_-0x413bc0-0x416c4a-.log"

# L1 prime + probe
trace_file = "/export/ssd1/zliudc/VirtualEnv/0079.tvmgen_default_fused_nn_contrib_conv2d_NCHWc_add_add_nn_relu_5_compute_-0x41afd0-0x41c60b-.log"

# LLC prime + scope
trace_file = "/export/ssd1/zliudc/VirtualEnv/output.log"
skip = 3800

with open(trace_file, 'r') as f:
    counter = 0
    while counter < skip:
        line = f.readline()
        if not line:
            break
        counter += 1
    
    cur_pic = []
    while len(cur_pic) < height:
        for i in range(image_length):
            line = f.readline()
            if not line:
                break
            if not (line.startswith("0") or line.startswith("1")):
                break
            vec = line.strip().split()
            vec = [int(c) for c in vec]
            cur_pic.append(vec)

img  = Image.new( mode = "L", size = (width, height) )
print(img.size)
pixels = img.load()
for i in range(image_length):
    for j in range(64):
        target = 120 if cur_pic[i][j] else 230
        for b1 in range(block):
            for b2 in range(block):
                pixels[j*block+b1,i*block+b2] = target

img = img.save("tmp.jpg")
