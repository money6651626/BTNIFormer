import torch
import numpy as np
from model.basemodel.networks import define_G
def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description="PyTorch Model Training", add_help=add_help)
    parser.add_argument("--dataset_name", default="WHU_CD")
    parser.add_argument('--net_G', default="ChangeFormerV6", help='initial learning rate for adam')
    parser.add_argument("--data_path", default=r"F:\Pycharm_program\lunwen\datasets\CD_use", type=str, help="dataset path")
    parser.add_argument("--classes", default=2, type=int, help="model predict kind")
    parser.add_argument(
        "-b", "--batch-size", default=1, type=int, help="images per gpu, the total batch size is $NGPU x batch_size")
    parser.add_argument("--test_only", default=True, type=bool, help="just use the model to test")
    parser.add_argument("--num_workers", default=4, type=int, help="images per gpu, the total batch size is $NGPU x batch_size")
    parser.add_argument("--init_weight", default=r"F:\Pycharm_program\lunwen\runs\weights_pretrain\BmmtNet_NAT_V7\WHU_CD\best_ckpt.pth", type=str,
                        help="the init model path of first epoch")
    parser.add_argument("--subdata", default=False, type=bool, help="sub datasets to test code")
    parser.add_argument('--loss', default='bce', type=str)
    parser.add_argument("--one_hot_flag", default=True, type=bool, help="some loss need one-hot trans")
    parser.add_argument('--embed_dim', default=256, type=int)
    parser.add_argument('--wDice', default=0.1, type=float)
    parser.add_argument('--get_flops', default=True, type=bool)
    parser.add_argument('--img_size', default=256, type=int)
    return parser



#--------------------------------------------------------------------#
#作用：计算模型需要占用的显存，方便知道显卡够不够用
#使用方法：将模型初始化之后，传入Calculate_gpu_memory()即可
#--------------------------------------------------------------------#

import torch
import numpy as np
import torchvision
import torch.nn as nn

def Calculate_gpu_memory(Model,input):
    import torch

    # 模型初始化




def calculate_infer_time(model):
    import numpy as np
    from torchvision.models import resnet50
    import torch
    from torch.backends import cudnn
    import tqdm
    cudnn.benchmark = True

    device = 'cuda:0'
    repetitions = 50

    dummy_input = torch.rand(16, 3, 256, 256).to(device)

    # 预热, GPU 平时可能为了节能而处于休眠状态, 因此需要预热
    print('warm up ...\n')
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input,dummy_input)

    # synchronize 等待所有 GPU 任务处理完才返回 CPU 主线程
    torch.cuda.synchronize()

    # 设置用于测量时间的 cuda Event, 这是PyTorch 官方推荐的接口,理论上应该最靠谱
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # 初始化一个时间容器
    timings = np.zeros((repetitions, 1))

    print('testing ...\n')
    with torch.no_grad():
        for rep in tqdm.tqdm(range(repetitions)):
            starter.record()
            _ = model(dummy_input,dummy_input)
            ender.record()
            torch.cuda.synchronize()  # 等待GPU任务完成
            curr_time = starter.elapsed_time(ender)  # 从 starter 到 ender 之间用时,单位为毫秒
            timings[rep] = curr_time

    avg = timings.sum() / repetitions
    print('\navg={}\n'.format(avg))


def calculate_training_time(model,cretion):
    import numpy as np
    import torch
    from torch.backends import cudnn
    import tqdm
    cudnn.benchmark = True

    device = 'cuda:0'
    repetitions = 10

    dummy_input = torch.rand( 16, 3, 256, 256).to(device)
    label_input = torch.rand(16, 2, 256, 256).to(device)
    # 预热, GPU 平时可能为了节能而处于休眠状态, 因此需要预热
    print('warm up ...\n')

    for _ in range(10):
        _ = model(dummy_input,dummy_input)

    # synchronize 等待所有 GPU 任务处理完才返回 CPU 主线程
    torch.cuda.synchronize()

    # 设置用于测量时间的 cuda Event, 这是PyTorch 官方推荐的接口,理论上应该最靠谱
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # 初始化一个时间容器
    timings = np.zeros((repetitions, 1))

    print('testing ...\n')
    for rep in tqdm.tqdm(range(repetitions)):
        starter.record()
        out= model(dummy_input,dummy_input)
        loss = cretion(out,label_input)
        loss.backward()
        ender.record()
        torch.cuda.synchronize()  # 等待GPU任务完成
        curr_time = starter.elapsed_time(ender)  # 从 starter 到 ender 之间用时,单位为毫秒
        timings[rep] = curr_time

    avg = timings.sum() / repetitions
    print('\navg={}\n'.format(avg))

def calculate_mem(model,criterion):
    device = 'cuda:0'
    print(torch.cuda.memory_allocated()/ (2 ** 20))

    # 输入定义
    dummy_input = torch.randn(16, 3, 256, 256).to(device)  # shape = (1024,1024) # + 4194304
    print(torch.cuda.memory_allocated()/(2**20))

    # 前向传播
    loss = criterion(model(dummy_input,dummy_input),torch.randn(16, 2, 256, 256).to(device))  # shape = (1) # memory + 4194304 + 512
    print(torch.cuda.memory_allocated()/(2**20))

    # 后向传播
    loss.backward()  # memory - 4194304 + 4194304 + 4096
    print(torch.cuda.memory_allocated()/(2**20))

def calculate_peak_mem(model,criterion):
    import torch

    device = 'cuda:0'

    # 初始化显存追踪器
    torch.cuda.reset_max_memory_allocated()

    # 输入定义
    dummy_input = torch.randn(16, 3, 256, 256).to(device)
    torch.cuda.synchronize()  # 确保在追踪之前所有操作都已完成
    print("Current GPU memory usage:", torch.cuda.memory_allocated() / (2 ** 20))  # 当前显存占用
    print("Peak GPU memory usage:", torch.cuda.max_memory_allocated() / (2 ** 20))  # 峰值显存占用

    # 前向传播
    loss = criterion(model(dummy_input, dummy_input), torch.randn(16, 2, 256, 256).to(device))
    torch.cuda.synchronize()
    print("Current GPU memory usage:", torch.cuda.memory_allocated() / (2 ** 20))
    print("Peak GPU memory usage:", torch.cuda.max_memory_allocated() / (2 ** 20))

    # 后向传播
    loss.backward()
    torch.cuda.synchronize()
    print("Current GPU memory usage:", torch.cuda.memory_allocated() / (2 ** 20))
    print("Peak GPU memory usage:", torch.cuda.max_memory_allocated() / (2 ** 20))


def main(args):
    torch.cuda.empty_cache()
    print(torch.cuda.memory_allocated()/ (2 ** 20))

    model=define_G(args,init_type='normal', init_gain=0.02, gpu_ids=[0])


    criterion = nn.BCEWithLogitsLoss()
    #calculate_training_time(model,criterion)
    #calculate_infer_time(model)
    #calculate_mem(model,criterion)
    calculate_peak_mem(model, criterion)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    args = get_args_parser().parse_args(args=[])
    main(args)
