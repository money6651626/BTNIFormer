# coding=utf-8
import os
import random

import numpy as np
import torch
import torch.nn as nn

from model.basemodel.networks import define_G
from utils.data_trans import get_dataloader
from utils.evaluate import log_eva
from utils.losses import BCL, DiceLoss, cross_entropy
from utils.val_model import val_one_epoch,val_one_epoch_test,val_one_epoch_Tiny_CD
from utils.val_model import val_one_epoch_DSAMNet,val_one_epoch_STA


def seed_torch(seed=2023):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


# torch.use_deterministic_algorithms(True)  # 有检查操作，看下文区别




def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description="PyTorch Model Training", add_help=add_help)
    parser.add_argument("--dataset_name", default="WHU_CD")
    parser.add_argument('--net_G', default="BmmtNet_NAT_V7", help='initial learning rate for adam')
    parser.add_argument("--data_path", default=r"F:\Pycharm_program\lunwen\datasets\CD_use", type=str, help="dataset path")
    parser.add_argument("--classes", default=2, type=int, help="model predict kind")
    parser.add_argument(
        "-b", "--batch-size", default=1, type=int, help="images per gpu, the total batch size is $NGPU x batch_size")
    parser.add_argument("--test_only", default=True, type=bool, help="just use the model to test")
    parser.add_argument("--num_workers", default=1, type=int, help="images per gpu, the total batch size is $NGPU x batch_size")
    parser.add_argument("--init_weight", default=r"pretrain_weight/WHU_CD.pth", type=str,
                        help="the init model path of first epoch")
    parser.add_argument("--subdata", default=False, type=bool, help="sub datasets to test code")
    parser.add_argument("--one_hot_flag", default=True, type=bool, help="some loss need one-hot trans")
    parser.add_argument('--embed_dim', default=64, type=int)
    parser.add_argument('--wDice', default=0.1, type=float)
    parser.add_argument('--get_flops', default=False, type=bool)
    parser.add_argument('--img_size', default=256, type=int)
    return parser

def main(args):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device evaling.".format(device))

    test_data_loader = get_dataloader(args)

    # define model
    model=define_G(args,init_type='normal', init_gain=0.02, gpu_ids=[0])

    if args.init_weight :
        checkpoint = torch.load(args.init_weight, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    else:
        pass

    criterion = nn.BCEWithLogitsLoss()
    model.to(device)

    epoch_eva_dict,epoch_loss_val=val_one_epoch(model=model, criterion=criterion, device=device,dataloader_val=test_data_loader,
                     args=args)
    print(epoch_eva_dict)






if __name__ == "__main__":
    torch.cuda.empty_cache()
    seed_torch(2023)
    args = get_args_parser().parse_args(args=[])
    main(args)