# coding=utf-8
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from model.basemodel.networks import define_G
from utils.data_trans import get_dataloader
from utils.evaluate import log_eva
from utils.losses import cross_entropy_Loss
from utils.train_model import train_one_epoch
from utils.val_model import val_one_epoch


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
    parser.add_argument("--log_message", default="BiT_LEVIR_CD")
    parser.add_argument("--dataset_name", default="LEVIR_CD")
    parser.add_argument('--net_G', default="Bit", help='initial learning rate for adam')
    parser.add_argument("--data_path", default=r"F:\Pycharm_program\lunwen\datasets\CD_use", type=str, help="dataset path")
    parser.add_argument("--epochs", default=200, type=int, help="model all epoch")
    parser.add_argument("--classes", default=2, type=int, help="model predict kind")
    parser.add_argument(
        "-b", "--batch-size", default=32, type=int, help="images per gpu, the total batch size is $NGPU x batch_size")
    parser.add_argument("--num_workers", default=6, type=int, help="images per gpu, the total batch size is $NGPU x batch_size")
    parser.add_argument("--resume", default=r"E:\实验\weights\BiT_LEVIR_CD\model_196.pth", type=str, help="path of checkpoint precede over start_epoch")
    parser.add_argument("--start_epoch", default=0, type=int, help="start epoch")
    parser.add_argument("--test_only", default=False, type=bool, help="just use the model to test")
    parser.add_argument("--init_weight", default="", type=str,
                        help="the init model path of first epoch")
    parser.add_argument("--output_dir", default=r"E:\实验\weights")
    parser.add_argument("--one_hot_flag", default=True, type=bool, help="some loss need one-hot trans")
    parser.add_argument("--amp", default=True, type=bool, help="Use torch.cuda.amp for mixed precision training")
    parser.add_argument("--subdata", default=False, type=bool, help="sub datasets to test code")
    parser.add_argument('--lr_policy', default='linear', type=str,
                        help='linear | step')
    parser.add_argument('--lr_decay_iters', default=200, type=int)
    parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate for adam')
    parser.add_argument('--loss', default='bce', type=str)
    parser.add_argument('--optimizer', default='SGD', type=str)
    parser.add_argument("--tensorboard_log",default=r"E:\实验\model_runs/",type=str)

    return parser

def main(args):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device training.".format(device))



    train_data_loader,val_data_loader = get_dataloader(args)

    # define model

    model=define_G(args,init_type='normal', init_gain=0.02, gpu_ids=[0])

    if args.loss == 'ce':
        criterion = cross_entropy_Loss()
    elif args.loss == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise NotImplemented(args.loss)

    # find optimizer by args
    if args.optimizer=="SGD":
        optimizer=optim.SGD(model.parameters(), lr=args.lr,
                                     momentum=0.9,
                                     weight_decay=5e-4)
    elif args.optimizer=="Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    else:
        raise NotImplemented(args.loss)

    # use the scaler if we want amp
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # find lr_scheduler by args
    if args.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - epoch / float(args.epochs+ 1)
            return lr_l
        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    elif args.lr_policy == 'step':
        step_size = args.epochs//4
        # args.lr_decay_iters
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    else:
        lr_scheduler = None

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        """
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        """
        if lr_scheduler:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
    else:
        pass
        #model.init_weights("init_weight/resnet18-5c106cde.pth")



    if args.tensorboard_log:
        writer_train = SummaryWriter(args.tensorboard_log+args.log_message+"/train")
        writer_val = SummaryWriter(args.tensorboard_log+args.log_message+'/val')
    else:
        writer_train=None
        writer_val=None


    model.to(device)

    # training
    for epoch in range(args.start_epoch, args.epochs):
        #print("lr:" + str(optimizer.state_dict()['param_groups'][0]['lr']))
        epoch_eva_dict,epoch_loss_train=train_one_epoch(model=model,criterion=criterion,device=device,dataloader_train=train_data_loader,optimizer=optimizer,
                                             scaler=scaler,args=args)
        log_eva(epoch + 1, epoch_eva_dict, epoch_loss_train, writer_train)
        if lr_scheduler is not None:
            lr_scheduler.step()

        epoch_eva_dict,epoch_loss_val=val_one_epoch(model=model, criterion=criterion, device=device,dataloader_val=val_data_loader,
                         args=args)
        log_eva(epoch + 1, epoch_eva_dict, epoch_loss_val,writer_val)

        if os.path.join(args.output_dir,args.log_message):
            if not os.path.exists(os.path.join(args.output_dir,args.log_message)):
                os.makedirs(os.path.join(args.output_dir,args.log_message))
            if lr_scheduler:
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                }
            else:
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,

                }
            if args.amp:
                checkpoint["scaler"] = scaler.state_dict()
            torch.save(checkpoint, os.path.join(os.path.join(args.output_dir,args.log_message), f"model_{epoch}.pth"))



if __name__ == "__main__":
    torch.cuda.empty_cache()
    seed_torch(2023)
    args = get_args_parser().parse_args(args=[])
    main(args)