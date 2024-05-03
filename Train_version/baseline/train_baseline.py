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
    parser.add_argument("--dataset_name", default="LEVIR_CD")
    parser.add_argument('--net_G', default="DTCDSCN", help='initial learning rate for adam')
    parser.add_argument("--data_path", default="F:\Pycharm_program\lunwen\datasets/CD_use", type=str, help="dataset path")
    parser.add_argument("--epochs", default=200, type=int, help="model all epoch")
    parser.add_argument("--classes", default=2, type=int, help="model predict kind")
    parser.add_argument(
        "-b", "--batch-size", default=16, type=int, help="images per gpu, the total batch size is $NGPU x batch_size")
    parser.add_argument("--num_workers", default=4, type=int, help="cpu kernel use")
    parser.add_argument("--resume", default=False, type=bool,
                        help="resume the training by the latest_checkpoint")
    parser.add_argument("--start_epoch", default=0, type=int, help="start epoch")
    parser.add_argument("--test_only", default=False, type=bool, help="just use the model to test")
    parser.add_argument("--init_weight", default="", type=str,
                        help="the init model path of first epoch")
    parser.add_argument("--output_dir", default=r"F:\Pycharm_program\lunwen\runs/weights")
    parser.add_argument("--one_hot_flag", default=True, type=bool, help="some loss need one-hot trans")
    parser.add_argument("--amp", default=True, type=bool, help="Use torch.cuda.amp for mixed precision training")
    parser.add_argument("--subdata", default=False, type=bool, help="sub datasets to test code")
    parser.add_argument('--lr_policy', default='linear', type=str,
                        help='linear | step')
    parser.add_argument('--lr_decay_iters', default=200, type=int)
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
    parser.add_argument('--loss', default='ce', type=str)
    parser.add_argument('--optimizer', default='AdamW', type=str)
    parser.add_argument("--tensorboard_log",default=r"F:\Pycharm_program\lunwen\runs/model_runs/",type=str)
    parser.add_argument('--embed_dim', default=64, type=int)
    parser.add_argument('--img_size', default=256, type=int)
    return parser

def main(args):
    best_eval={
        "epoch": 0,
        "Iou": 0.0,
        "F1": 0.0,
    }
    args.log_message=os.path.join(args.net_G,args.dataset_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device training.".format(device))
    train_data_loader,val_data_loader = get_dataloader(args)

    #train_prefetcher = DataPrefetcher(train_data_loader,device)
    #val_prefetcher = DataPrefetcher(val_data_loader, device)

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
                                     momentum=0.99,
                                     weight_decay=5e-4)
    elif args.optimizer=="AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                                    betas=(0.9, 0.999), weight_decay=0.01)
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


    if args.resume: #肯定不会是0 恢复的是latest.pth
        checkpoint = torch.load(os.path.join(args.output_dir, args.log_message, "latest_ckpt.pth"), map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        best_eval["Iou"]=checkpoint["best_Iou"]
        best_eval["F1"] =checkpoint["best_F1"]
        best_eval["epoch"] =checkpoint["best_epoch"]
        if lr_scheduler:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if args.start_epoch==args.epochs:
            print("train_finished!")
            exit()
        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        print("load resume point epoch:{} ".format(args.start_epoch))
    else:
        if args.init_weight:
            model.load_state_dict(torch.load(args.init_weight,map_location=device)["model_G_state_dict"], strict=False)


    if args.tensorboard_log:
        writer_train = SummaryWriter(args.tensorboard_log+args.log_message+"/train")
        writer_val = SummaryWriter(args.tensorboard_log+args.log_message+'/val')
    else:
        writer_train=None
        writer_val=None


    model.to(device)

    # training
    for epoch in range(args.start_epoch, args.epochs):
        # print("lr:" + str(optimizer.state_dict()['param_groups'][0]['lr']))
        epoch_eva_dict, epoch_loss = train_one_epoch(model=model, criterion=criterion, device=device,
                                                     dataloader_train=train_data_loader, optimizer=optimizer,
                                                     scaler=scaler, args=args)
        log_eva(epoch + 1, epoch_eva_dict, epoch_loss, writer_train)
        if lr_scheduler is not None:
            lr_scheduler.step()

        epoch_eva_dict, epoch_loss_val = val_one_epoch(model=model, criterion=criterion, device=device,
                                                       dataloader_val=val_data_loader,
                                                       args=args)
        log_eva(epoch + 1, epoch_eva_dict, epoch_loss_val, writer_val)

        if os.path.join(args.output_dir, args.log_message):
            if not os.path.exists(os.path.join(args.output_dir, args.log_message)):
                os.makedirs(os.path.join(args.output_dir, args.log_message))
            latest_checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_Iou": best_eval["Iou_1"],
                "best_F1": best_eval["F1_1"],
                "best_epoch": best_eval["epoch"],
            }
            if lr_scheduler:
                latest_checkpoint["lr_scheduler"] = lr_scheduler.state_dict()
            if args.amp:
                latest_checkpoint["scaler"] = scaler.state_dict()
            if epoch_eva_dict["F1_1"] > best_eval["F1"] and epoch_eva_dict["Iou_1"] > best_eval["Iou"]:
                best_checkpoint = {
                    "model": model.state_dict(),
                    "best_Iou": epoch_eva_dict["Iou_1"],#Iou_1
                    "best_F1": epoch_eva_dict["F1_1"],#F1_1
                    "best_epoch": epoch + 1,
                }
                # 更新记录
                best_eval["F1"] = epoch_eva_dict["F1_1"]#F1_1
                best_eval["Iou"] = epoch_eva_dict["Iou_1"]#Iou_1
                best_eval["epoch"] = epoch + 1
                latest_checkpoint["best_Iou"] = epoch_eva_dict["Iou"]#Iou_1
                latest_checkpoint["best_F1"] = epoch_eva_dict["F1"]#F1_1
                latest_checkpoint["best_epoch"] = epoch
                torch.save(best_checkpoint, os.path.join(args.output_dir, args.log_message, "best_ckpt.pth"))
                print("best_update! F1:{:.4f},Iou:{:.4f}".format(epoch_eva_dict["mF1"], epoch_eva_dict["mIou"]))
            torch.save(latest_checkpoint, os.path.join(args.output_dir, args.log_message, "latest_ckpt.pth"))
        else:
            raise NotImplementedError('need check_point_path [%s]', args.output_dir)

    print(best_eval)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    seed_torch(2023)
    args = get_args_parser().parse_args(args=[])
    main(args)