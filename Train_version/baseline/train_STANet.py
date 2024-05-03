# coding=utf-8
import os
import random

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from model.basemodel.networks import define_G
from utils.data_trans import get_dataloader
from utils.evaluate import log_eva
from utils.losses import BCL, DiceLoss
from utils.train_model import train_one_epoch_STA,train_one_epoch
from utils.val_model import val_one_epoch_STA,val_one_epoch


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
    parser.add_argument("--log_message", default="STANet_LEVIR")
    parser.add_argument("--dataset_name", default="LEVIR_CD")
    parser.add_argument('--net_G', default="STANet", help='initial learning rate for adam')
    parser.add_argument("--data_path", default=r"F:\Pycharm_program\lunwen\datasets\CD_use", type=str, help="dataset path")
    parser.add_argument("--epochs", default=200, type=int, help="model all epoch")
    parser.add_argument("--classes", default=2, type=int, help="model predict kind")
    parser.add_argument(
        "-b", "--batch-size", default=1, type=int, help="images per gpu, the total batch size is $NGPU x batch_size")
    parser.add_argument("--num_workers", default=4, type=int, help="images per gpu, the total batch size is $NGPU x batch_size")
    parser.add_argument("--resume", default=False, type=bool,
                        help="resume the training by the latest_checkpoint")
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
    parser.add_argument('--wDice', default=0.1, type=float)
    parser.add_argument('--img_size', default=256, type=float)
    return parser

def main(args):

    best_eval={
        "epoch": 0,
        "mIou": 0.0,
        "mF1": 0.0,
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device training.".format(device))



    train_data_loader,val_data_loader = get_dataloader(args)

    # define model

    model=define_G(args,init_type='normal', init_gain=0.02, gpu_ids=[0])

    criterion = BCL().to(device, dtype=torch.float)


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

    if args.resume: #肯定不会是0 恢复的是latest.pth
        checkpoint = torch.load(os.path.join(args.output_dir, args.log_message, "latest_ckpt.pth"), map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        best_eval["mIou"]=checkpoint["best_Iou"]
        best_eval["mF1"] =checkpoint["best_F1"]
        best_eval["epoch"] =checkpoint["best_epoch"]
        if lr_scheduler:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        print("load resume point epoch:{} ".format(args.start_epoch))
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
        epoch_eva_dict,epoch_loss= train_one_epoch(model=model,criterion=criterion,device=device,dataloader_train=train_data_loader,optimizer=optimizer,
                                             scaler=scaler,args=args)
        log_eva(epoch + 1, epoch_eva_dict, epoch_loss, writer_train)
        if lr_scheduler is not None:
            lr_scheduler.step()

        epoch_eva_dict,epoch_loss_val= val_one_epoch(model=model, criterion=criterion, device=device,dataloader_val=val_data_loader,
                         args=args)
        log_eva(epoch + 1, epoch_eva_dict, epoch_loss_val,writer_val)

        if os.path.join(args.output_dir, args.log_message):
            if not os.path.exists(os.path.join(args.output_dir, args.log_message)):
                os.makedirs(os.path.join(args.output_dir, args.log_message))
            latest_checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_Iou": best_eval["mIou"],
                "best_F1": best_eval["mF1"],
                "best_epoch": best_eval["epoch"],
            }
            if lr_scheduler:
                latest_checkpoint["lr_scheduler"] = lr_scheduler.state_dict()
            if args.amp:
                latest_checkpoint["scaler"] = scaler.state_dict()
            if epoch_eva_dict["mF1"] > best_eval["mF1"] and epoch_eva_dict["mIou"] > best_eval["mIou"]:

                best_checkpoint = {
                    "model": model.state_dict(),
                    "best_Iou": epoch_eva_dict["mIou"],
                    "best_F1": epoch_eva_dict["mF1"],
                    "best_epoch": epoch+1,
                }
                #更新记录
                best_eval["mF1"]=epoch_eva_dict["mF1"]
                best_eval["mIou"]=epoch_eva_dict["mIou"]
                best_eval["epoch"]=epoch+1
                latest_checkpoint["best_Iou"]=epoch_eva_dict["mIou"]
                latest_checkpoint["best_F1"] = epoch_eva_dict["mF1"]
                latest_checkpoint["best_epoch"] = epoch
                torch.save(best_checkpoint, os.path.join(args.output_dir, args.log_message, "best_ckpt.pth"))
                print("best_update! F1:{:.4f},Iou:{:.4f}".format(epoch_eva_dict["mF1"],epoch_eva_dict["mIou"]))
            torch.save(latest_checkpoint, os.path.join(args.output_dir, args.log_message, "latest_ckpt.pth"))
        else:
            raise NotImplementedError('need check_point_path [%s]', args.output_dir)

    print(best_eval)



if __name__ == "__main__":
    torch.cuda.empty_cache()
    seed_torch(2023)
    args = get_args_parser().parse_args(args=[])
    main(args)