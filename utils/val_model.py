from tqdm.auto import tqdm
import torch
import numpy as np
from utils.evaluate import ConfusionMatrix
from  utils.new_eval import ConfuseMatrixMeter
import cv2

def count_img(img):
    counts = torch.zeros(256)
    if isinstance(img, np.ndarray):
        # 创建一个示例narray
        arr = img.flatten()
        # 使用bincount函数计算每个类别的数量
        counts = np.bincount(arr)
    # 判断变量的类型是否为张量
    if isinstance(img, torch.Tensor):

        for i in range(256):
            counts[i] = torch.sum(img == i)

    print(counts)

def save_batch_img(save_path,img_tensor,name_list):
    batch_size,c,h,w = img_tensor.shape

    img_array=np.asarray(img_tensor)
    img_array=img_array.transpose(0,2,3,1)

    for i in range(batch_size):
        cv2.imwrite(save_path + name_list[i], img_array[i,:,:,:]*255)


def val_one_epoch( model,  criterion, device, dataloader_val,args):
    epoch_val_loss = []
    batch_hist_val=ConfusionMatrix(num_classes=args.classes)
    model.eval()
    with torch.no_grad():
        for b_A, b_B, label,_ in tqdm(dataloader_val):
            pred = model(b_A.cuda(), b_B.cuda())

            loss = criterion(pred, label.to(device).float())

            epoch_val_loss.append(loss.item())
            pred = pred.cpu().detach().numpy()
            pred = np.argmax(pred, axis=1)

            label = label.cpu().detach().numpy()
            if args.one_hot_flag:
                label = np.argmax(label, axis=1)
            batch_hist_val.update(label,pred)

    return (batch_hist_val.compute(), np.nanmean(epoch_val_loss))

def val_one_epoch_test( model,  criterion, device, dataloader_val,args):
    epoch_val_loss = []
    batch_hist_val=ConfuseMatrixMeter(n_class=args.classes)
    model.eval()
    with torch.no_grad():
        for b_A, b_B, label,_ in tqdm(dataloader_val):
            pred = model(b_A.cuda(), b_B.cuda())

            loss = criterion(pred, label.to(device).float())

            epoch_val_loss.append(loss.item())
            pred = pred.cpu().detach().numpy()
            pred = np.argmax(pred, axis=1)

            label = label.cpu().detach().numpy()
            if args.one_hot_flag:
                label = np.argmax(label, axis=1)
            batch_hist_val.update_cm(pred,label)

    return (batch_hist_val.get_scores(), np.nanmean(epoch_val_loss))


def val_one_epoch_STA( model,  criterion, device, dataloader_val,args):
    epoch_val_loss = []
    batch_hist_val=ConfusionMatrix(num_classes=args.classes)
    model.eval()
    with torch.no_grad():
        for b_A, b_B, label in tqdm(dataloader_val):
            pred = model(b_A.cuda(), b_B.cuda())
            label = torch.argmax(label, 1).unsqueeze(1)
            loss = criterion(pred, label.to(device).float())

            epoch_val_loss.append(loss.item())

            pred = (pred > 1).cpu().detach().numpy()
            label = label.cpu().detach().numpy()
            epoch_val_loss.append(loss.item())
            batch_hist_val.update(label, pred)

    return (batch_hist_val.compute(), np.nanmean(epoch_val_loss))



def val_one_epoch_DSAMNet( model,  criterion1,criterion2, device, dataloader_val,args):
    epoch_val_loss = []
    batch_hist_val = ConfusionMatrix(num_classes=args.classes)
    model.eval()
    with torch.no_grad():
        for b_A, b_B, label,_ in tqdm(dataloader_val):
            label = label.to(device, dtype=torch.float)
            pred, ds2, ds3 = model(b_A.cuda(), b_B.cuda())
            dsloss2 = criterion2(ds2, label)
            dsloss3 = criterion2(ds3, label)

            Dice_loss = 0.5 * (dsloss2 + dsloss3)

            # contrative loss
            label = torch.argmax(label, 1).unsqueeze(1)
            CT_loss = criterion1(pred, label.float())

            # CD loss
            loss = CT_loss + args.wDice * Dice_loss

            epoch_val_loss.append(loss.item())

            pred = (pred > 1).cpu().detach().numpy()
            label = label.cpu().detach().numpy()
            epoch_val_loss.append(loss.item())
            batch_hist_val.update(label, pred)

        return (batch_hist_val.compute(), np.nanmean(epoch_val_loss))

def val_one_epoch_Tiny_CD( model,  criterion, device, dataloader_val,args):
    epoch_val_loss = []
    batch_hist_val = ConfusionMatrix(num_classes=args.classes)
    model.eval()
    with torch.no_grad():
        for b_A, b_B, label,_ in tqdm(dataloader_val):
            label = label.to(device, dtype=torch.float)
            pred= model(b_A.cuda(), b_B.cuda())

            loss = criterion(pred, label.float())


            epoch_val_loss.append(loss.item())

            pred = (pred > 0.5).cpu().detach().numpy()
            label = label.cpu().detach().numpy()
            epoch_val_loss.append(loss.item())
            batch_hist_val.update(label, pred)

        return (batch_hist_val.compute(), np.nanmean(epoch_val_loss))