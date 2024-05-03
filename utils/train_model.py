import torch
import numpy as np
import torch
from tqdm.auto import tqdm
from utils.evaluate import ConfusionMatrix
np.seterr(divide='ignore', invalid='ignore')


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

def train_one_epoch(model,criterion, device, dataloader_train,optimizer,scaler,args):
    # find loss function by args

    model.train()
    epoch_train_loss = []
    batch_hist_train=ConfusionMatrix(num_classes=args.classes)
    #batch_hist_train = np.array([[0,0], [0, 0]]).astype(float)
    for b_A, b_B,label in tqdm(dataloader_train):

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            pred = model(b_A.to(device),b_B.to(device))
            loss = criterion(pred,label.to(device).float())
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        pred = pred.cpu().detach().numpy()
        pred = np.argmax(pred, axis=1)

        label = label.cpu().detach().numpy()
        if args.one_hot_flag:
            label = np.argmax(label, axis=1)
        epoch_train_loss.append(loss.item())
        batch_hist_train.update(label,pred)

    return (batch_hist_train.compute(), np.nanmean(epoch_train_loss))

def train_one_epoch_STA(model,criterion, device, dataloader_train,optimizer,scaler,args):
    # find loss function by args

    model.train()
    epoch_train_loss = []
    batch_hist_train=ConfusionMatrix(num_classes=args.classes)
    #batch_hist_train = np.array([[0,0], [0, 0]]).astype(float)
    for b_A, b_B,label in tqdm(dataloader_train):

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            pred = model(b_A.to(device),b_B.to(device))
            label = torch.argmax(label, 1).unsqueeze(1)
            loss = criterion(pred,label.to(device).float())
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        pred = (pred > 0.5).cpu().detach().numpy()

        label = label.cpu().detach().numpy()
        epoch_train_loss.append(loss.item())
        batch_hist_train.update(label, pred)

    return (batch_hist_train.compute(), np.nanmean(epoch_train_loss))




def train_one_epoch_DSAMNet(model,criterion1,criterion2, device, dataloader_train,optimizer,scaler,args):
    model.train()
    epoch_train_loss = []
    batch_hist_train=ConfusionMatrix(num_classes=args.classes)
    for b_A, b_B, label in tqdm(dataloader_train):
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            pred, ds2, ds3 = model(b_A.to(device), b_B.to(device))  # 两个DS监督backbone进行有效训练
            label = label.to(device, dtype=torch.float)
            # Diceloss
            dsloss2 = criterion2(ds2, label)
            dsloss3 = criterion2(ds3, label)

            Dice_loss = 0.5 * (dsloss2 + dsloss3)

            # contrative loss
            label = torch.argmax(label, 1).unsqueeze(1)
            CT_loss = criterion1(pred, label.float())

            # CD loss
            loss = CT_loss + args.wDice * Dice_loss

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()



        pred = (pred > 1).cpu().detach().numpy()
        label = label.cpu().detach().numpy()
        epoch_train_loss.append(loss.item())
        batch_hist_train.update(label, pred)

    return (batch_hist_train.compute(), np.nanmean(epoch_train_loss))





