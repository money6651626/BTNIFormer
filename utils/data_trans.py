import torch.utils.data as Data
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
import torchvision.transforms
from PIL import Image
import random
from utils.data_utils import CDDataAugmentation
from torch.utils.data import DataLoader


def sub_test_data(dataset, expansion=0.003):
    out_size = int(expansion * len(dataset))
    if out_size < 4:
        out_size = 4
    out_datas, space_datas = torch.utils.data.random_split(dataset, [out_size, len(dataset) - out_size])
    return out_datas



class CD_Dataset(Data.Dataset):
    def __init__(self, dataA_path, dataB_path,pic_size=256,label_path=None, use_type_train=True, one_hot=False,data_name=None):
        self.one_hot = one_hot

        self.use_type_train = use_type_train
        self.dataA_path = dataA_path
        self.dataB_path = dataB_path
        self.label_path = label_path
        self.pic_size=pic_size
        self.all_len=0
        if use_type_train:
            self.augm = CDDataAugmentation(
                img_size=self.pic_size,
                with_random_hflip=True,
                with_random_vflip=True,
                with_scale_random_crop=True,
                with_random_blur=True,
                random_color_tf=True,
                time_exchange=True,
                label_onehot=one_hot,
                data_name=data_name,
            )
        else:
            self.augm = CDDataAugmentation(
                img_size=self.pic_size,
                label_onehot = one_hot,
                data_name=data_name
            )

    def __len__(self):
        num_a=len(os.listdir(self.dataA_path))
        num_b=len(os.listdir(self.dataB_path))
        num_c=len(os.listdir(self.label_path))

        if num_a == num_b == num_c:
            #self.all_len=num_a
            return len(os.listdir(self.dataA_path))
        else:
            print(num_a, num_b, num_c)
            print("图片数量不匹配")
            exit()


    def get_img_bypath(self , path, idx,label=False):

        img_name = os.listdir(path)[idx]
        if label:
            img = np.asarray(Image.open(os.path.join(path, img_name)).convert('L'))
            if not ((img == 0) | (img == 1)).all():#标签如果只为0，1那么不处理
                img = img//255
            else:
                pass
        else:
            img = np.asarray(Image.open(os.path.join(path,img_name)).convert('RGB'))
        return img

    def __getitem__(self, idx):

        img_A = self.get_img_bypath(self.dataA_path, idx)
        img_B = self.get_img_bypath(self.dataB_path, idx)
        label = self.get_img_bypath(self.label_path, idx,label=True) #3×long×width
        [img_A, img_B], [label] = self.augm.transform([img_A, img_B], [label], to_tensor=True,nomalize_spe=True)

        #return img_A, img_B, label

        if self.use_type_train:
            return img_A, img_B, label
        else:
            img_name = os.listdir(self.dataA_path)[idx]
            return img_A, img_B, label, img_name


def show_img(pic):
    cv2.imshow("233",np.asarray(pic[0]))
    cv2.waitKey(0)

def get_dataloader(args):
    data_path =os.path.join(args.data_path,args.dataset_name)


    train_dataset = CD_Dataset(dataA_path=os.path.join(data_path,"train/A/"),
                               dataB_path=os.path.join(data_path,"train/B/"),
                               label_path=os.path.join(data_path,"train/label/"),
                               use_type_train=True, one_hot=args.one_hot_flag,pic_size=args.img_size,data_name=args.dataset_name)
    val_dataset = CD_Dataset(dataA_path=os.path.join(data_path,  "val/A/"),
                             dataB_path=os.path.join(data_path, "val/B/"),
                             label_path=os.path.join(data_path, "val/label/"),
                             use_type_train=False, one_hot=args.one_hot_flag,pic_size=args.img_size,data_name=args.dataset_name)
    test_dataset=CD_Dataset(dataA_path=os.path.join(data_path,  "test/A/"),
                             dataB_path=os.path.join(data_path, "test/B/"),
                             label_path=os.path.join(data_path, "test/label/"),
                             use_type_train=False, one_hot=args.one_hot_flag,pic_size=args.img_size,data_name=args.dataset_name)

    if args.subdata:
        train_dataset = sub_test_data(train_dataset, expansion=0.1)
        val_dataset = sub_test_data(val_dataset, expansion=0.1)
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   pin_memory=True,
                                   num_workers=args.num_workers,
                                   )


    val_data_loader = DataLoader(val_dataset,
                                   batch_size=args.batch_size ,
                                   shuffle=True,
                                   pin_memory=True,
                                   num_workers=args.num_workers,
                                   )

    test_data_loader = DataLoader(test_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 pin_memory=True,
                                 num_workers=args.num_workers,
                                 )
    if args.test_only:
       return test_data_loader
    else:
        return (train_data_loader,val_data_loader)
