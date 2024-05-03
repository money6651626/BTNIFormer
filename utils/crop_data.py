import cv2
import os
from tqdm.auto import tqdm

crop_path="E:/DataSets/Change_Detection/LEVIR_CD/val/"
save_path ="E:/DataSets/Change_Detection/LEVIR_CD/crop/"
crop_size=(256,256)#w,h
down_rate=2

def img_crop(crop_path,save_path,crop_size,down_rate):
    out_path =save_path
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    imgs_list=os.listdir(crop_path)
    for img_name in tqdm(imgs_list):
        idx=0
        img=cv2.imread(crop_path+img_name)
        img_shape=img.shape #h,w,c

        if img_shape[0]%crop_size[0]==0:
            w_std=crop_size[0]
        else:
            w_std=crop_size[0]//down_rate
        if img_shape[1]%crop_size[1]==0:
            h_std=crop_size[1]
        else:
            h_std=crop_size[1]//down_rate
        for w_num in range(0, img_shape[0],w_std ):
            for h_num in range(0, img_shape[1], h_std):

                if (w_num + crop_size[0]) > img_shape[1] or (h_num + crop_size[1]) > img_shape[0]:
                    #步长越界
                    continue

                img_out=img[w_num:w_num + crop_size[0], h_num:h_num + crop_size[1]]
                img_path=out_path+"/"+crop_path+img_name+"_"+str(idx)+".png"
                img_msg = cv2.imwrite(img_path,img_out)
                if not img_msg:
                    print(img_path + "出错")
                    exit()
                idx+=1
    print("finished!")
img_crop(crop_path,save_path,crop_size,down_rate)