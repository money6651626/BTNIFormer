import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib
from sklearn.manifold import TSNE

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

"""
参考 
https://github.com/gmayday1997/SceneChangeDet/blob/master/code/utils/metric.py
https://github.com/rentainhe/visualization/blob/master/visualize/grid_attention_visualization/visualize_attention_map_V2.py
"""

def visualize_grid_attention_v2(img, save_path, attention_mask, ratio=1, cmap="jet", save_image=True,
                                save_original_image=False,reduce_method="TSNE"):
    """
    img:   an  Image tensor batch(b,c,h,w)
    save_path:  image file path to save
    attention_mask:  2-D attention map with np.array type, e.g, (h, w) or (w, h)
    ratio:  scaling factor to scale the output h and w
    cmap:  attention style, default: "jet"
    """
    c,h,w=attention_mask.shape
    if reduce_method=="TSNE":
        mask_img=attention_mask.reshape(c, h * w)
        mask_img = np.transpose(mask_img, (1, 0))
        tsne = TSNE(perplexity=c, n_components=1, init='pca', n_iter=250)
        mask_out=tsne.fit_transform(mask_img)
        mask_out = np.squeeze(mask_out,1)
        mask_out=mask_out.reshape(h,w)

    else:
        mask_out = np.average(attention_mask, 0)


    plt.subplots(nrows=1, ncols=1, figsize=(0.02 * h, 0.02 * w))

    img_h, img_w = img.size[0], img.size[1]
    # scale the image
    img_h, img_w = int(img_h * ratio), int(img_w * ratio)
    img = img.resize((img_h, img_w))
    plt.imshow(img, alpha=1)
    plt.axis('off')

    # normalize the attention map
    normed_mask = mask_out / mask_out.max()
    normed_mask = (normed_mask * 255).astype('uint8')
    plt.imshow(normed_mask, alpha=0.5, interpolation='nearest', cmap=cmap)  # cmap指定绘制色域空间

    if save_image:
        # build save path
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        img_name = img_path.split('/')[-1].split('.')[0] + "_with_attention.jpg"
        img_with_attention_save_path = os.path.join(save_path, img_name)

        # pre-process and save image
        print("save image to: " + save_path + " as " + img_name)
        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(img_with_attention_save_path, dpi=300)

img_path="F:/Pycharm_program/lunwen/Train_version/baseline/test_save/A/00000.jpg"
save_path="F:\\Pycharm_program\\lunwen\\Train_version\\baseline\\test_save\\attention_map/"
attention_mask_path="F:/Pycharm_program/lunwen/Train_version/baseline/test_save/label/00000.jpg"
img = Image.open(img_path, mode='r')

attention_mask=torch.randint(0,1,(4,256,256)).float()
#attention_mask = cv2.imread(attention_mask_path,cv2.IMREAD_GRAYSCALE)
visualize_grid_attention_v2(img,save_path,attention_mask,reduce_method="TSNE")