
import numpy as np
import logging
import os
import sys
import cv2
# 混淆矩阵

#接收numpy类型混淆矩阵进行计算
class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes  # 分类个数(加了背景之后的)
        self.mat = None  # 混淆矩阵

    def update(self, gt, pred):  # 计算混淆矩阵,a = Ture,b = Predict
        n = self.num_classes
        if self.mat is None:  # 创建混淆矩阵
            self.mat = np.zeros((n, n), dtype=np.int64)

        k = (gt >= 0) & (gt < n)
        inds = n * gt[k].astype(int) + pred[k]  # 统计像素真实类别a[k]被预测成类别b[k]的个数(这里的做法很巧妙)
        self.mat += np.bincount(inds, minlength=n ** 2).reshape(n, n)

    def reset(self):
        if self.mat is not None:
            self.mat.zero_()

    def compute(self):  # 计算分割任务的性能指标
        hist = self.mat.astype(float)
        total = hist.sum()

        epsilon = 1e-7
        n_class = hist.shape[0]
        tp = np.diag(hist)
        sum_a1 = hist.sum(axis=1)
        sum_a0 = hist.sum(axis=0)
        # ---------------------------------------------------------------------- #
        # 1. Accuracy & Class Accuracy
        # ---------------------------------------------------------------------- #
        acc = tp.sum() / (total + np.finfo(np.float32).eps)

        # recall
        recall = tp / (sum_a1 + np.finfo(np.float32).eps)
        mean_R = np.nanmean(recall)

        # precision
        precision = tp / (sum_a0 + np.finfo(np.float32).eps)
        mean_P = np.nanmean(precision)

        # F1 score
        F1 = 2 * recall * precision / (recall + precision + np.finfo(np.float32).eps)
        mean_F1 = np.nanmean(F1)
        # ---------------------------------------------------------------------- #
        # 2. Frequency weighted Accuracy & Mean IoU
        # ---------------------------------------------------------------------- #
        iu = tp / (sum_a1 + hist.sum(axis=0) - tp + np.finfo(np.float32).eps)
        mean_iu = np.nanmean(iu)

        po = np.trace(hist) / total
        pe = np.sum(np.sum(hist, axis=0) * np.sum(hist, axis=1)) / total ** 2

        kap = (po - pe) / (1 - pe + epsilon)

        cls_iou = dict(zip(['Iou_' + str(i) for i in range(n_class)], iu))

        cls_precision = dict(zip(['Precision_' + str(i) for i in range(n_class)], precision))
        cls_recall = dict(zip(['Recall_' + str(i) for i in range(n_class)], recall))
        cls_F1 = dict(zip(['F1_' + str(i) for i in range(n_class)], F1))

        score_dict = {'ACC': acc, 'mIou': mean_iu, 'mF1': mean_F1,'mR_score':mean_R,'mP_score':mean_P, 'Kapa': kap}
        score_dict.update(cls_iou)
        score_dict.update(cls_F1)
        score_dict.update(cls_precision)
        score_dict.update(cls_recall)

        return score_dict


def log_eva_txt(epoch_index, evalist, logger, loss):
    logger.setLevel(logging.INFO)  # Log等级总开关  此时是INFO
    # 第二步，创建一个handler，用于写入日志文件
    log_name=os.path.basename(sys.argv[0])
    logfile = './' + log_name[:-3] + '.txt'
    fh = logging.FileHandler(logfile, mode="a")  # open的打开模式这里可以进行参考
    fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关

    # 第三步，再创建一个handler，用于输出到控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)  # 输出到console的log等级的开关

    # 第四步，定义handler的输出格式（时间，文件，行数，错误级别，错误提示）
    formatter = logging.Formatter("%(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # 第五步，将logger添加到handler里面
    logger.addHandler(fh)
    logger.addHandler(ch)
    #print("Epoch:{},Loss:{},Acc:{},P_score:{},R_score:{},F1_score:{},Kappa:{},MD:{},FA:{},mIOU:{}\n".format(epoch_index, loss, *(evalist)))
    logger.info("Epoch:{},Loss:{},Acc:{},P_score:{},R_score:{},F1_score:{},Kappa:{},IOU:{}\n".format(epoch_index, loss, *(evalist)))

    logger.removeHandler(fh)
    logger.removeHandler(ch)
    fh.close()
    ch.close()





def log_eva(epoch_index, eva_dict, loss,writer=None):
    if writer:
        for key,value in eva_dict.items():
            writer.add_scalar(key,value, epoch_index)  # 标量名，y轴，x轴
        writer.add_scalar("Loss", loss, epoch_index)
    print("\nEpoch:{},Loss:{:.5f},Acc:{:.4f},P_score:{:.4f},R_score:{:.4f},F1_score:{:.5f},Kappa:{:.5f},mIOU:{:.4f},F1_1:{:.4f},Iou_1:{:.4f}\n".format(
        epoch_index,loss,eva_dict["ACC"],eva_dict["mP_score"],eva_dict["mR_score"],eva_dict["mF1"],eva_dict["Kapa"],eva_dict["mIou"],eva_dict["F1_1"],eva_dict["Iou_1"]))


if __name__ == '__main__':
    label=cv2.imread(r"F:\Pycharm_program\lunwen\demo\label\1325.png",0)//255
    pred=cv2.imread(r"F:\Pycharm_program\lunwen\demo\pred\1325.png",0)//255
    max=ConfusionMatrix(2)
    max.update(label,pred)

    print(max.compute())