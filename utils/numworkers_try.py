import time
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from utils.data_trans import MyDataset
train_tfm_unchange = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
])

train_tfm_change = transforms.Compose([
    transforms.RandomGrayscale(p=0.5)
])

if __name__ == '__main__':
    data_path="F:\\Pycharm_program\\lunwen\\datasets\\CDD/"
    train_dataset = MyDataset(dataA_path=data_path + "train/A/",
                              dataB_path=data_path + "train/B/",
                              label_path=data_path + "train/label/",
                              use_type_train=True, one_hot=True, transform_change=train_tfm_change,
                              trasnform_unchange=train_tfm_unchange)



    BATCH_SIZE = 8




    train_data_loader = DataLoader(train_dataset,
                                        batch_size=BATCH_SIZE,
                                        shuffle=True,
                                        pin_memory=True,
                                   )

    for num_workers in range(20):
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,pin_memory=True, num_workers=num_workers)
        # training ...
        start = time.time()
        for epoch in range(1):
            for step, (batch_x, batch_y,bt) in enumerate(train_loader):
                pass
        end = time.time()
        print('num_workers is {} and it took {} seconds'.format(num_workers, end - start))