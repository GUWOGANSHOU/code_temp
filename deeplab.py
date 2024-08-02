import torchkeras
from argparse import Namespace
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os
from torchkeras.data import resize_and_pad_image
from torchkeras.plots import joint_imgs_col
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import glob

config = Namespace(
    img_size=128,
    lr=1e-4,
    batch_size=4,
)


class MyDataset(Dataset):
    def __init__(self, img_files, img_size, transforms=None):
        self.__dict__.update(locals())

    def __len__(self) -> int:
        return len(self.img_files)

    def get(self, index):
        img_path = self.img_files[index]
        mask_path = img_path.replace('images', 'masks').replace('.jpg', '.png')
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        return image, mask

    def __getitem__(self, index):
        image, mask = self.get(index)

        image = resize_and_pad_image(image, self.img_size, self.img_size)
        mask = resize_and_pad_image(mask, self.img_size, self.img_size)

        image_arr = np.array(image, dtype=np.float32) / 255.0

        mask_arr = np.array(mask, dtype=np.float32)
        mask_arr = np.where(mask_arr > 100.0, 1.0, 0.0).astype(np.int64)

        sample = {
            "image": image_arr,
            "mask": mask_arr
        }

        if self.transforms is not None:
            sample = self.transforms(**sample)

        sample['mask'] = sample['mask'][None, ...]

        return sample

    def show_sample(self, index):
        image, mask = self.get(index)
        image_result = joint_imgs_col(image, mask)
        return image_result


def get_train_transforms():
    return A.Compose(
        [
            A.OneOf([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)]),
            ToTensorV2(p=1),
        ],
        p=1.0
    )


def get_val_transforms():
    return A.Compose(
        [
            ToTensorV2(p=1),
        ],
        p=1.0
    )


# 训练和验证图像分别存放在两个不同的目录下
train_dir = 'datasets/lane_lines/train/images'
val_dir = 'datasets/lane_lines/val/images'
# 获取所有图像文件的路径
train_imgs = glob.glob(os.path.join(train_dir, '*.jpg'))  # 图像格式为 .jpg
val_imgs = glob.glob(os.path.join(val_dir, '*.jpg'))  # 图像格式为 .jpg

train_transforms = get_train_transforms()
val_transforms = get_val_transforms()

ds_train = MyDataset(train_imgs, img_size=config.img_size, transforms=train_transforms)
ds_val = MyDataset(val_imgs, img_size=config.img_size, transforms=val_transforms)

dl_train = DataLoader(ds_train, batch_size=config.batch_size)
dl_val = DataLoader(ds_val, batch_size=config.batch_size)

ds_train.show_sample(10)

import segmentation_models_pytorch as smp
import torch
from torchkeras import KerasModel
from torch.nn import functional as F
from torchkeras.metrics import IOU
from torchkeras.kerascallbacks import WandbCallback

num_classes = 1
net = smp.DeepLabV3Plus(
    encoder_name="mobilenet_v2",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights='imagenet',  # use `imagenet` pretrained weights for encoder initialization
    in_channels=3,  # model input channels (1 for grayscale images, 3 for RGB, etc.)
    classes=num_classes,  # model output channels (number of classes in your dataset)
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 由于输入数据batch结构差异，需要重写StepRunner并覆盖
class StepRunner:
    def __init__(self, net, loss_fn, accelerator, stage="train", metrics_dict=None,
                 optimizer=None, lr_scheduler=None
                 ):
        self.net, self.loss_fn, self.metrics_dict, self.stage = net, loss_fn, metrics_dict, stage
        self.optimizer, self.lr_scheduler = optimizer, lr_scheduler
        self.accelerator = accelerator

        if self.stage == 'train':
            self.net.train()
        else:
            self.net.eval()

    def __call__(self, batch):
        features, labels = batch['image'], batch['mask']

        # loss
        preds = self.net(features)
        loss = self.loss_fn(preds, labels)

        # backward()
        if self.optimizer is not None and self.stage == "train":
            self.accelerator.backward(loss)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()

        all_preds = self.accelerator.gather(preds)
        all_labels = self.accelerator.gather(labels)
        all_loss = self.accelerator.gather(loss).sum()

        # losses
        step_losses = {self.stage + "_loss": all_loss.item()}

        # metrics
        step_metrics = {self.stage + "_" + name: metric_fn(all_preds, all_labels).item()
                        for name, metric_fn in self.metrics_dict.items()}

        if self.optimizer is not None and self.stage == "train":
            step_metrics['lr'] = self.optimizer.state_dict()['param_groups'][0]['lr']

        return step_losses, step_metrics


KerasModel.StepRunner = StepRunner


class DiceLoss(nn.Module):
    def __init__(self, smooth=0.001, num_classes=1, weights=None):
        super().__init__()
        self.smooth = smooth
        self.num_classes = num_classes
        self.weights = weights

    def forward(self, logits, targets):

        # binary
        if self.num_classes == 1:
            preds = logits.contiguous().view(logits.size()[0], -1).sigmoid()
            targets = targets.contiguous().view(targets.size()[0], -1).float()
            loss = self.compute_loss(preds, targets)
            return loss

        # multiclass
        else:
            preds = logits.softmax(axis=1).contiguous().view(
                logits.size()[0], self.num_classes, -1)
            t = targets.contiguous().view(
                targets.size()[0], -1)
            targets = torch.nn.functional.one_hot(t, self.num_classes).permute(0, 2, 1)
            totalLoss = 0.0
            for i in range(self.num_classes):
                diceLoss = self.compute_loss(preds[:, i], targets[:, i])
                if self.weights is not None:
                    diceLoss *= self.weights[i]
                totalLoss += diceLoss
            return totalLoss

    def compute_loss(self, preds, targets):
        a = torch.sum(preds * targets, 1)  # |X?Y|
        b = torch.sum(preds * preds, 1) + self.smooth  # |X|
        c = torch.sum(targets * targets, 1) + self.smooth  # |Y|
        score = (2 * a) / (b + c)
        loss = torch.mean(1 - score)
        return loss


class MixedLoss(nn.Module):
    def __init__(self, bce_ratio=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_ratio = bce_ratio

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets.float())
        dice_loss = self.dice(logits, targets)
        total_loss = bce_loss * self.bce_ratio + dice_loss * (1 - self.bce_ratio)
        return total_loss


optimizer = torch.optim.AdamW(net.parameters(), lr=config.lr)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer=optimizer,
    T_max=8,
    eta_min=0
)

metrics_dict = {'iou': IOU(num_classes=1)}

model = KerasModel(net,
                   loss_fn=MixedLoss(bce_ratio=0.5),
                   metrics_dict=metrics_dict,
                   optimizer=optimizer,
                   lr_scheduler=lr_scheduler
                   )

dfhistory = model.fit(train_data=dl_train,
                      val_data=dl_val,
                      epochs=100,
                      ckpt_path='checkpoint.pt',
                      patience=10,
                      monitor="val_iou",
                      mode="max",
                      mixed_precision='no',
                      plot=True
                      )

from matplotlib import pyplot as plt

metrics_dict = {'iou': IOU(num_classes=1, if_print=True)}

model = KerasModel(net,
                   loss_fn=MixedLoss(bce_ratio=0.5),
                   metrics_dict=metrics_dict,
                   optimizer=optimizer,
                   lr_scheduler=lr_scheduler
                   )
model.evaluate(dl_val)
batch = next(iter(dl_val))

with torch.no_grad():
    model.eval()
    logits = model(batch["image"].cuda())

pr_masks = logits.sigmoid()

for image, gt_mask, pr_mask in zip(batch["image"], batch["mask"], pr_masks):
    plt.figure(figsize=(16, 10))

    plt.subplot(1, 3, 1)
    plt.imshow(image.numpy().transpose(1, 2, 0))  # convert CHW -> HWC
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(gt_mask.numpy().squeeze())
    plt.title("Ground truth")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pr_mask.cpu().numpy().squeeze())
    plt.title("Prediction")
    plt.axis("off")

    plt.show()