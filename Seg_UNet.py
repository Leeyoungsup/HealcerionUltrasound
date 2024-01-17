import matplotlib.pyplot as plt
import numpy as np
import helper
import time
import datetime
import torch.nn as nn
import torchvision.models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
import torchvision.utils
import torch
import pandas as pd
from torchinfo import summary
from PIL import Image
from torchvision.transforms import ToTensor
from glob import glob
from torch.utils.data import Dataset, DataLoader, random_split
from copy import copy
from collections import defaultdict
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import time
from sklearn.metrics import classification_report
from tqdm import tqdm
import math
from torcheval.metrics import BinaryAccuracy
import os
import torchmetrics
import timm
import segmentation_models_pytorch as smp
import random
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
batch_size = 24
image_count = 50
img_size = 224
tf = ToTensor()


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


train_df = pd.read_csv(
    '../../data/segmentationDDH/train_aug_segmentation_dataset.csv')
val_df = pd.read_csv(
    '../../data/segmentationDDH/val_aug_segmentation_dataset.csv')

train_img_list = train_df['file name'].to_list()
train_label_list = train_df['standard mask'].to_list()
train_case_list = train_df['case'].to_list()
train_img_path = '../../data/segmentationDDH/aug_dataset/train/'
val_img_list = val_df['file name'].to_list()
val_label_list = val_df['standard mask'].to_list()
val_case_list = val_df['case'].to_list()
val_img_path = '../../data/segmentationDDH/aug_dataset/val/'

val_image = torch.zeros((len(val_img_list), 3, img_size, img_size))
val_mask = torch.zeros(
    (len(val_img_list), 5, img_size, img_size), dtype=torch.uint8)
train_image = torch.zeros((len(train_img_list), 3, img_size, img_size))
train_mask = torch.zeros(
    (len(train_img_list), 5, img_size, img_size), dtype=torch.uint8)

for i in tqdm(range(len(train_img_list))):
    train_image[i] = tf(np.array(Image.open(
        train_img_path+str(train_case_list[i])+'/image/'+train_img_list[i]).resize((img_size, img_size))))
    train_mask[i, 1] = tf(np.array(Image.open(
        train_img_path+str(train_case_list[i])+'/mask/'+str(train_label_list[i]).zfill(5)+'/1'+train_img_list[i][train_img_list[i].find('_'):]).resize((img_size, img_size))))
    train_mask[i, 2] = tf(np.array(Image.open(
        train_img_path+str(train_case_list[i])+'/mask/'+str(train_label_list[i]).zfill(5)+'/2'+train_img_list[i][train_img_list[i].find('_'):]).resize((img_size, img_size))))
    train_mask[i, 3] = tf(np.array(Image.open(
        train_img_path+str(train_case_list[i])+'/mask/'+str(train_label_list[i]).zfill(5)+'/3'+train_img_list[i][train_img_list[i].find('_'):]).resize((img_size, img_size))))
    train_mask[i, 4] = tf(np.array(Image.open(
        train_img_path+str(train_case_list[i])+'/mask/'+str(train_label_list[i]).zfill(5)+'/4'+train_img_list[i][train_img_list[i].find('_'):]).resize((img_size, img_size))))
    train_mask[i, 0] = torch.where(
        (train_mask[i, 1]+train_mask[i, 2]+train_mask[i, 3]+train_mask[i, 4]) == 0, 1, 0)


for i in tqdm(range(len(val_img_list))):
    val_image[i] = tf(np.array(Image.open(
        val_img_path+str(val_case_list[i])+'/image/'+val_img_list[i]).resize((img_size, img_size))))
    val_mask[i, 1] = tf(np.array(Image.open(
        val_img_path+str(val_case_list[i])+'/mask/'+str(val_label_list[i]).zfill(5)+'/1.png').resize((img_size, img_size))))
    val_mask[i, 2] = tf(np.array(Image.open(
        val_img_path+str(val_case_list[i])+'/mask/'+str(val_label_list[i]).zfill(5)+'/2.png').resize((img_size, img_size))))
    val_mask[i, 3] = tf(np.array(Image.open(
        val_img_path+str(val_case_list[i])+'/mask/'+str(val_label_list[i]).zfill(5)+'/3.png').resize((img_size, img_size))))
    val_mask[i, 4] = tf(np.array(Image.open(
        val_img_path+str(val_case_list[i])+'/mask/'+str(val_label_list[i]).zfill(5)+'/4.png').resize((img_size, img_size))))
    val_mask[i, 0] = torch.where(
        (val_mask[i, 1]+val_mask[i, 2]+val_mask[i, 3]+val_mask[i, 4]) == 0, 1, 0)


class CustomDataset(Dataset):
    def __init__(self, image_list, label_list):
        self.img_path = image_list
        self.label = label_list

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        image_path = self.img_path[idx]
        label_path = self.label[idx]
        return image_path, label_path


train_dataset = CustomDataset(train_image, train_mask[:, 1:])

val_dataset = CustomDataset(val_image, val_mask[:, 1:])
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
validation_dataloader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

model = smp.Unet('mobilenet_v2', in_channels=3, classes=4).to(device)
summary(model, (batch_size, 3, img_size, img_size))


def dice_loss(pred, target, num_classes=4):
    smooth = 1.
    dice_per_class = torch.zeros(num_classes).to(pred.device)

    for class_id in range(num_classes):
        pred_class = pred[:, class_id, ...]
        target_class = target[:, class_id, ...]

        intersection = torch.sum(pred_class * target_class)
        A_sum = torch.sum(pred_class * pred_class)
        B_sum = torch.sum(target_class * target_class)

        dice_per_class[class_id] = 1 - \
            (2. * intersection + smooth) / (A_sum + B_sum + smooth)

    return torch.mean(dice_per_class)


train_loss_list = []
val_loss_list = []
train_acc_list = []
val_acc_list = []
MIN_loss = 5000
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4)
metrics = defaultdict(float)
for epoch in range(300):
    train = tqdm(train_dataloader)
    count = 0
    running_loss = 0.0
    acc_loss = 0
    for x, y in train:
        model.train()
        y = y.to(device).float()
        count += 1
        x = x.to(device).float()
        optimizer.zero_grad()  # optimizer zero 로 초기화
        predict = model(x).to(device)
        cost = dice_loss(predict, y)  # cost 구함
        acc = 1-dice_loss(predict, y)
        cost.backward()  # cost에 대한 backward 구함
        optimizer.step()
        running_loss += cost.item()
        acc_loss += acc
        y = y.to('cpu')

        x = x.to('cpu')
        train.set_description(
            f"epoch: {epoch+1}/{300} Step: {count+1} dice_loss : {running_loss/count:.4f} dice_score: {1-running_loss/count:.4f}")
    train_loss_list.append((running_loss/count))
    train_acc_list.append((acc_loss/count).cpu().detach().numpy())
# validation
    val = tqdm(validation_dataloader)
    model.eval()
    count = 0
    val_running_loss = 0.0
    acc_loss = 0
    with torch.no_grad():
        for x, y in val:
            y = y.to(device).float()
            count += 1
            x = x.to(device).float()

            predict = model(x).to(device)
            cost = dice_loss(predict, y)  # cost 구함
            acc = 1-dice_loss(predict, y)
            val_running_loss += cost.item()
            acc_loss += acc
            y = y.to('cpu')
            x = x.to('cpu')
            val.set_description(
                f"Validation epoch: {epoch+1}/{300} Step: {count+1} dice_loss : {val_running_loss/count:.4f}  dice_score: {1-val_running_loss/count:.4f}")
        val_loss_list.append((val_running_loss/count))
        val_acc_list.append((acc_loss/count).cpu().detach().numpy())

    if MIN_loss > (val_running_loss/count):
        torch.save(model.state_dict(),
                   '../../model/segmentation/Unet_callback.pt')
        MIN_loss = (val_running_loss/count)

plt.figure(figsize=(10, 10))

plt.title('dice_graph')
plt.plot(np.arange(epoch+1), train_acc_list, label='train_dice')
plt.plot(np.arange(epoch+1), val_acc_list, label='train_dice')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.ylim([0, 1])
plt.legend()
plt.savefig('../../model/segmenatation/Unet.png')
torch.save(model.state_dict(),
           '../../model/segmentation/Unet.pt')
print('batch size= 4')
print('image size= 224')
print('learning rate= 0.0001')
