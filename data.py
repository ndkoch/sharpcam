import os
import random
from torch.utils.data import Dataset
import numpy as np
import cv2 as cv
import albumentations as A
from PIL import Image

class FramePacketPrediction(Dataset):
  def __init__(self, path, transform, packet_len):
    self.path = path
    self.transform = transform
    self.total_frames = os.listdir(path)
    self.packet_len = packet_len

  def __len__(self):
    return len(self.total_frames)

  def __getitem__(self, idx):
    img_num = int((self.total_frames[idx])[:-4])
    firstImg = True
    img_loc = os.path.join(self.path, self.construct_img_name(img_num))
    try:
      image = cv.cvtColor(cv.imread(img_loc, cv.IMREAD_COLOR), cv.COLOR_BGR2RGB)
    except:
      img_loc = os.path.join(self.train_dir, self.construct_img_name(img_num))
      image = cv.cvtColor(cv.imread(img_loc, cv.IMREAD_COLOR), cv.COLOR_BGR2RGB)
    packet = image
    # for i in range(4, -1, -1):
    #   img_loc = os.path.join(self.path, self.construct_img_name(img_num - i))
    #   try:
    #     image = cv.cvtColor(cv.imread(img_loc, cv.IMREAD_COLOR), cv.COLOR_BGR2RGB)
    #   except:
    #     img_loc = os.path.join(self.train_dir, self.construct_img_name(img_num))
    #     image = cv.cvtColor(cv.imread(img_loc, cv.IMREAD_COLOR), cv.COLOR_BGR2RGB)
    #   if firstImg:
    #     packet = image
    #     firstImg = False
    #   else:
    #     packet = np.concatenate((packet,image),axis=2)
    tensor = self.transform(packet)
    return tensor

  def construct_img_name(self, img_num):
    img_num = img_num if img_num >= 0 else 0
    a = str(img_num)
    c = a.rjust(5, "0")
    return "%s.jpg" % c

class FramePacket(Dataset):
  def __init__(self, train_dir, gt_dir, transform, packet_len):
    self.train_dir = train_dir
    self.gt_dir = gt_dir
    self.transform = transform
    self.total_train_imgs = os.listdir(train_dir)
    self.total_gt_imgs = os.listdir(gt_dir)
    self.packet_len = packet_len

  def __len__(self):
    return len(self.total_train_imgs)

  def __getitem__(self, idx):
    augment = A.augmentations.transforms.GaussNoise(p=0.3,var_limit=(0,128))
    gt_img_info = (self.total_gt_imgs[idx])[:-4].split("_")
    img_number = int(gt_img_info[0])
    patch_number = gt_img_info[2]
    gt_loc = os.path.join(self.gt_dir, self.total_gt_imgs[idx])
    gt_tensor = cv.cvtColor(cv.imread(gt_loc, cv.IMREAD_COLOR), cv.COLOR_BGR2RGB)
    gt_tensor = augment(image=gt_tensor)
    gt_tensor = self.transform(gt_tensor['image'])
    train_packet = None
    firstImg = True
    train_loc = os.path.join(self.train_dir, self.construct_img_name(img_number, patch_number))
    try:
      train_image = cv.cvtColor(cv.imread(train_loc, cv.IMREAD_COLOR), cv.COLOR_BGR2RGB)
    except:
      train_loc = os.path.join(self.train_dir, self.construct_img_name(img_number, patch_number))
      train_image = cv.cvtColor(cv.imread(train_loc, cv.IMREAD_COLOR), cv.COLOR_BGR2RGB)
    train_packet = train_image
    # for i in range(4, -1, -1):
    #   train_loc = os.path.join(self.train_dir, self.construct_img_name(img_number - i, patch_number))
    #   try:
    #     train_image = cv.cvtColor(cv.imread(train_loc, cv.IMREAD_COLOR), cv.COLOR_BGR2RGB)
    #   except:
    #     train_loc = os.path.join(self.train_dir, self.construct_img_name(img_number, patch_number))
    #     train_image = cv.cvtColor(cv.imread(train_loc, cv.IMREAD_COLOR), cv.COLOR_BGR2RGB)
    #   if firstImg:
    #     train_packet = train_image
    #     firstImg = False
    #   else:
    #     train_packet = np.concatenate((train_packet,train_image),axis=2)
    train_packet = augment(image=train_packet)
    train_tensor = self.transform(train_packet['image'])
    return train_tensor, gt_tensor

  def construct_img_name(self, img_num, patch_number):
    # if we are one of the first four preceding frames
    # just repeat the first frame in place of the missing frame
    img_num = img_num if img_num >= 0 else 0
    a = str(img_num)
    c = a.rjust(5, "0")
    return "%s_patch_%s.jpg" % (c,patch_number)
  