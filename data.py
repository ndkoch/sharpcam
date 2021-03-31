import os
from torch.utils.data import Dataset
import numpy as np
import cv2 as cv

class FramePacket(Dataset):
  def __init__(self, train_dir, gt_dir, transform, packet_len):
    self.train_dir = train_dir
    self.gt_dir = gt_dir
    self.transform = transform
    self.total_train_imgs = os.listdir(train_dir)
    self.total_gt_imgs = os.listdir(gt_dir)
    self.packet_len = packet_len

  def __len__(self):
    return len(self.total_train_imgs) - self.packet_len + 1

  def __getitem__(self, idx):
    gt_loc = os.path.join(self.gt_dir, self.total_gt_imgs[idx])
    gt_tensor = self.transform(cv.cvtColor(cv.imread(gt_loc, cv.IMREAD_COLOR), cv.COLOR_BGR2RGB))
    train_packet = None
    firstImg = True
    for i in range(idx, idx + self.packet_len):
      train_loc = os.path.join(self.train_dir, self.total_train_imgs[i])
      train_image = cv.cvtColor(cv.imread(train_loc, cv.IMREAD_COLOR), cv.COLOR_BGR2RGB)
      if firstImg:
        train_packet = train_image
        firstImg = False
      else:
        train_packet = np.concatenate((train_packet,train_image),axis=2)
    train_tensor = self.transform(train_packet)
    return train_tensor, gt_tensor

