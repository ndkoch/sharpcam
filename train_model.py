import os
import torch
import random
import numpy as np
import argparse
from model.deblurnet import DeblurNet
from data import FramePacket
from torchvision import transforms

def parseArgs():
  parser = argparse.ArgumentParser()

  # parser.add_argument("--model", type=str, default=None, help="model name")
  parser.add_argument("--batch_size", type=int, default=64, help="size of batch from training data")
  parser.add_argument("--max_iters", type=int, default=80000, help="max iterations")
  parser.add_argument("--trainset_dir", type=str, default=None, help="Directory for training data")
  parser.add_argument("--testset_dir", type=str, default=None, help="Directory for testing data")
  parser.add_argument("--validset_dir", type=str, default=None, help="Directory for validation data")
  parser.add_argument("--output_dir", type=str, default=None, help="directory to output testing results")

  return parser.parse_args()

def loadModel(args):
  return DeblurNet()

def train(args):
  ##############################################################
  # define training parameters
  maxIters = args.max_iters
  batch_size = args.batch_size
  decayRate = 0.5
  decayEvery = 8000
  decayStart = 24000
  lrMin = 10e-6
  lr = 0.005
  ###############################################################
  # set up the model and the optimizer
  deblurNet = loadModel(args)
  optimizer = torch.optim.Adam(deblurNet.parameters(),lr=lr)

  trainDir = os.path.join(os.path.dirname(os.path.abspath(__file__)),args.trainset_dir)
  trainDirs, trainNames = scanSetFolder(trainDir)
  it = 0
  while it < maxIters:
    # check to see if the learning rate needs to be updated
    if it >= decayStart and ((it - decayStart) % decayEvery == 0):
      lr = lr * decayRate
      if lr < lrMin:
        lr = lrMin
      # now update optimizer
      for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    batchFrames = loadRandomBatch(trainDirs, 5)
    frameLoader = torch.utils.data.DataLoader(batchFrames, batch_size=batch_size)
    for idx, (trainPacket, gtImg) in enumerate(frameLoader):
      print(trainPacket.shape)
      print(gtImg.shape)
    it += 1

def loadRandomBatch(dirs, packet_size):
  transform = transforms.Compose([transforms.ToTensor()])
  randDir = random.choice(dirs)
  trainDir = os.path.join(randDir,"input")
  gtDir = os.path.join(randDir,"GT")
  # trainingFrames = FramePacket(trainDir, transform, packet_size)
  # gtFrames = FramePacket(gtDir, transform, packet_size)
  # return trainingFrames, gtFrames
  packet = FramePacket(trainDir, gtDir, transform, packet_size)
  return packet
  

# helper borrowed from
# https://github.com/pidan1231239/DeepVideoDeblurring/blob/cc368f4365cec7762d7700e02466098ec0a62b69/run_model.py
def scanSetFolder(folder):
  names = os.listdir(folder)
  names.sort()
  dirs = [os.path.join(folder, name) for name in names]
  return dirs, names

if __name__ == "__main__":
  args = parseArgs()
  train(args)
