import os
import torch
import random
import numpy as np
import argparse
from model.deblurnet import DeblurNet
from data import FramePacket
from torchvision import transforms
import sys
import time

def parseArgs():
  parser = argparse.ArgumentParser()

  parser.add_argument("--batch_size", type=int, default=64, help="size of batch from training data")
  parser.add_argument("--max_iters", type=int, default=80000, help="max iterations")
  parser.add_argument("--trainset_dir", type=str, default=None, help="Directory for training data")
  parser.add_argument("--testset_dir", type=str, default=None, help="Directory for testing data")
  parser.add_argument("--validset_dir", type=str, default=None, help="Directory for validation data")
  parser.add_argument("--output_dir", type=str, default=None, help="directory to output testing results")
  parser.add_argument("--use_cuda", type=bool, default=False, help="does this machine support cuda?")
  parser.add_argument("--average_loss_every", type=int, default=20, help="average the loss every x iterations")

  return parser.parse_args()

def loadModel(args):
  return DeblurNet().cuda if args.use_cuda else DeblurNet()

def train(args):
  ##############################################################
  # define training parameters
  max_iters = args.max_iters
  batch_size = args.batch_size
  use_cuda = args.use_cuda
  decayRate = 0.5
  decayEvery = 8000
  decayStart = 24000
  lrMin = 10e-6
  lr = 0.005
  log_every = args.average_loss_every
  print("batch size:            %d" % batch_size)
  print("max iterations:        %d" % max_iters)
  print("use cuda:              %s" % use_cuda)
  print("learning rate start:   %f" % lr)
  print("learning rate minimum: %f\n" % lrMin)
  ###############################################################
  # set up the model and the optimizer
  print("Loading Model...\n")
  deblurNet = loadModel(args)
  optimizer = torch.optim.Adam(deblurNet.parameters(),lr=lr)
  criterion = torch.nn.MSELoss()
  trainDir = os.path.join(os.path.dirname(os.path.abspath(__file__)),args.trainset_dir)
  trainDirs, trainNames = scanSetFolder(trainDir)
  it = 0
  tic = time.time()
  avgSpeed = None
  print("Begin Training...\n")
  total_loss = 0
  avg_loss = 0
  while it < max_iters:
    # check to see if the learning rate needs to be updated
    if it >= decayStart and ((it - decayStart) % decayEvery == 0):
      lr = lr * decayRate
      if lr < lrMin:
        lr = lrMin
      # now update optimizer
      for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    ################################################
    ## Choose Directory and then train on all frames
    batchFrames = loadRandomBatch(trainDirs, 5)
    frameLoader = torch.utils.data.DataLoader(batchFrames, batch_size=batch_size)
    deblurNet.train()
    j = 0
    for idx, (trainPacket, gtImg) in enumerate(frameLoader):
      if use_cuda:
        trainPacket, gtImg = trainPacket.cuda(), gtImg.cuda()
      optimizer.zero_grad()
      output = deblurNet(trainPacket)
      loss = criterion(output, gtImg)
      loss.backward()
      optimizer.step()
      # data collection
      total_loss += loss
      avg_loss += loss
      if j > 9:
        break
      j += 1
    ####### training for that directory should be finished
    #######################################################
    it += 1
    toc = time.time()
    speed = 1/(toc-tic) # iterations per second
    avgSpeed = avgSpeed or speed
    avgSpeed = 0.1 * speed + 0.9 * avgSpeed
    timeLeft = (max_iters - it) / avgSpeed # in seconds
    running_loss = total_loss / it
    toc = tic
    tic = time.time()
    timeLeftMins = timeLeft / 60
    print("iteration:            %d" % it)
    print("loss:                 %f" % loss)
    print("running loss average: %f" % running_loss)
    if avg_loss % log_every == 0:
      avg_loss = avg_loss / log_every
      print("average loss:         %f" % avg_loss)
      avg_loss = 0
    print("learning rate:        %f" % lr)
    print("speed:                %.2f it/s" % speed)
    print("average speed:        %.2f" % avgSpeed)
    print("Time left:            %d hr %.1f min" % (timeLeftMins / 60, timeLeftMins % 60))

def loadRandomBatch(dirs, packet_size):
  transform = transforms.Compose([transforms.ToTensor()])
  randDir = random.choice(dirs)
  trainDir = os.path.join(randDir,"input_patches")
  gtDir = os.path.join(randDir,"GT_patches")
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
