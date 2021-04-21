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

def main():
  args = parseArgs()
  batch_size = args.batch_size
  model = loadModel(args)
  use_cuda = args.use_cuda
  validset_dir = args.validset_dir
  loss_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.losses_dir)
  if not os.path.exists(loss_dir):
    os.makedirs(loss_dir)
  valid_losses_file = open(os.path.join(loss_dir,"validation_losses.txt"), 'w+')
  valid_losses_file.truncate(0)
  criterion = torch.nn.MSELoss()
  for i in range(0, args.num_passes):
    print(i)
    validation_loss = test(model, criterion, batch_size, use_cuda, validset_dir)
    valid_losses_file.write("%f\n" % validation_loss)
  valid_losses_file.close()

def parseArgs():
  parser = argparse.ArgumentParser()
  parser.add_argument("--batch_size", type=int, default=64, help="size of batch from training data")
  parser.add_argument("--validset_dir", type=str, default=None, help="Directory for validation data")
  parser.add_argument("--use_cuda", type=bool, default=False, help="does this machine support cuda?")
  parser.add_argument("--losses_dir", type=str, default="losses/", help="directory for saving loss values")
  parser.add_argument("--num_passes", type=int, default=500, help="number of validation data points")

  return parser.parse_args()

def loadModel(args):
  model = DeblurNet().cuda() if args.use_cuda else DeblurNet()
  if args.model_load_dir != None:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.model_load_dir)
    if args.use_cuda:
      model.load_state_dict(torch.load(path))
    else:
      model.load_state_dict(torch.load(path, map_location='cpu'))
  return model

def test(model, criterion, batch_size, use_cuda, testset_dir):
  testDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), testset_dir)
  testDirs, testNames = scanSetFolder(testDir)
  batchFrames = loadRandomBatch(testDirs, 5)
  frameLoader = torch.utils.data.DataLoader(batchFrames, batch_size=batch_size)
  model.eval()
  with torch.no_grad():
    testPacket, gtImg = next(iter(frameLoader))
    if use_cuda:
      testPacket, gtImg = testPacket.cuda(), gtImg.cuda()
    output = model(testPacket)
    loss = criterion(output, gtImg)
    return loss

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
  main()

