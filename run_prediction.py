import torch
from model.deblurnet import DeblurNet
from data import FramePacketPrediction
from torchvision import transforms
import argparse
import random
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def parseArgs():
  parser = argparse.ArgumentParser()
  parser.add_argument("--use_cuda", type=bool, default=False, help="does this machine support cuda?")
  parser.add_argument("--video_directory", type=str, default=None, help="directry for video")
  parser.add_argument("--model_weights", type=str,default=None, help="directory path for model weights (.pt file)")
  parser.add_argument("--patchify", type=bool, default=False, help="Break the input image into patches of 128x128 and then repatch the final image")
  return parser.parse_args()

def loadModel(args):
  path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.model_weights)
  model = DeblurNet().cuda() if args.use_cuda else DeblurNet()
  if args.use_cuda:
    model.load_state_dict(torch.load(path))
  else:
    model.load_state_dict(torch.load(path, map_location='cpu'))
  model.eval()
  return model

def loadRandomVideo(dirs, packet_size):
  transform = transforms.ToTensor()
  randDir = random.choice(dirs)
  randDir = os.path.join(randDir,"input")
  video = FramePacketPrediction(randDir, transform, packet_size)
  return video

def scanSetFolder(folder):
  names = os.listdir(folder)
  names.sort()
  dirs = [os.path.join(folder, name) for name in names]
  return dirs

def main():
  args = parseArgs()
  videos = scanSetFolder(os.path.join(os.path.dirname(os.path.abspath(__file__)), args.video_directory))
  video = loadRandomVideo(videos, 5)
  frameLoader = torch.utils.data.DataLoader(video, batch_size=1)
  model = loadModel(args)
  frame = next(iter(frameLoader))
  if args.use_cuda:
    frame = frame.cuda()
  if args.patchify:
    n = 128
    height = frame.size()[2]
    width = frame.size()[3]
    y = torch.zeros((1,3,height,width))
    for i in range(0,height,n):
      for j in range(0,width,n):
        i_end = min(i+n,height)
        j_end = min(j+n,width)
        patch = frame[:,:,i:i_end,j:j_end]
        y[:,:,i:i_end,j:j_end] = model(patch) # patch[:,0:3,:,:,]
    x = frame[0]
    x = torch.split(x, split_size_or_sections=3, dim=0)
    x = x[0]
    a = transforms.ToPILImage()
    x = a(x)
    x.save('before.png')
    y = y[0]
    y = a(y)
    y.save('after.png')
  else:
    x = frame[0]
    x = torch.split(x, split_size_or_sections=3, dim=0)
    x = x[0]
    a = transforms.ToPILImage()
    x = a(x)
    x.save('before.png')
    y = model(frame)
    y = y[0]
    y = a(y)
    y.save('after.png')

if __name__ == "__main__":
  main()
