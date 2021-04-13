import torch
from model.deblurnet import DeblurNet
from data import FramePacketPrediction
from torchvision import transforms
import argparse
import random
import os
from PIL import Image
import numpy as np

def parseArgs():
  parser = argparse.ArgumentParser()
  parser.add_argument("--use_cuda", type=bool, default=False, help="does this machine support cuda?")
  parser.add_argument("--video_directory", type=str, default=None, help="directry for video")
  parser.add_argument("--model_weights", type=str,default=None, help="directory path for model weights (.pt file)")
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
  transform = transforms.Compose([transforms.ToTensor()])
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
  x = frame.cpu().detach().numpy()
  x = x[0]
  x = x[0:3]
  print(x.shape)
  y = model(frame)
  y = y.cpu().detach().numpy()
  y = y[0]
  y = np.ascontiguousarray(y.transpose(1,2,0))
  x = np.ascontiguousarray(x.transpose(1,2,0))
  img = Image.fromarray(y, 'RGB')
  imgx = Image.fromarray(x, 'RGB')
  img.save('t1.png')
  imgx.save('f1.png')

if __name__ == "__main__":
  main()
