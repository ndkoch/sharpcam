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
  print("Loading Model...\n")
  model = loadModel(args)
  max_iters = args.max_iters
  batch_size = args.batch_size
  use_cuda = args.use_cuda
  decayRate = 0.5
  decayEvery = max_iters / 10
  decayStart = decayEvery * 3
  lrMin = 10e-6
  lr = 0.005
  log_every = args.average_loss_every
  trainset_dir = args.trainset_dir
  validset_dir = args.validset_dir
  output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output_dir)
  validate_every = args.validate_every
  loss_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.losses_dir)
  if not os.path.exists(loss_dir):
    os.makedirs(loss_dir)
  losses_file = open(os.path.join(loss_dir,"losses.txt"), 'w+')
  valid_losses_file = open(os.path.join(loss_dir,"validation_losses.txt"), 'w+')
  losses_file.truncate(0)
  valid_losses_file.truncate(0)
  print("batch size:            %d" % batch_size)
  print("max iterations:        %d" % max_iters)
  print("use cuda:              %s" % use_cuda)
  print("learning rate start:   %f" % lr)
  print("learning rate minimum: %f\n" % lrMin)

  optimizer = torch.optim.Adam(model.parameters(),lr=lr)
  criterion = torch.nn.MSELoss()
  it = 1
  avg_train_speed = None
  total_train_loss = 0
  total_valid_loss = 0
  avg_train_loss = 0
  tic = time.time()
  print("Begin Training...\n")
  while it <= max_iters:
    # check to see if the learning rate needs to be updated
    if it >= decayStart and ((it - decayStart) % decayEvery == 0):
      lr = lr * decayRate
      if lr < lrMin:
        lr = lrMin
      for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    training_loss = train(model, criterion, optimizer, batch_size, use_cuda, trainset_dir)

    toc = time.time()
    train_speed = 1/(toc-tic) # iterations per second
    avg_train_speed = avg_train_speed or train_speed
    avg_train_speed = 0.1 * train_speed + 0.9 * avg_train_speed
    timeLeft = (max_iters - it) / avg_train_speed # in seconds
    total_train_loss += training_loss
    avg_train_loss += training_loss
    running_train_loss = total_train_loss / it
    timeLeftMins = timeLeft / 60
    losses_file.write("%f\n" % training_loss)
    print("iteration:                                  %d" % it)
    print("training loss:                              %f" % training_loss)
    print("running training loss average:              %f" % running_train_loss)
    if it % log_every == 0:
      avg_train_loss = avg_train_loss / log_every
      print("average training loss (every %d):           %f" % (log_every,avg_train_loss))
      avg_train_loss = 0
    print("learning rate:                              %f" % lr)
    print("speed:                                      %.2f it/s" % train_speed)
    print("average speed:                              %.2f" % avg_train_speed)
    print("Time left:                                  %d hr %.1f min\n" % (timeLeftMins / 60, timeLeftMins % 60))

    if it % validate_every == 0:
      validation_loss = test(model, criterion, batch_size, use_cuda, validset_dir)
      valid_losses_file.write("%f\n" % validation_loss)
      total_valid_loss += validation_loss
      n = it / validate_every
      print("validation loss:                            %f" % validation_loss)
      print("running valid loss:                         %f\n" % (total_valid_loss / n))
    toc = tic
    tic = time.time()
    it += 1
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  output_dir = os.path.join(output_dir, "%s.pt" % args.model_save_name)
  torch.save(model.state_dict(),output_dir)
  losses_file.close()
  valid_losses_file.close()

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
  parser.add_argument("--validate_every", type=int, default=1000, help="run a validation batch every x iterations")
  parser.add_argument("--losses_dir", type=str, default="lossses/", help="directory for saving loss values")
  parser.add_argument("--model_save_name", type=str, default="deblurnet_state_dict", help="name for model state dict")

  return parser.parse_args()

def loadModel(args):
  return DeblurNet().cuda() if args.use_cuda else DeblurNet()

def train(model, criterion, optimizer, batch_size, use_cuda, trainset_dir):
  trainDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), trainset_dir)
  trainDirs, trainNames = scanSetFolder(trainDir)
  batchFrames = loadRandomBatch(trainDirs, 5)
  frameLoader = torch.utils.data.DataLoader(batchFrames, batch_size=batch_size)
  model.train()
  trainPacket, gtImg = next(iter(frameLoader))
  if use_cuda:
    trainPacket, gtImg = trainPacket.cuda(), gtImg.cuda()
  optimizer.zero_grad()
  output = model(trainPacket)
  loss = criterion(output, gtImg)
  loss.backward()
  optimizer.step()
  return loss

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
