import numpy as np
import torch
from torch.nn import ReLU, Conv2d, ConvTranspose2d, BatchNorm2d, Module, Sigmoid

class DeblurNet(Module):
  def __init__(self):
    super(DeblurNet, self).__init__()
    self.F0  = Conv2d(15,64,5,1,2)
    self.D1  = Conv2d(64,64,3,2,1)
    self.F1  = Conv2d(64,128,3,1,1)
    self.F2  = Conv2d(128,128,3,1,1)
    self.D2  = Conv2d(128,256,3,2,1)
    self.F3  = Conv2d(256,256,3,1,1)
    self.F4  = Conv2d(256,256,3,1,1)
    self.F5  = Conv2d(256,256,3,1,1)
    self.D3  = Conv2d(256,512,3,2,1)
    self.F6  = Conv2d(512,512,3,1,1)
    self.F7  = Conv2d(512,512,3,1,1)
    self.F8  = Conv2d(512,512,3,1,1)
    self.U1  = ConvTranspose2d(512,256,4,2,1)
    self.F9  = Conv2d(256,256,3,1,1)
    self.F10 = Conv2d(256,256,3,1,1)
    self.F11 = Conv2d(256,256,3,1,1)
    self.U2  = ConvTranspose2d(256,128,4,2,1)
    self.F12 = Conv2d(128,128,3,1,1)
    self.F13 = Conv2d(128,128,3,1,1)
    self.U3  = ConvTranspose2d(128,64,4,2,1)
    self.F14 = Conv2d(64,15,3,1,1)
    self.F15 = Conv2d(15,3,3,1,1)
    self.batchnorm3 = BatchNorm2d(3,1e-3)
    self.batchnorm15 = BatchNorm2d(15,1e-3)
    self.batchnorm64 = BatchNorm2d(64,1e-3)
    self.batchnorm128 = BatchNorm2d(128,1e-3)
    self.batchnorm256 = BatchNorm2d(256,1e-3)
    self.batchnorm512 = BatchNorm2d(512,1e-3)
    self.relu = ReLU(True)
    self.sigmoid = Sigmoid()

  
  def forward(self, x):
    inputi = x
    # F0
    F0_out = self.F0(x)
    F0_out = self.batchnorm64(F0_out)
    F0_out = self.relu(F0_out)
    # D1
    D1_out = self.D1(F0_out)
    D1_out = self.batchnorm64(D1_out)
    D1_out = self.relu(D1_out)
    # F1
    F1_out = self.F1(D1_out)
    F1_out = self.batchnorm128(F1_out)
    F1_out = self.relu(F1_out)
    # F2
    F2_out = self.F2(F1_out)
    F2_out = self.batchnorm128(F2_out)
    F2_out = self.relu(F2_out)
    # D2
    D2_out = self.D2(F2_out)
    D2_out = self.batchnorm256(D2_out)
    D2_out = self.relu(D2_out)
    # F3
    F3_out = self.F3(D2_out)
    F3_out = self.batchnorm256(F3_out)
    F3_out = self.relu(F3_out)
    # F4
    F4_out = self.F4(F3_out)
    F4_out = self.batchnorm256(F4_out)
    F4_out = self.relu(F4_out)
    # F5
    F5_out = self.F5(F4_out)
    F5_out = self.batchnorm256(F5_out)
    F5_out = self.relu(F5_out)
    # D3
    D3_out = self.D3(F5_out)
    D3_out = self.batchnorm512(D3_out)
    D3_out = self.relu(D3_out)
    # F6
    F6_out = self.F6(D3_out)
    F6_out = self.batchnorm512(F6_out)
    F6_out = self.relu(F6_out)
    # F7
    F7_out = self.F6(F6_out)
    F7_out = self.batchnorm512(F7_out)
    F7_out = self.relu(F7_out)
    # F8
    F8_out = self.F6(F7_out)
    F8_out = self.batchnorm512(F8_out)
    F8_out = self.relu(F8_out)
    # U1
    U1_out = self.U1(F8_out)
    U1_out = self.batchnorm256(U1_out) + F5_out # Skip Connection 1
    U1_out = self.relu(U1_out)
    # F9
    F9_out = self.F9(U1_out)
    F9_out = self.batchnorm256(F9_out)
    F9_out = self.relu(F9_out)
    # F10
    F10_out = self.F10(F9_out)
    F10_out = self.batchnorm256(F10_out)
    F10_out = self.relu(F10_out)
    # F11
    F11_out = self.F10(F10_out)
    F11_out = self.batchnorm256(F11_out)
    F11_out = self.relu(F11_out)
    # U2
    U2_out = self.U2(F11_out)
    U2_out = self.batchnorm128(U2_out) + F2_out # Skip Connection 2
    U2_out = self.relu(U2_out)
    # F12
    F12_out = self.F10(U2_out)
    F12_out = self.batchnorm128(F12_out)
    F12_out = self.relu(F12_out)
    # F13
    F13_out = self.F13(F12_out)
    F13_out = self.batchnorm64(F13_out)
    F13_out = self.relu(F13_out)
    # U3
    U3_out = self.U3(F13_out)
    U3_out = self.batchnorm64(U3_out) + F0_out # Skip Connection 3
    U3_out = self.relu(U3_out)
    # F14
    F14_out = self.F14(U3_out)
    F14_out = self.batchnorm15(F14_out)
    F14_out = self.relu(F14_out)
    # F15
    F15_out = self.F15(F14_out)
    F15_out = self.batchnorm3(F15_out) + inputi # Skip Connection 4
    F15_out = self.sigmoid(F15_out)
    return F15_out