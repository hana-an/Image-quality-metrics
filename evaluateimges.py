from __future__ import print_function
import os
import argparse
from glob import glob

from PIL import Image
import tensorflow as tf

from utils import *

from math import log10, sqrt
import cv2
import numpy as np

from skimage.metrics import structural_similarity as ssim
import argparse
import imutils


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def SSIM(original,compressed):
    grayA = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(compressed, cv2.COLOR_BGR2GRAY)
    (score, diff) = ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    return score

import lpips
import torchvision.transforms as transforms
from PIL import Image
    

def LPIPS(original,compressed):
    #loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
    loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization
    img0 = cv2.normalize(original, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)                                 #-1 to 1
    img1 = cv2.normalize(compressed, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
    to_tensor = transforms.ToTensor()
    tensor0 = to_tensor(img0)
    tensor1 = to_tensor(img1)
    #tensor0 = tensor0.unsqueeze(0)
    #tensor1 = tensor1.unsqueeze(0)

    #print(np.shape(tensor0))
    #print(np.shape(img11))
    #d = loss_fn_alex(img0, img1)
    d = loss_fn_vgg(tensor0, tensor1)
   
    #print(f"LPIPSvgg value is {m} dB")
    return d
   
highdir = []
testdir = []

#highdir=glob('/content/drive/MyDrive/RetinexNet/data/LOLdataset/eval15/high/*.png')
highdir=glob('/content/drive/MyDrive/RetinexNet/data/syn/high20/*.png')
highdir.sort()
#testdir=glob('/content/drive/MyDrive/Experiments-LIE/LLFlow/results/LOL-pc/000/*.png') #fflow
#testdir=glob('/content/drive/MyDrive/RetinexNet/test_resultsopti1.8/*.png') #ours
#testdir=glob('/content/drive/MyDrive/KinD/res_evaluate/*.png') #kind
#testdir=glob('/content/drive/MyDrive/KinD_plus/test_reseval/*.png')

testdir=glob('/content/drive/MyDrive/RetinexNet/test_results/*.png')
#dslr: dslr/data/res:



testdir.sort()
#print(highdir)
value1=0
v4=0
v5=0
n=len(highdir)
for idx in range(len(highdir)):
  original = cv2.imread(highdir[idx])
  compressed=cv2.imread(testdir[idx])
  value = cv2.PSNR(original, compressed)
  #print(f"PSNR value is {value} dB")

  value2=SSIM(original,compressed)
  #print(f"SSIM value is {value2} dB")

  value3=LPIPS(original,compressed)
  #print(f"LPIPS alex value is {value3} dB")

  value1=value1+value
  avg=value1/n
  v4=v4+value2
  avg2=v4/n
  v5=v5+value3
  avg3=v5/n
print(f"avgpsnr: {avg}")
print(f"avgssim: {avg2}")
print(f"avglpips: {avg3}")