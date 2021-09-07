### Import useful packages
import argparse
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchvision import transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pims
import pathlib
import torch.optim as optim
from torch.autograd import Variable
import skimage as skm
import glob
from PIL import Image
from datetime import datetime

### Import useful scripts
from datasets import *
from model import *

print('model.py: imported packages.')


### Crop an array
def cropTen(a, b): #a<=b
	sa1, sa2 = len(a), len(a[0])
	sb1, sb2 = len(b), len(b[0])
	d1, d2 = (sb1-sa1)//2, (sb2-sa2)//2
	return b[d1:sb1-d1, d2:sb2-d2]



### Function to load image as pytorch tensor
def image_loader(image_name, loader, device):
	"""load image, returns cuda tensor"""

	## Open image path as PIL image
	image = Image.open(image_name)

	## Load image with torch loader
	image = loader(image).float()

	## Use torch autograd variable
	image = Variable(image, requires_grad=True)

	## Add extra dimension to the tensor
	image = image.unsqueeze(0)  # this is for VGG, may not be needed for ResNet

	## Mount to device and return tensor
	return image.to(device)  # assumes that you're using GPU



### Run UNet model on a folder containing images
def detect_folder(weightPath, imDir, model):

	## Check if runs directory exists
	if len(glob.glob('runs/')) == 0:
		os.mkdir('runs/')
		os.mkdir('runs/detect/')

	## Check if detect directory exists
	elif len(glob.glob('runs/detect/')) == 0:
		os.mkdir('runs/detect/')

	## Number of detect experiments
	expNum = len(glob.glob('runs/detect/*'))

	## Current detect experiment number
	expNum = expNum + 1

	## Experiment directory path
	expPath = 'runs/detect/exp' + str(expNum) + '/'

	## Create experiment directory
	os.mkdir(expPath)

	## Create labels directory
	os.mkdir(expPath + 'labels/')

	## Select device, CPU/GPU
	device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
	print(f"Detecting on device {device}.")

	## Mount model to device
	model.to(device)

	## Load trained weights to model
	model.load_state_dict(torch.load(weightPath))

	print('\nLoaded trained UNet.\n')

	## Choose image input size
	imSize = model.inputDims

	## Create pytorch image loader, will rescale image and crop out the center
	loader = transforms.ToTensor() # transforms.Compose([transforms.Scale(imSize), transforms.CenterCrop((imSize,imSize)), transforms.ToTensor()])

	## Read image paths
	imPaths = glob.glob(imDir + '*')

	## Total number of images
	imTot = len(imPaths)

	## For every image path
	for i, imPath in enumerate(imPaths):

		## Start time
		startTime = datetime.now()

		## Load image as tensor
		testImage = image_loader(imPath, loader, device)

		## Run model on image and get output as numpy array
		rawOut = model(testImage)

		## End time
		endTime = datetime.now()

		## Delta time
		deltaTime = endTime - startTime

		## Convert delta time to seconds
		deltaTime = deltaTime.seconds + (1e-3)*deltaTime.microseconds

		## Display message
		message = 'image ' + str(i+1) + '/' + str(imTot) + ': ' + str(imPath) + ' ' + str(imSize) + 'x' + str(imSize) + ', Done. (' + str(deltaTime) + 's)'
		print(message)

		## Get numpy array from output tensor
		outRay = rawOut[0].to("cpu").detach().numpy()

		## Path to output label
		pathOutLabel = expPath + 'labels/' + imPath.split('\\')[-1].split('.')[-2] + '.txt'

		## Save numpy array as label txt
		np.savetxt(pathOutLabel, outRay)



### Main functioning of script
def main(args):

	## Path to trained model
	weightPath = args.w

	## Path to weights
	imDir = args.src

	## Run detection on folder
	detect_folder(weightPath, imDir, model=CNN())



### Main functioning of script
if __name__ == '__main__':

	## Call new argument parser
	parser = argparse.ArgumentParser()

	## Add weights argument
	parser.add_argument('--w', action='store', nargs='?', type=str, default='runs/train/exp1/weights/best.pth', help='Path to model trained weights (.pth).')

	## Add image directory argument
	parser.add_argument('--src', action='store', nargs='?', type=str, default='data/test/', help='Path to directory containing images for detection.')

	## Parse all arguments
	args = parser.parse_args()

	## Call main with arguments
	main(args)