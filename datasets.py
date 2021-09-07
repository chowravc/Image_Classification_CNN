### Import useful packages
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
import cv2

print('datasets.py: imported packages.')


### Train dataset object
class TrainDataset(Dataset):
	
	## Constructor
	def __init__(self, dataPath):

		# Get relevant frames
		self.frames = glob.glob(dataPath + '/train/images/*.png')

		# Get label filepaths from frames
		self.names = [frame.replace('images', 'labels').replace('png', 'txt') for frame in self.frames]

		# Function to transform image/array to tensor
		self.to_tensor = transforms.ToTensor()

	## Get length of object frames
	def __len__(self):

		# Return length
		return len(self.frames)

	## Get next timen from datset
	def __getitem__(self, idx):

		# If tensor is inputted, convert it to a list
		if torch.is_tensor(idx):

			# Index conversion to list
			idx.tolist()

		# Read label as a numpy array from txt file and convert it to a tensor
		label = torch.zeros((10))
		label[int(np.loadtxt(self.names[idx]))] = 1

		# Read frame as a pims image and convert to a tensor
		tensor = self.to_tensor(cv2.imread(self.frames[idx])/255).to(dtype=torch.float32)

		# Create list of image tensor and unet mask
		sample = [tensor, label]

		# Return list
		return sample



### Test dataset object
class TestDataset(Dataset):
	
	## Constructor
	def __init__(self, dataPath):

		# Get relevant frames
		self.frames = glob.glob(dataPath + '/test/images/*.png')

		# Get label filepaths from frames
		self.names = [frame.replace('images', 'labels').replace('png', 'txt') for frame in self.frames]

		# Function to transform image/array to tensor
		self.to_tensor = transforms.ToTensor()

	## Get length of object frames
	def __len__(self):

		# Return length
		return len(self.frames)

	## Get next timen from datset
	def __getitem__(self, idx):

		# If tensor is inputted, convert it to a list
		if torch.is_tensor(idx):

			# Index conversion to list
			idx.tolist()

		# Read label as a numpy array from txt file and convert it to a tensor
		label = torch.zeros((10))
		label[int(np.loadtxt(self.names[idx]))] = 1

		# Read frame as a pims image and convert to a tensor
		tensor = self.to_tensor(cv2.imread(self.frames[idx])/255).to(dtype=torch.float32)

		# Create list of image tensor and unet mask
		sample = [tensor, label]

		# Return list
		return sample



### Main functioning of script
if __name__ == '__main__':

	from model import *

	## Setup test train split

	## Decide batch size: This is the number of train image-labels that will be fed at once.
	## Choose the largest one you can without running out of RAM.
	bs = 1

	## Path to dataset
	dataPath = 'data/Galaxy10_DECals/'

	## Instantiate the train and test datasets
	train_dataset, test_dataset = TrainDataset(dataPath), TestDataset(dataPath)

	## Define the train dataset loader. This will be shuffled.
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = bs, shuffle = True)

	## Define the test dataset loader. This will not be shuffled.
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = bs, shuffle = False)

	## Load next image-label batch
	images, labels = next(iter(train_loader))

	print(images.shape)
	
	model = CNN()

	## Mount model to device
	# model

	## Load trained weights to model
	model.load_state_dict(torch.load('runs/train/exp2/weights/best.pth'))

	outs = model(images)

	print(outs)

	print(labels)

