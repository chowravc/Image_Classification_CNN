### Import useful packages
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

print('model.py: imported packages.')



### Convolution sequence
def conv_seq(in_c, out_c):

	## Create sequential function
	conv = nn.Sequential(

		# Define convolutional layer
		nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),

		# Define activation layer
		nn.Tanh(),

		# Define maxpool layer
		nn.MaxPool2d(2)

	)

	## Return sequential convolution
	return conv



### Class CNN (from Pytorch book)
class CNN(nn.Module):

	## Class constructor
	def __init__(self):

		# Initialize parent object
		super().__init__()

		# Define input image dimensions
		self.inputDims = 256

		# Define first convolutional sequence
		self.conv_seq1 = conv_seq(3, 128)

		# Define second convolutional sequence
		self.conv_seq2 = conv_seq(128, 64)

		# Define third convolutional sequence
		self.conv_seq3 = conv_seq(64, 32)

		# Define fourth convolutional sequence
		self.conv_seq4 = conv_seq(32, 16)

		# Define fifth convolutional sequence
		self.conv_seq5 = conv_seq(16, 8)

		# Define first linear layer
		self.fc1 = nn.Linear(8 * 8 * 8, 32)

		# Define first activation layer
		self.act = nn.Tanh()

		# Define second linear layer
		self.fc2 = nn.Linear(32, 10)

	## Function to generate output from model
	def forward(self, x):

		# Apply first set of layers
		out = self.conv_seq1(x)
		# print(out.shape)

		# Apply second set of layers
		out = self.conv_seq2(out)
		# print(out.shape)

		# Apply third set of layers
		out = self.conv_seq3(out)
		# print(out.shape)

		# Apply fourth set of layers
		out = self.conv_seq4(out)
		# print(out.shape)

		# Apply fifth set of layers
		out = self.conv_seq5(out)
		# print(out.shape)

		# Reshape the input tensor to vector
		out = out.view(-1, 8 * 8 * 8)
		# print(out.shape)

		# Use first linear and third activation layer
		out = self.act(self.fc1(out))
		# print(out.shape)

		# Use the output linear layer
		out = self.fc2(out)
		# print(out.shape)

		# Return the result
		return out



### Main functioning of script
if __name__ == '__main__':
	
	CNNModel = CNN().cuda()

	print(CNNModel)

	imTest = torch.rand((1, 3, 256, 256)).cuda()

	outTensor = CNNModel(imTest)

	print(imTest.shape)
	print(outTensor.shape)