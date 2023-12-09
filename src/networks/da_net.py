"""
This repo is largely based on the code base of "SharinGAN: Combining Synthetic and Real Data for Unsupervised GeometryEstimation"
(https://github.com/koutilya-pnvr/SharinGAN) and heavily borrows from "EndoSLAM" (https://github.com/CapsuleEndoscope/EndoSLAM).

Edited by Anita Rau, a.rau.16@ucl.ac.uk, 2023
"""

import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Discriminator(nn.Module):
	def __init__(self, nout=3, nin=3, last_layer_activation=True, last_layer_sigmoid=False):
		super(Discriminator, self).__init__()
		self.last_layer_activation = last_layer_activation
		self.last_layer_sigmoid = last_layer_sigmoid

		self.net = nn.Sequential(
			nn.Conv2d(nin,32,3,1, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.Conv2d(32,32,3,2, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.Conv2d(32,64,3,1, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.Conv2d(64,64,3,2, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.Conv2d(64,128,3,1, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.Conv2d(128,128,3,2, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.Conv2d(128,256,3,1, padding=1),
			nn.BatchNorm2d(256),
			nn.ReLU(),
			nn.Conv2d(256,256,3,2, padding=1),
			nn.BatchNorm2d(256),
			nn.ReLU(),
			nn.Conv2d(256,512,3,1, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(),
			nn.Conv2d(512,512,3,2, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(),
			nn.Conv2d(512,512,3,1, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(),
			nn.Conv2d(512,512,3,2, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(),
			nn.Conv2d(512,512,3,1, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(),
			nn.Conv2d(512,512,3,2, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(),
			Flatten(),
			nn.Linear(2048,1024), 
			nn.BatchNorm1d(1024),
			nn.ReLU(),
			nn.Linear(1024,512),
			nn.BatchNorm1d(512),
			nn.ReLU(),
			nn.Linear(512,nout),
			)
		self.final_act = nn.LogSoftmax(dim=1)
		self.final_sig = nn.Sigmoid()
	
	def forward(self, image):

		prob = self.net(image)
		if self.last_layer_activation:
			prob = self.final_act(prob)
		elif self.last_layer_sigmoid:
			prob = self.final_sig(prob)
		return prob