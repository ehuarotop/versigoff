import os
import cv2 as cv
import skeletonizator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
#import utils
import time
import sys
import click
import ast
from itertools import chain, combinations
import multiprocessing

import mapply

#from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

#For CLIP feature generation
import sys
sys.path.append('./CLIP')
import clip
from tqdm import tqdm
import gc
#Defining needed variables and functions for clip
input_resolution = 224
import torch
torch.cuda.empty_cache()
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, FiveCrop, ToTensor, Normalize, Lambda, GaussianBlur
import torchvision.transforms.functional as tf
import torch.nn as nn
#Defining device (gpu/cpu)
device = "cuda" if torch.cuda.is_available() else "cpu"
from PIL import Image

mapply.init(n_workers=multiprocessing.cpu_count()//2)

#Registering global lock
lock = multiprocessing.Lock()

#Declaring number of bins
n_bins = [6,11,16,21,26]
n_overlaps = [0.05,0.10,0.15,0.20]
n_grids = [1,2,3,4]

#global variables for CLIP feature generation
img_crops = []
img_histograms = {}
df_clip_final = []

def get_r2_histogram(image_path, n_bins):
	histograms = []

	#Reading image
	image = cv.imread(image_path, 0)

	#Saving original image into variable in order to use it later
	original_image = image/255

	#Getting image height and width
	img_height, img_width = image.shape

	#Getting the skeleton
	skeleton, skeleton_image = skeletonizator.getSkeleton(image)

	#Getting R2 image
	new_skeleton, r2_image, error = skeletonizator.MovingLeastSquares(skeleton, skeleton_image, original_image)

	if error:
		print("Image {0} produce errors on MovingLeastSquares".format(image_path))

	for n_bin in n_bins:
		#histogram = np.histogram(new_skeleton[:,2], bins=np.linspace(0,1,n_bin))
		histogram = np.histogram(new_skeleton, bins=np.linspace(0,1,n_bin))
		histograms.append(histogram)	

	#return img_height, img_width, height, width, histograms
	return img_height, img_width, histograms

#################### CLIP feature generation ####################
def load_clip_rn50():
	model, _ = clip.load("RN50", device=device)
	return model.encode_image, clip_normalization()

def to_rgb(image):
	return image.convert("RGB")

init_preprocess_image = Compose(
						[	
						    to_rgb,
							ToTensor(),
						]
					)

def clip_normalization():
	# SRC https://github.com/openai/CLIP/blob/e5347713f46ab8121aa81e610a68ea1d263b91b7/clip/clip.py#L73
	return Normalize(
		(0.48145466, 0.4578275, 0.40821073),
		(0.26862954, 0.26130258, 0.27577711),
	)

def custom_pad(img):
    if img.height > img.width:
        h = round(max(img.height*0.5, img.width))
        pl = np.abs(h-img.width)//2
        pt = 0

        pr = (h-img.width) - pt
        pb = 0

        pads = [pl, pt, pr, pb]
        pads = [pad for pad in pads]

        img = tf.pad(img, pads, 255)
    else:
        h = round(max(img.width*0.5, img.height))
        pt = np.abs(h-img.height)//2
        pl = 0

        pb = (h-img.height) - pt
        pr = 0

        pads = [pl, pt, pr, pb]
        pads = [pad for pad in pads]

        img = tf.pad(img, pads, 255)
    return img

def custom_crop(image):
    image_data = np.asarray(image)
    image_data_bw = image_data
    non_empty_columns = np.where(image_data_bw.mean(axis=0) < 254)[0]
    non_empty_rows = np.where(image_data_bw.mean(axis=1) < 254)[0]
    cropBox = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))
    image_data_new = image_data[cropBox[0]:cropBox[1]+1, cropBox[2]:cropBox[3]+1]
    new_image = Image.fromarray(image_data_new)
    
    return new_image

def getImageCrops(filenames):
	for img_filename in filenames:
		#Getting images currently mapped into img_crops
		imgs = [x[0] for x in img_crops]
		
		#This validation could be improved with genuine.list and forgery.list. 
		#In fact, is not extremely slow (around 40 sec for CEDAR) but could be improved anyway
		if img_filename not in imgs:
			#Getting PIL image
			img = Image.open(img_filename)

			preprocess_image = Compose(
					[
						custom_crop,
					    custom_pad,
					    Resize(input_resolution),
					]
				)

			img = preprocess_image(img)
			width, height = img.size

			preprocess_image = Compose(
					[
						FiveCrop(input_resolution),
					]
				)

			#Transforming image and getting image crop
			imgs = preprocess_image(img)

			if width > height:
				imgs = imgs[2:]
				imgs = (imgs[0], imgs[2], imgs[1])
			else:
				imgs = (imgs[0], imgs[4], imgs[2])

			#return imgs
			for ix, image in enumerate(imgs):
				img_crops.append([img_filename, "imgcrop{}".format(ix+1), image])

def postProcessingCLIP(img1_filename, img2_filename, df_clip_crops):
	#Getting dataframe containing information only about the current filename
	df_filename = df_clip_crops[df_clip_crops['img_filename'] == img1_filename]
	#Getting clip features and converting it to np.array
	img1_clip_features = np.mean(np.array(df_filename['clip_features'].values.tolist()), axis=0)
	#Normalizing (again) clip_features --- CLIP_RENORM
	img1_clip_features = img1_clip_features/np.linalg.norm(img1_clip_features)

	### Img2 post processing
	df_filename = df_clip_crops[df_clip_crops['img_filename'] == img2_filename]
	img2_clip_features = np.mean(np.array(df_filename['clip_features'].values.tolist()), axis=0)
	img2_clip_features = img2_clip_features/np.linalg.norm(img2_clip_features)

	return np.array(img1_clip_features.tolist() + img2_clip_features.tolist())

# Dataset loader
class ImagesDataset(Dataset):
	def __init__(self, df, preprocess, input_resolution):
		super().__init__()
		self.df = df
		self.preprocess = preprocess
		self.empty_image = torch.zeros(3, input_resolution, input_resolution)
		
	def __len__(self):
		return len(self.df)
		
	def __getitem__(self, index):
		row = self.df.iloc[index]
		
		try:
			image = self.preprocess(row['PILImg'])
		except:
			image = self.empty_image
		
		return image

def generate_clip_features(df_clip):
	#Loading model and defining preprocessing pipeline
	model, image_normalization = load_clip_rn50()
	preprocess = Compose([init_preprocess_image, image_normalization])

	#Getting img crops for img1 and img2
	df_clip.apply(lambda x: getImageCrops([x["img1"], x["img2"]]), axis=1)
	#df_clip.apply(lambda x: getImageCrops(x["img2"], 2), axis=1)

	df_clip_crops = pd.DataFrame(img_crops, columns=["img_filename", 'Crop', 'PILImg'])

	ds = ImagesDataset(df_clip_crops, preprocess, input_resolution)

	dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)
	
	# Sample one output from model just to check output_dim
	x = torch.zeros(1, 3, input_resolution, input_resolution, device=device)
	with torch.no_grad():
		x_out = model(x)
	output_dim = x_out.shape[1]
	
	# Features data
	X = np.empty((len(ds), output_dim), dtype=np.float32)
	
	# Begin feature generation
	i = 0
	for images in tqdm(dl):
		n_batch = len(images)

		with torch.no_grad():
			emb_images = model(images.to(device))
			if emb_images.ndim == 4:
				emb_images = emb_images.reshape(n_batch, output_dim, -1).mean(-1)
			emb_images = emb_images.cpu().float().numpy()

		# Save normalized features
		X[i:i+n_batch] = emb_images / np.linalg.norm(emb_images, axis=1, keepdims=True)
		
		i += n_batch
		
	del model, image_normalization, ds, dl
	gc.collect()

	#Assigning features to the pandas dataframe
	df_clip_crops['clip_features'] = X.tolist()
	#print(df_clip_crops)

	#Getting filenames dataframe from df_clip
	df_clip["clip_features"] = df_clip.apply(lambda x: postProcessingCLIP(x["img1"], x["img2"], df_clip_crops), axis=1)

	return df_clip

#global variables
bins = [6,11,16,21,26]

def generate_handcrafted_features_per_image(filenames):
	for filename in filenames:
		#Getting unique filenames (in order to not generate again the histogram)
		if filename not in img_histograms:
			#Getting r2 histograms
			#img_height, img_width, height, width, histograms = get_r2_histogram(filename, bins)
			img_height, img_width, histograms = get_r2_histogram(filename, bins)
			#Getting actually histograms
			histograms = [x[0] for x in histograms]
			#Concatenating histograms in a single hinp.array(stogram)
			histogram = [j for i in histograms for j in i]

			img_histograms[filename] = histogram

	#Conforming handcrafted features as: img1_hist + img2_hist + quadratic_diff (img1_hist, img2_hist) + l2_diff(img1_hist, img2_hist)
	img1_hist = np.array(img_histograms[filenames[0]])
	img2_hist = np.array(img_histograms[filenames[1]])

	#Normalizing image histograms in both cases (part of the handcrafted feature generation pipeline)
	img1_hist = img1_hist/np.linalg.norm(img1_hist)
	img2_hist = img2_hist/np.linalg.norm(img2_hist)

	#Subhistograms flag not necessary because it applied to histograms medians, kl_div and ks_value which finally are not being considered as handcrafted features

	quadratic_difference = np.abs(img1_hist-img2_hist)**2
	l2_difference = np.array([np.linalg.norm(img1_hist-img2_hist)])
	
	#returning histograms for each filename in filenames (is assumed that the length of filenames list is 2)
	#return pd.Series([img_histograms[filenames[0]], img_histograms[filenames[1]]], index=["x","y"])
	return np.concatenate((img1_hist, img2_hist, quadratic_difference, l2_difference))

def generate_handcrafted_features(df):
	df["handcrafted_features"] = df.mapply(lambda x: generate_handcrafted_features_per_image([x["img1"], x["img2"]]), axis=1)
	return df

def generate_features(df, features_file):
	#For img1
	df = generate_clip_features(df)
	print("clip features generated")
	df = generate_handcrafted_features(df)
	print("hand crafted features generated")
	df.to_pickle(features_file)
	return df


#PENDINGS
# Optimize feature generation (img_histograms to be shared with mapply, apparently each worker has its own version of this dictionary control)