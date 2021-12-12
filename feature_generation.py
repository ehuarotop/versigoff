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
from sklearn.decomposition import PCA

mapply.init(n_workers=multiprocessing.cpu_count()//2)

#Registering global lock
lock = multiprocessing.Lock()

#Declaring number of bins
n_bins = [6,11,16,21,26]
n_overlaps = [0.05,0.10,0.15,0.20]
n_grids = [1,2,3,4]

#global variables for CLIP feature generation
df_clip_crops = None
df_clip = None

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

def getImageCrops(filename):
	print(filename)

	#Getting PIL image
	img = Image.open(filename)

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

	return [imgs[0], imgs[1], imgs[2]]

def postProcessingCLIP(img1_filename):
	print(img1_filename)
	#Getting dataframe containing information only about the current filename
	df_filename = df_clip_crops[df_clip_crops['imagepath'] == img1_filename]
	#Getting clip features and converting it to np.array
	img1_clip_features = np.mean(np.array(df_filename['clip_features'].values.tolist()), axis=0)
	#Normalizing (again) clip_features --- CLIP_RENORM
	img1_clip_features = img1_clip_features/np.linalg.norm(img1_clip_features)

	return np.array(img1_clip_features.tolist())

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

def generate_clip_features(df_images):
	global df_clip_crops

	#Loading model and defining preprocessing pipeline
	model, image_normalization = load_clip_rn50()
	preprocess = Compose([init_preprocess_image, image_normalization])

	#Getting img crops for img1 and img2
	print("Generating image crops")
	df_clip_crops = df_images.copy()
	df_clip_crops["imgcrop1"], df_clip_crops["imgcrop2"], df_clip_crops["imgcrop3"] = df_clip_crops.mapply(lambda x: getImageCrops(x["imagepath"]), axis=1, result_type="expand").T.values
	df_clip_crops = pd.DataFrame(np.concatenate((	df_clip_crops[["imagepath", "imgcrop1"]].assign(Crop=1).values, 
													df_clip_crops[["imagepath", "imgcrop2"]].assign(Crop=2).values, 
													df_clip_crops[["imagepath", "imgcrop3"]].assign(Crop=3).values)), columns=["imagepath", "PILImg", "Crop"])
	print("image crops generated")

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
	#Freeing memory
	X = None
	del X
	gc.collect()

	#Getting filenames dataframe from df_clip
	print("Joining image crops information")
	df_images["clip_features"] = df_images.mapply(lambda x: postProcessingCLIP(x["imagepath"]), axis=1)
	#Freeing memory
	df_clip_crops = None
	del df_clip_crops
	gc.collect()

	return df_images

#global variables
bins = [6,11,16,21,26]

def generate_histograms(filename):
	#Getting r2 histograms
	img_height, img_width, histograms = get_r2_histogram(filename, bins)
	histograms = [x[0] for x in histograms]
	#Concatenating histograms
	histogram = [j for i in histograms for j in i]
	histogram = np.array(histogram)
	#Normalizing final histogram
	histogram = histogram/np.linalg.norm(histogram)

	return histogram

def generate_final_features(row):
	#Getting info from img1 and img2
	img1_row = df_clip.loc[row["img1"],:]
	img2_row = df_clip.loc[row["img2"],:]

	clip_features = np.array(img1_row["clip_features"].tolist() + img2_row["clip_features"].tolist())

	quadratic_difference = np.abs(img1_row["histogram"]-img2_row["histogram"])**2
	l2_difference = np.array([np.linalg.norm(img1_row["histogram"]-img2_row["histogram"])])
	handcrafted_features = np.concatenate((img1_row["histogram"], img2_row["histogram"], quadratic_difference, l2_difference))

	return [clip_features, handcrafted_features]

def generate_features(df, imgs, features_file):
	global df_clip
	
	if imgs is not None:
		#Generating clip features
		print("Generating CLIP features")
		df_clip = generate_clip_features(imgs)

		#Generating histograms (hancrafted features)
		print("Generating histograms (hancrafted features)")
		df_clip["histogram"] = df_clip.mapply(lambda x: generate_histograms(x["imagepath"]), axis=1)
		#Setting index to imagepath
		df_clip = df_clip.set_index("imagepath")
		#Saving dataframe with clip and handcrafted features generated
		df_clip.to_pickle(features_file)
	else:
		df_clip = pickle.load(open(features_file, "rb"))

	#Applying PCA over clip_features
	pca = PCA(n_components=128)
	clip_features = np.array([x for x in df_clip["clip_features"].values])
	clip_features = pca.fit_transform(clip_features)
	df_clip = df_clip.assign(clip_features=clip_features)

	#Generating final features
	print("Generating final features")
	df["clip_features"], df["handcrafted_features"] = df.mapply(lambda x: generate_final_features(x), axis=1, result_type="expand").T.values
	#Freeing memory
	df_clip = None
	del df_clip

	gc.collect()

	return df