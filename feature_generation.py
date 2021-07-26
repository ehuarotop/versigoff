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

import swifter

#Registering global lock
lock = multiprocessing.Lock()

#Declaring number of bins
n_bins = [6,11,16,21,26]
n_overlaps = [0.05,0.10,0.15,0.20]
n_grids = [1,2,3,4]

def save_dataframe_from_features(features, grid_based, pickle_file):
	#Dataframe columns
	df_columns = ['writer','img1', 'img2', 'img1_height', 'img1_width', 'img1_height_width', 'img1_bbox_height', 'img1_bbox_width', 'img1_bbox_height_width', 
						'img2_height', 'img2_width', 'img2_height_width', 'img2_bbox_height', 'img2_bbox_width', 'img2_bbox_height_width']
	for index, n_bin in enumerate(n_bins):
		if grid_based:
			for overlap in n_overlaps:
				for n_grid in n_grids:
					df_columns = df_columns + ['imghist1_overlap{}_bin{}_grid{}'.format(str(overlap), str(n_bin), str(n_grid)), 'imghist2_overlap{}_bin{}_grid{}'.format(str(overlap),str(n_bin), str(n_grid))]
		else:
			df_columns = df_columns + ['imghist1_{}'.format(str(n_bin)), 'imghist2_{}'.format(str(n_bin))]
	df_columns = df_columns + ['class']

	print("Generating dataframe from features list")
	start_time = time.time()
	#Generating dataframe from features list
	df = pd.DataFrame.from_records(features, columns=df_columns)
	#Sorting the recently generated dataset
	df.sort_values(by=['img1', 'img2'])
	end_time = time.time()
	print("In {0}".format(str(end_time-start_time)))

	print("Saving dataframe to pickle file")
	start_time = time.time()
	#saving features to pickle file
	df.to_pickle(pickle_file)
	#pickle.dump(df, open(pickle_file, "wb"))
	end_time = time.time()
	print("In {0}".format(str(end_time-start_time)))

	print(df)

def process_pair_signatures(gen_sig_dir, forg_sig_dir, gen_signatures, forg_signatures, ix_writer, n_bins, grid_based, mls_type, weights_type, binarization_type, shared_list):
	gen_signatures_skeletons = []

	#Calculating skeletons and other associated information only once (in order to use it after)
	for g_sig in gen_signatures:
		try:
			#Getting histograms for genuine signature image (for each bin)
			g_img_height, g_img_width, g_bbox_height, g_bbox_width, g_sig_histograms = get_r2_histogram(os.path.join(gen_sig_dir, g_sig), n_bins, grid_based, mls_type, weights_type, binarization_type)
			gen_signatures_skeletons.append([g_img_height, g_img_width, g_bbox_height, g_bbox_width, g_sig_histograms])
		except RuntimeError as e:
			gen_signatures_skeletons.append([])
			utils.writeToFile(error_log, "{0}-->{1}".format(os.path.join(gen_sig_dir, g_sig), str(e)))
			pass

	#Generating pairs genuine/forged
	for f_sig in forg_signatures:
		try:
			#Getting histograms for forged signature image (for each bin)
			f_img_height, f_img_width, f_bbox_height, f_bbox_width, f_sig_histograms = get_r2_histogram(os.path.join(forg_sig_dir, f_sig), n_bins, grid_based, mls_type, weights_type, binarization_type)

			#Getting height*width products
			f_img_height_width = f_img_height * f_img_width
			f_bbox_height_width = f_bbox_height * f_bbox_width

			#Iterating over genuine signatures
			for ix in range(len(gen_signatures)):
				if len(gen_signatures_skeletons[ix]) != 0:
					#Getting histograms for genuine signature image (for each bin)
					g_img_height, g_img_width, g_bbox_height, g_bbox_width, g_sig_histograms = gen_signatures_skeletons[ix]

					#Getting height*width product
					g_img_height_width = g_img_height * g_img_width
					g_bbox_height_width = g_bbox_height * g_bbox_width

					#Generating all columns of the dataframe (different pairs for different bins)
					df_row = [ix_writer, gen_signatures[ix], f_sig, g_img_height, g_img_width, g_img_height_width, g_bbox_height, g_bbox_width, g_bbox_height_width,
								f_img_height, f_img_width, f_img_height_width, f_bbox_height, f_bbox_width, f_bbox_height_width]
					for index, n_bin in enumerate(n_bins):
						if grid_based:
							for ix_overlap, overlap in enumerate(n_overlaps):
								for ix_n_grid, n_grid in enumerate(n_grids):
									df_row = df_row + [g_sig_histograms[ix_overlap*len(n_bins) + index][ix_n_grid][0], f_sig_histograms[ix_overlap*len(n_bins) + index][ix_n_grid][0]]
						else:
							df_row = df_row + [g_sig_histograms[index][0], f_sig_histograms[index][0]]
					
					df_row = df_row + [0]

					#Appending df_row to shared_list
					shared_list.append(df_row)

		except RuntimeError as e:
			utils.writeToFile(error_log, "{0}-->{1}".format(os.path.join(forg_sig_dir, f_sig), str(e)))
			pass

	#Generating pairs genuine/genuine
	for ix, g_sig in enumerate(gen_signatures):

		try:

			#Getting histograms for genuine signature image (for each bin)
			g_img_height, g_img_width, g_bbox_height, g_bbox_width, g_sig_histograms = get_r2_histogram(os.path.join(gen_sig_dir, g_sig), n_bins, grid_based, mls_type, weights_type, binarization_type)

			#Getting height*width product
			g_img_height_width = g_img_height * g_img_width
			g_bbox_height_width = g_bbox_height * g_bbox_width

			for i in range(ix+1, len(gen_signatures)):#replacing gen_sig_per_writer by len(gen_signatures) because is correct.
				if len(gen_signatures_skeletons[i]) != 0:

					#Getting histograms for genuine signature image (for each bin)
					g_img_current_height, g_img_current_width, g_img_current_bbox_height, g_img_current_bbox_width, g_sig_histograms_current = gen_signatures_skeletons[i]

					#Getting height*width product
					g_img_current_height_width = g_img_current_height * g_img_current_width
					g_img_current_bbox_height_width = g_img_current_bbox_height * g_img_current_bbox_width

					#Generating all columns of the dataframe
					df_row = [ix_writer, g_sig, gen_signatures[i], g_img_height, g_img_width, g_img_height_width, g_bbox_height, g_bbox_width, g_bbox_height_width,
								g_img_current_height, g_img_current_width, g_img_current_height_width, g_img_current_bbox_height, g_img_current_height_width, g_img_current_bbox_height_width]
					for index, n_bin in enumerate(n_bins):
						if grid_based:
							for ix_overlap, overlap in enumerate(n_overlaps):
								for ix_n_grid, n_grid in enumerate(n_grids):
									df_row = df_row + [g_sig_histograms[ix_overlap*len(n_bins) + index][ix_n_grid][0], g_sig_histograms_current[ix_overlap*len(n_bins) + index][ix_n_grid][0]]
						else:
							df_row = df_row + [g_sig_histograms[index][0], g_sig_histograms_current[index][0]]
					df_row = df_row + [1]

					#Appending df_row to shared_list
					shared_list.append(df_row)

		except RuntimeError as e:
			utils.writeToFile(error_log, "{0}-->{1}".format(os.path.join(forg_sig_dir, f_sig), str(e)))
			pass

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

	### Calculating bounding box and size of his diagonal ###
	# Getting bounding box
	right = round(np.max(new_skeleton[:,0]))
	left = round(np.min(new_skeleton[:,0]))
	width = right - left
	#### Have to be careful here because of the order of the pixels (verify with the image)
	bottom = round(np.max(new_skeleton[:,1]))
	top = round(np.min(new_skeleton[:,1]))
	height = bottom-top

	for n_bin in n_bins:
		histogram = np.histogram(new_skeleton[:,2], bins=np.linspace(0,1,n_bin))
		histograms.append(histogram)	

	return img_height, img_width, height, width, histograms

def process_CEDAR_writers(writers_list, grid_based, rotations, scales, rotation_scales, biased, error_log, n_transformations, mls_type, weights_type, binarization_type, shared_list=[]):
	num_writers = 55
	gen_sig_per_writer = 24
	forg_sig_per_writer = 24

	for ix_writer in writers_list:
		print("Processing writer{}".format(ix_writer))

		if rotations:
			gen_signatures = ["original_{0}_{1}_rotation{2}.png".format(ix_writer,i,j) for i in range(1,gen_sig_per_writer+1) for j in range(1,n_transformations+1)]
			forg_signatures = ["forgeries_{0}_{1}_rotation{2}.png".format(ix_writer,i,j) for i in range(1,forg_sig_per_writer+1) for j in range(1,n_transformations+1)]
			if n_transformations == 1:
				forg_sig_dir = "../../datasets/CEDAR/full_forg_rotations1_new"
				gen_sig_dir = "../../datasets/CEDAR/full_org_rotations_1_new"
			elif n_transformations == 5:
				forg_sig_dir = "../../datasets/CEDAR/full_forg_rotations"
				gen_sig_dir = "../../datasets/CEDAR/full_org_rotations"
		elif scales:
			gen_signatures = ["original_{0}_{1}_scale{2}.png".format(ix_writer,i,j) for i in range(1,gen_sig_per_writer+1) for j in range(1,n_transformations+1)]
			forg_signatures = ["forgeries_{0}_{1}_scale{2}.png".format(ix_writer,i,j) for i in range(1,forg_sig_per_writer+1) for j in range(1,n_transformations+1)]

			if n_transformations == 1:
				forg_sig_dir = "../../datasets/CEDAR/full_forg_scales_1"
				gen_sig_dir = "../../datasets/CEDAR/full_org_scales_1"
		elif rotation_scales:
			gen_signatures = ["original_{0}_{1}_rotation_scale{2}.png".format(ix_writer,i,j) for i in range(1,gen_sig_per_writer+1) for j in range(1,n_transformations+1)]
			forg_signatures = ["forgeries_{0}_{1}_rotation_scale{2}.png".format(ix_writer,i,j) for i in range(1,forg_sig_per_writer+1) for j in range(1,n_transformations+1)]

			if n_transformations == 1:
				forg_sig_dir = "../../datasets/CEDAR/full_forg_rotation_scales_1"
				gen_sig_dir = "../../datasets/CEDAR/full_org_rotation_scales_1"
		elif biased:
			gen_signatures = ["original_{0}_{1}.png".format(ix_writer,i) for i in range(1,gen_sig_per_writer+1)]
			forg_signatures = ["forgeries_{0}_{1}.png".format(ix_writer,i) for i in range(1,forg_sig_per_writer+1)]
			forg_sig_dir = '../../datasets/CEDAR/full_forg'
			gen_sig_dir = '../../datasets/CEDAR/full_org'
		else:
			gen_signatures = ["original_{0}_{1}.png".format(ix_writer,i) for i in range(1,gen_sig_per_writer+1)]
			forg_signatures = ["forgeries_{0}_{1}.png".format(ix_writer,i) for i in range(1,forg_sig_per_writer+1)]
			forg_sig_dir = '../../datasets/CEDAR/full_forg'
			gen_sig_dir = '../../datasets/CEDAR/full_org_hist_transform'

		#Processing pairs of signatures ORIGINAL-ORIGINAL , ORIGINAL-FORGERIE and getting features.
		process_pair_signatures(gen_sig_dir, forg_sig_dir, gen_signatures, forg_signatures, ix_writer, n_bins, grid_based, mls_type, weights_type, binarization_type, shared_list)

def process_Bengali_writers(writers_list, grid_based, rotations, scales, rotation_scales, trim, error_log, n_transformations, mls_type, weights_type, binarization_type, shared_list=[]):
	num_writers = 100
	gen_sig_per_writer = 24
	forg_sig_per_writer = 30

	for ix_writer in writers_list:
		print("Processing writer {}".format(ix_writer))

		'''here rotations, scales, and rotations_scales not implemented yet'''

		if rotations:
			gen_signatures = ["B-S-{ix_writer}-G-{:02d}_trimmed_rotation{}.png".format(i,j, ix_writer=ix_writer) for i in range(1, gen_sig_per_writer+1) for j in range(1,n_transformations+1)]
			forg_signatures = ["B-S-{ix_writer}-F-{:02d}_trimmed_rotation{}.png".format(i,j, ix_writer=ix_writer) for i in range(1, forg_sig_per_writer+1) for j in range(1,n_transformations+1)]

			if n_transformations == 1:
				forg_sig_dir = "../../datasets/BHSig260/Bengali_trimmed_rotations/{:03d}".format(ix_writer)
				gen_sig_dir = "../../datasets/BHSig260/Bengali_trimmed_rotations/{:03d}".format(ix_writer)
		elif trim:
			gen_signatures = ["B-S-{ix_writer}-G-{:02d}_trimmed.png".format(i,ix_writer=ix_writer) for i in range(1, gen_sig_per_writer+1)]
			forg_signatures = ["B-S-{ix_writer}-F-{:02d}_trimmed.png".format(i,ix_writer=ix_writer) for i in range(1, forg_sig_per_writer+1)]

			forg_sig_dir = '../../datasets/BHSig260/Bengali_trimmed/{:03d}'.format(ix_writer)
			gen_sig_dir = '../../datasets/BHSig260/Bengali_trimmed/{:03d}'.format(ix_writer)

		elif scales:
			gen_signatures = ["B-S-{ix_writer}-G-{:02d}_trimmed_scale{}.png".format(i,j, ix_writer=ix_writer) for i in range(1, gen_sig_per_writer+1) for j in range(1,n_transformations+1)]
			forg_signatures = ["B-S-{ix_writer}-F-{:02d}_trimmed_scale{}.png".format(i,j, ix_writer=ix_writer) for i in range(1, forg_sig_per_writer+1) for j in range(1,n_transformations+1)]

			if n_transformations == 1:
				forg_sig_dir = "../../datasets/BHSig260/Bengali_trimmed_scales/{:03d}".format(ix_writer)
				gen_sig_dir = "../../datasets/BHSig260/Bengali_trimmed_scales/{:03d}".format(ix_writer)
		elif rotation_scales:
			gen_signatures = ["B-S-{ix_writer}-G-{:02d}_trimmed_rotation_scale{}.png".format(i,j, ix_writer=ix_writer) for i in range(1, gen_sig_per_writer+1) for j in range(1,n_transformations+1)]
			forg_signatures = ["B-S-{ix_writer}-F-{:02d}_trimmed_rotation_scale{}.png".format(i,j, ix_writer=ix_writer) for i in range(1, forg_sig_per_writer+1) for j in range(1,n_transformations+1)]

			if n_transformations == 1:
				forg_sig_dir = "../../datasets/BHSig260/Bengali_trimmed_rotation_scales/{:03d}".format(ix_writer)
				gen_sig_dir = "../../datasets/BHSig260/Bengali_trimmed_rotation_scales/{:03d}".format(ix_writer)
		else:
			gen_signatures = ["B-S-{ix_writer}-G-{:02d}.tif".format(i, ix_writer=ix_writer) for i in range(1, gen_sig_per_writer+1)]
			forg_signatures = ["B-S-{ix_writer}-F-{:02d}.tif".format(i, ix_writer=ix_writer) for i in range(1, forg_sig_per_writer+1)]
			forg_sig_dir = '../../datasets/BHSig260/Bengali/{:03d}'.format(ix_writer)
			gen_sig_dir = '../../datasets/BHSig260/Bengali/{:03d}'.format(ix_writer)

		print(gen_sig_dir)
		print(forg_sig_dir)

		#Processing pairs of signatures ORIGINAL-ORIGINAL , ORIGINAL-FORGERIE and getting features.
		process_pair_signatures(gen_sig_dir, forg_sig_dir, gen_signatures, forg_signatures, ix_writer, n_bins, grid_based, mls_type, weights_type, binarization_type, shared_list)

def process_Hindi_writers(writers_list, grid_based, rotations, scales, rotation_scales, trim, error_log, n_transformations, mls_type, weights_type, binarization_type, shared_list=[]):
	num_writers = 160
	gen_sig_per_writer = 24
	forg_sig_per_writer = 30

	for ix_writer in writers_list:
		print("Processing writer {}".format(ix_writer))

		'''here rotations, scales, and rotations_scales not implemented yet'''

		if rotations:
			if ix_writer not in [11,17,18,35,76,87,93,123]:
				#Generating list of genuine and forgeries image names
				gen_signatures = ["H-S-{ix_writer}-G-{:02d}_trimmed_rotation{}.png".format(i,j, ix_writer=ix_writer) for i in range(1, gen_sig_per_writer+1) for j in range(1,n_transformations+1)]
				forg_signatures = ["H-S-{ix_writer}-F-{:02d}_trimmed_rotation{}.png".format(i,j, ix_writer=ix_writer) for i in range(1, forg_sig_per_writer+1) for j in range(1,n_transformations+1)]
			elif ix_writer==123 :
				#Generating list of genuine and forgeries image names
				gen_signatures = ["H-S-{ix_writer}-G-{:02d}_trimmed_rotation{}.png".format(i,j, ix_writer=ix_writer+1) for i in range(1, gen_sig_per_writer+1) for j in range(1,n_transformations+1)]
				forg_signatures = ["H-S-{ix_writer}-F-{:02d}_trimmed_rotation{}.png".format(i,j, ix_writer=ix_writer) for i in range(1, forg_sig_per_writer+1) for j in range(1,n_transformations+1)]
			else:
				#Generating list of genuine and forgeries image names
				gen_signatures = ["H-S-{:03d}-G-{:02d}_trimmed_rotation{}.png".format(ix_writer, i,j) for i in range(1, gen_sig_per_writer+1) for j in range(1,n_transformations+1)]
				forg_signatures = ["H-S-{:03d}-F-{:02d}_trimmed_rotation{}.png".format(ix_writer, i,j) for i in range(1, forg_sig_per_writer+1) for j in range(1,n_transformations+1)]

			if n_transformations == 1:
				forg_sig_dir = '../../datasets/BHSig260/Hindi_trimmed_rotations/{:03d}'.format(ix_writer)
				gen_sig_dir = '../../datasets/BHSig260/Hindi_trimmed_rotations/{:03d}'.format(ix_writer)

		elif trim:
			if ix_writer not in [11,17,18,35,76,87,93,123]:
				#Generating list of genuine and forgeries image names
				gen_signatures = ["H-S-{ix_writer}-G-{:02d}_trimmed.png".format(i, ix_writer=ix_writer) for i in range(1, gen_sig_per_writer+1)]
				forg_signatures = ["H-S-{ix_writer}-F-{:02d}_trimmed.png".format(i, ix_writer=ix_writer) for i in range(1, forg_sig_per_writer+1)]
			elif ix_writer==123 :
				#Generating list of genuine and forgeries image names
				gen_signatures = ["H-S-{ix_writer}-G-{:02d}_trimmed.png".format(i, ix_writer=ix_writer+1) for i in range(1, gen_sig_per_writer+1)]
				forg_signatures = ["H-S-{ix_writer}-F-{:02d}_trimmed.png".format(i, ix_writer=ix_writer) for i in range(1, forg_sig_per_writer+1)]
			else:
				#Generating list of genuine and forgeries image names
				gen_signatures = ["H-S-{:03d}-G-{:02d}_trimmed.png".format(ix_writer, i) for i in range(1, gen_sig_per_writer+1)]
				forg_signatures = ["H-S-{:03d}-F-{:02d}_trimmed.png".format(ix_writer,i) for i in range(1, forg_sig_per_writer+1)]

			forg_sig_dir = '../../datasets/BHSig260/Hindi_trimmed/{:03d}'.format(ix_writer)
			gen_sig_dir = '../../datasets/BHSig260/Hindi_trimmed/{:03d}'.format(ix_writer)

		elif scales:
			if ix_writer not in [11,17,18,35,76,87,93,123]:
				#Generating list of genuine and forgeries image names
				gen_signatures = ["H-S-{ix_writer}-G-{:02d}_trimmed_scale{}.png".format(i,j, ix_writer=ix_writer) for i in range(1, gen_sig_per_writer+1) for j in range(1,n_transformations+1)]
				forg_signatures = ["H-S-{ix_writer}-F-{:02d}_trimmed_scale{}.png".format(i,j, ix_writer=ix_writer) for i in range(1, forg_sig_per_writer+1) for j in range(1,n_transformations+1)]
			elif ix_writer==123 :
				#Generating list of genuine and forgeries image names
				gen_signatures = ["H-S-{ix_writer}-G-{:02d}_trimmed_scale{}.png".format(i,j, ix_writer=ix_writer+1) for i in range(1, gen_sig_per_writer+1) for j in range(1,n_transformations+1)]
				forg_signatures = ["H-S-{ix_writer}-F-{:02d}_trimmed_scale{}.png".format(i,j, ix_writer=ix_writer) for i in range(1, forg_sig_per_writer+1) for j in range(1,n_transformations+1)]
			else:
				#Generating list of genuine and forgeries image names
				gen_signatures = ["H-S-{:03d}-G-{:02d}_trimmed_scale{}.png".format(ix_writer, i,j) for i in range(1, gen_sig_per_writer+1) for j in range(1,n_transformations+1)]
				forg_signatures = ["H-S-{:03d}-F-{:02d}_trimmed_scale{}.png".format(ix_writer, i,j) for i in range(1, forg_sig_per_writer+1) for j in range(1,n_transformations+1)]

			if n_transformations == 1:
				forg_sig_dir = '../../datasets/BHSig260/Hindi_trimmed_scales/{:03d}'.format(ix_writer)
				gen_sig_dir = '../../datasets/BHSig260/Hindi_trimmed_scales/{:03d}'.format(ix_writer)
		elif rotation_scales:
			if ix_writer not in [11,17,18,35,76,87,93,123]:
				#Generating list of genuine and forgeries image names
				gen_signatures = ["H-S-{ix_writer}-G-{:02d}_trimmed_rotation_scale{}.png".format(i,j, ix_writer=ix_writer) for i in range(1, gen_sig_per_writer+1) for j in range(1,n_transformations+1)]
				forg_signatures = ["H-S-{ix_writer}-F-{:02d}_trimmed_rotation_scale{}.png".format(i,j, ix_writer=ix_writer) for i in range(1, forg_sig_per_writer+1) for j in range(1,n_transformations+1)]
			elif ix_writer==123 :
				#Generating list of genuine and forgeries image names
				gen_signatures = ["H-S-{ix_writer}-G-{:02d}_trimmed_rotation_scale{}.png".format(i,j, ix_writer=ix_writer+1) for i in range(1, gen_sig_per_writer+1) for j in range(1,n_transformations+1)]
				forg_signatures = ["H-S-{ix_writer}-F-{:02d}_trimmed_rotation_scale{}.png".format(i,j, ix_writer=ix_writer) for i in range(1, forg_sig_per_writer+1) for j in range(1,n_transformations+1)]
			else:
				#Generating list of genuine and forgeries image names
				gen_signatures = ["H-S-{:03d}-G-{:02d}_trimmed_rotation_scale{}.png".format(ix_writer, i,j) for i in range(1, gen_sig_per_writer+1) for j in range(1,n_transformations+1)]
				forg_signatures = ["H-S-{:03d}-F-{:02d}_trimmed_rotation_scale{}.png".format(ix_writer, i,j) for i in range(1, forg_sig_per_writer+1) for j in range(1,n_transformations+1)]

			if n_transformations == 1:
				forg_sig_dir = '../../datasets/BHSig260/Hindi_trimmed_rotation_scales/{:03d}'.format(ix_writer)
				gen_sig_dir = '../../datasets/BHSig260/Hindi_trimmed_rotation_scales/{:03d}'.format(ix_writer)
		else:
			if ix_writer not in [11,17,18,35,76,87,93,123]:
				#Generating list of genuine and forgeries image names
				gen_signatures = ["H-S-{ix_writer}-G-{:02d}.tif".format(i, ix_writer=ix_writer) for i in range(1, gen_sig_per_writer+1)]
				forg_signatures = ["H-S-{ix_writer}-F-{:02d}.tif".format(i, ix_writer=ix_writer) for i in range(1, forg_sig_per_writer+1)]
			elif ix_writer==123 :
				#Generating list of genuine and forgeries image names
				gen_signatures = ["H-S-{ix_writer}-G-{:02d}.tif".format(i, ix_writer=ix_writer+1) for i in range(1, gen_sig_per_writer+1)]
				forg_signatures = ["H-S-{ix_writer}-F-{:02d}.tif".format(i, ix_writer=ix_writer) for i in range(1, forg_sig_per_writer+1)]
			else:
				#Generating list of genuine and forgeries image names
				gen_signatures = ["H-S-{:03d}-G-{:02d}.tif".format(ix_writer, i) for i in range(1, gen_sig_per_writer+1)]
				forg_signatures = ["H-S-{:03d}-F-{:02d}.tif".format(ix_writer, i) for i in range(1, forg_sig_per_writer+1)]

			forg_sig_dir = '../../datasets/BHSig260/Hindi/{:03d}'.format(ix_writer)
			gen_sig_dir = '../../datasets/BHSig260/Hindi/{:03d}'.format(ix_writer)

		#Processing pairs of signatures ORIGINAL-ORIGINAL , ORIGINAL-FORGERIE and getting features.
		process_pair_signatures(gen_sig_dir, forg_sig_dir, gen_signatures, forg_signatures, ix_writer, n_bins, grid_based, mls_type, weights_type, binarization_type, shared_list)

def generate_features_r2_histogram(dataset, pickle_file, grid_based, rotations, scales, rotation_scales, trim, biased, error_log, n_transformations, n_procs, mls_type, weights_type, binarization_type):

	if dataset == "CEDAR":
		num_writers = 55
		gen_sig_per_writer = 24
		forg_sig_per_writer = 24

		print("Generating features")
		start_time = time.time()
		features = utils.execParallel(process_CEDAR_writers, [grid_based, rotations, scales, rotation_scales, biased, error_log, n_transformations, mls_type, weights_type, binarization_type], range(1, num_writers+1), n_procs)
		end_time = time.time()
		print("In {0}".format(str(end_time-start_time)))

		#Saving dataframe from features list just obtained
		save_dataframe_from_features(features, grid_based, pickle_file)

	elif dataset == "Bengali":
		num_writers = 100
		gen_sig_per_writer = 24
		forg_sig_per_writer = 30

		print("Generating features")
		start_time = time.time()
		features = utils.execParallel(process_Bengali_writers, [grid_based, rotations, scales, rotation_scales, trim, error_log, n_transformations, mls_type, weights_type, binarization_type], range(1, num_writers+1), n_procs)
		end_time = time.time()
		print("In {0}".format(str(end_time-start_time)))

		#Saving dataframe from features list just obtained
		save_dataframe_from_features(features, grid_based, pickle_file)

	elif dataset == "Hindi":
		num_writers = 160
		gen_sig_per_writer = 24
		forg_sig_per_writer = 30

		print("Generating features")
		start_time = time.time()
		features = utils.execParallel(process_Hindi_writers, [grid_based, rotations, scales, rotation_scales, trim, error_log, n_transformations, mls_type, weights_type, binarization_type], range(1, num_writers+1), n_procs)
		end_time = time.time()
		print("In {0}".format(str(end_time-start_time)))

		#Saving dataframe from features list just obtained
		save_dataframe_from_features(features, grid_based, pickle_file)


#################### CLIP feature generation ####################
img_crops = []
df_clip_final = []

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

def getImageCrops(img_filename, n_img):
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

	#Normalizing (again) clip_features
	img1_clip_features = img1_clip_features/np.linalg.norm(img1_clip_features)

	### Img2 post processing
	df_filename = df_clip_crops[df_clip_crops['img_filename'] == img2_filename]
	img2_clip_features = np.mean(np.array(df_filename['clip_features'].values.tolist()), axis=0)
	img2_clip_features = img2_clip_features/np.linalg.norm(img2_clip_features)

	#df_clip_final.append([filename, clip_features.tolist()])
	return img1_clip_features.tolist() + img2_clip_features.tolist()

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

	#Getting img crops for img1 and img2 (using swifter to speed apply operation)
	df_clip.apply(lambda x: getImageCrops(x["img1"], 1), axis=1)
	df_clip.apply(lambda x: getImageCrops(x["img2"], 2), axis=1)
	#df_clip.swifter.apply(lambda x: getImageCrops(x["img1"], 1), axis=1)
	#df_clip.swifter.apply(lambda x: getImageCrops(x["img2"], 2), axis=1)

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
	print(df_clip_crops)

	#Getting filenames dataframe from df_clip
	df_clip["clip_features"] = df_clip.apply(lambda x: postProcessingCLIP(x["img1"], x["img2"], df_clip_crops), axis=1)

	return df_clip

#global variables
bins = [6,11,16,21,26]

def generate_handcrafted_features_per_image(filename):
	#Getting r2 histograms
	img_height, img_width, height, width, histograms = get_r2_histogram(filename, bins)
	#Getting actually histograms
	histograms = [x[0] for x in histograms]
	#Concatenating histograms in a single histogram
	histogram = [j for i in histograms for j in i]

	return histogram

def generate_handcrafted_features(df):
	df["img1_handcrafted_features"] = df.apply(lambda x: generate_handcrafted_features_per_image(x["img1"]), axis=1)
	df["img2_handcrafted_features"] = df.apply(lambda x: generate_handcrafted_features_per_image(x["img2"]), axis=1)
	return df

def generate_features(df):
	#For img1
	df = generate_clip_features(df)
	#df = generate_handcrafted_features(df)
	return df