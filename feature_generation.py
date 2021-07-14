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


#n_rotations was changed by n_transformation to be more generic.

@click.command()
@click.option('--dataset', default="CEDAR", help='dataset from which features will be generated')
@click.option('--grid_based', is_flag=True, help="deciding if features will be calculated globally or by grids (0 meand no grid, otherwise number of grids")
@click.option('--rotations', is_flag=True, help="calculate features over rotations dataset version")
@click.option('--scales', is_flag=True, help="calculate features over dataset with scalement")
@click.option('--rotation_scales', is_flag=True, help="calculate features over dataset with rotations and scalement")
@click.option('--trim', is_flag=True, help="calculate features over trimmed dataset (only valid for Bengali and Hindi datasets)")
@click.option('--biased', is_flag=True, help="calculate features over original dataset")
@click.option('--n_procs', default=-1, help="number of processors to be used in parallel, -1 means using all available processors")
@click.option('--output_pickle_file', default="test.pk", help="filename of pickle file generated containing features dataframe")
@click.option('--error_log', default="feature_generation.log", help="logfile for feature generation")
@click.option('--n_transformations', default=1, help="number of rotations being considered")
@click.option('--mls_type', default="skeleton_points", help="indicating type of MLS to use: skeleton_points or all_box, default=skeleton_points")
@click.option('--weights_type', default="EUCL_COLOR", help="indicating weighting type for MLS: EUCL, EUCL_COLOR, EUCL_COLOR_SIGMA, default=EUCL_COLOR")
@click.option('--binarization_type', default="GLOBAL_BINARY_INV", help="type of threshold used for image binarization")
def main(dataset, grid_based, rotations, scales, rotation_scales, trim, 
			biased, n_procs, output_pickle_file, n_transformations, error_log, mls_type, weights_type, binarization_type):
	
	generate_features_r2_histogram(dataset, output_pickle_file, grid_based, rotations, scales, rotation_scales, 
									trim, biased, error_log, n_transformations, n_procs, mls_type, weights_type, binarization_type)

if __name__ == "__main__":
	main()