from skimage.morphology import skeletonize
from skimage import data
import matplotlib.pyplot as plt
from skimage.util import invert
from skimage.io import imread
import cv2 as cv
import numpy as np
from scipy.optimize import curve_fit

import os
import itertools

def MovingLeastSquares(skeleton, skeleton_image, original_img):
	error = False
	#List to save new skeleton points
	new_skeleton = []

	def f_sigma(x):
		xmin = 0.2
		xmax = 0.8
		if x < xmin:
			return 0.0
		elif x > xmax:
			return 1.0
		else:
			return (x-xmin)/(xmax-xmin)

	def linear_function(x, a, b):
		return a*x + b

	#Defining side length
	L = 10

	new_img = np.zeros(np.shape(skeleton_image), dtype=np.float32)

	for point in skeleton:

		try:
			#if mls_type == "skeleton_points":
			#Getting points inside the square centered on the point
			point_positions_sqr = np.where(	(skeleton[:,0] >= (point[0] - L/2)) & 
											(skeleton[:,0] <= (point[0] + L/2)) &
											(skeleton[:,1] >= (point[1] - L/2)) &
											(skeleton[:,1] <= (point[1] + L/2)) )

			#Getting positions inside the square centered on the point (in the original image)
			y_index = np.shape(original_img)[0] - point[1] #Getting again the value of the pixel in the original image (y_index)

			#Considering only squares with five points or more (VERIFY THIS --- 11/01/2021)
			if np.shape(point_positions_sqr)[1] >= 5:
				#Getting actual skeleton points
				skeleton_points = skeleton[point_positions_sqr]
				
				#Getting x_data and y_data from points
				x_data = skeleton_points[:,0]
				y_data = skeleton_points[:,1]

				#Calculating euclidean_distance
				euclidean_distance = ((x_data - point[0])**2 + (y_data - point[1])**2)**0.5 #All weights_type needs euclidean distance
				
				#Calculating distance from black
				distance_from_black = np.zeros((len(x_data)))
				for index in range(len(x_data)):
					distance_from_black[index] = original_img[original_img.shape[0]-y_data[index]][x_data[index]]

				#If there are points to consider in the calc of new skeleton, #when 0, there are no points, when 1 curve_fit can not work
				if x_data.shape[0] > 1:
					#Calculating sigma weights
					sigma_weights = euclidean_distance * distance_from_black + 1

					#Fitting the curve
					#param_x, param_cov_x = curve_fit(linear_function, x_data, y_data, sigma=euclidean_distance)
					param_x, param_cov_x = curve_fit(linear_function, x_data, y_data, sigma=sigma_weights)

					#param_y, param_cov_y = curve_fit(linear_function, y_data, x_data, sigma=euclidean_distance)
					param_y, param_cov_y = curve_fit(linear_function, y_data, x_data, sigma=sigma_weights)

					swap = False

					if np.abs(param_x[0]) > np.abs(param_y[0]):
						param = param_y
						tmp = y_data
						y_data = x_data
						x_data = tmp
						swap = True
					else:
						param = param_x

					#### Calculating R-squared based on steps described on the link below ####
					#https://stackoverflow.com/questions/19189362/getting-the-r-squared-value-using-curve-fit

					#Getting the residuals (i.e. substracting from y_data the result of evaluate the function on x_data with the optimized parameters)
					residuals = y_data - linear_function(x_data, param[0], param[1])
					residuals = residuals / sigma_weights

					#Getting the residual sum of squares
					ss_res = np.sum(residuals**2)

					#Getting total sum of squares
					weights = 1 / sigma_weights
					weighted_average = np.sum(y_data * weights) / np.sum(weights)
					ss_tot = np.sum(((y_data - weighted_average) / sigma_weights)**2)

					#Getting r_squared
					r_squared = 1 - (ss_res / (ss_tot+10**-6))

					#Replacing color in y, x in the image (r-squared visualization, not went so good)
					new_img[np.shape(new_img)[0]-point[1]][point[0]] = r_squared

					#if r_squared >= 0.5:
					if True:
						#Obtaining intersection points for the current square
						'''if swap:
							intersection_points = calc_intersection_points(point[1] - L/2, point[1] + L/2, point[0] + L/2, point[0]-L/2, param)
						else:
							intersection_points = calc_intersection_points(point[0] - L/2, point[0] + L/2, point[1] + L/2, point[1]-L/2, param)	'''

						#Converting intersection_points to numpy array
						#intersection_points = np.asarray(intersection_points)

						#Appending midpoint to new_skeleton
						#intersection_point.x, intersection_point.y, r_squared value
						#new_skeleton.append([np.mean(intersection_points[:,0]), np.mean(intersection_points[:,1]), r_squared])
						new_skeleton.append(r_squared)

						'''if swap:
							tmp = new_skeleton[-1][1]
							new_skeleton[-1][1] = new_skeleton[-1][0]
							new_skeleton[-1][0] = tmp'''
		except RuntimeError as e:
			error = True
			pass

	#new skeleton contains x and y positions, the third value is the r-squared value associated with each position
	return np.asarray(new_skeleton), new_img, error

def getSkeleton(image):
	#Image binarization and invertion
	ret, transformed_image = cv.threshold(image, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

	#Normalizing image in range from 0 to 1
	transformed_image = transformed_image/255

	#Getting skeleton from binarized and inverted image
	skeleton_image = skeletonize(transformed_image)

	### Getting actual skeleton
	#Getting indexes where skeleton[i] == True (actual skeleton)
	y_indexes, x_indexes = np.where(skeleton_image == True)

	#inverting image around y axis (initial skeleton is inverted)
	y_indexes = np.shape(skeleton_image)[0] - y_indexes

	#Getting skeleton points in numpy array format (nx2 array)
	skeleton = np.vstack((x_indexes, y_indexes)).T

	return skeleton, skeleton_image