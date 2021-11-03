import utils
import feature_generation as fg
import numpy as np
import random
import pickle
import os
import pandas as pd
from sklearn.svm import LinearSVC

#Generating random seed
#np.random.seed(1337)
#random.seed(1337)
seed = 1337

def train(dataset, pairs_file, base_datasets_dir, features_file, save_classifier, clf_name, logfile, cross_val):
	########### Getting features from the pairs_file ###########
	#Getting related information to dataset
	num_writers, gen_sig_per_writer, forg_sig_per_writer = utils.get_dataset_info(dataset)

	if not os.path.exists(features_file):
		#Getting initial dataframe with image pairs, writer and label information
		df = utils.process_pair_file(pairs_file, dataset, base_datasets_dir)
		#Balancing the dataset
		df = utils.balance_dataset(df, seed, num_writers)
		#Generating features
		df = fg.generate_features(df, features_file)
	else:
		df = pickle.load(open(features_file, "rb"))

	#Verifying if dataset is unbalanced
	if df_writer[df_writer["label"] == 0].shape[0] != df_writer[df_writer["label"] == 1].shape[0]:
		df = utils.balance_dataset(df, seed, num_writers)

	#Getting only necessary columns
	df = df[["clip_features", "handcrafted_features", "writer", "label"]]

	########### Performing training ###########
	x_data = np.stack([np.concatenate((vec[0], vec[1], [vec[2]])) for vec in df.values])
	y_data = df["label"].values

	#Defining the classifier
	clf = LinearSVC(C=1)

	if cross_val:
		#Getting custom cross validator
		custom_cv = utils.custom_cross_validation(x_data, dataset, 10)

		x_data = x_data[:,:-1]

		utils.perform_cross_validation(clf, x_data, y_data, logfile, cv=custom_cv)
	else:
		clf.fit(x_data[:,:-1], y_data)
		if save_classifier:
			pickle.dump(clf, open(clf_name, "wb"))

def predict(df, clf):
	#df needs to be a pandas datafaframe containing features in the following format df[["clip_features", "handcrafted_features", "writer", "label"]]
	x_data = np.stack([np.concatenate((vec[0], vec[1], [vec[2]])) for vec in df.values])
	predictions = clf.predict(x_data)

	return predictions