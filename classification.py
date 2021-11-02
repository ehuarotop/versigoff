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

def train(dataset, pairs_file, base_datasets_dir, features_file, save_classifier, logfile):
	########### Getting features from the pairs_file ###########
	#Getting related information to dataset
	num_writers, gen_sig_per_writer, forg_sig_per_writer = utils.get_dataset_info(dataset)

	if not os.path.exists(features_file):
		#Getting initial dataframe with image pairs, writer and label information
		df = utils.process_pair_file(pairs_file, dataset, base_datasets_dir)
		#df = fg.generate_features(df, features_file)
	else:
		df = pickle.load(open(features_file, "rb"))

	#Getting only necessary columns
	#df = df[["clip_features", "handcrafted_features", "writer", "label"]]

	########### Balancing dataset ###########
	np.random.seed((int)(seed))
	random.seed((int)(seed))

	#Getting list of writers
	writers = list(np.arange(1,num_writers+1))

	df_writer_list = []

	#Iterate over writers to get balanced dataset
	for writer1 in writers:
		#Getting df filtered by writer
		df_writer = df[df['writer'] == writer1]

		df_writer_balanced = utils.get_df_writer_balanced(writer1, df_writer, (int)(seed))

		df_writer_list.append(df_writer_balanced)

	#Concatenating list of dataframes just generated
	df = pd.concat(df_writer_list)

	df = fg.generate_features(df, features_file)
	df = df[["clip_features", "handcrafted_features", "writer", "label"]]

	########### Performing training ###########
	#x_data = np.stack([np.concatenate((vec[0], vec[1]/np.linalg.norm(vec[1]), [vec[2]])) for vec in df.values])
	x_data = np.stack([np.concatenate((vec[0], vec[1], [vec[2]])) for vec in df.values])
	y_data = df["label"].values

	#Defining the classifier
	clf = LinearSVC(C=1)

	#Getting custom cross validator
	custom_cv = utils.custom_cross_validation(x_data, dataset, 10)

	x_data = x_data[:,:-1]

	utils.perform_cross_validation(clf, x_data, y_data, logfile, cv=custom_cv)

def predict(df):
	print("Hello world predict")