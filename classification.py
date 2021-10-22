import utils
import feature_generation as fg
import numpy as np
import random
import pickle
import os
from sklearn.svm import LinearSVC

#Generating random seed
np.random.seed(1337)
random.seed(1337)

def get_writer(filename):
	return (int)(os.path.basename(filename).split("_")[1])

def train(dataset, pairs_file, generate_features=False):
	#Getting related information to dataset
	num_writers, gen_sig_per_writer, forg_sig_per_writer = utils.get_dataset_info(dataset)

	#Getting list of writers
	writers = list(np.arange(1,num_writers+1))

	if generate_features:
		#Getting initial dataframe with image pairs, writer and label information
		df = utils.process_pair_file(pairs_file, dataset)
		
		#print(df, df.shape)
		df = fg.generate_features(df)
	else:
		df = pickle.load(open("CEDAR_UNBIASED_features.pk", "rb"))

	#Getting writer
	df["writer"] = df.apply(lambda x: get_writer(x["img1"]), axis=1)
	df["label"] = df.apply(lambda x: (int)(x["label"]), axis=1)
	df = df[["clip_features", "handcrafted_features", "writer", "label"]]
	print(type(df.iloc[0]["label"]))
	x_data = np.stack([np.concatenate((vec[0], vec[1]/np.linalg.norm(vec[1]), [vec[2]])) for vec in df.values])
	y_data = df["label"].values

	#Defining the classifier
	clf = LinearSVC(C=1)

	#Getting custom cross validator
	custom_cv = utils.custom_cross_validation(x_data, "CEDAR", 10)

	x_data = x_data[:,:-1]

	print(x_data.shape)

	utils.perform_cross_validation(clf, x_data, y_data, "", cv=custom_cv)

def predict(df):
	print("Hello world predict")