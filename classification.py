import utils
import feature_generation as fg
import numpy as np
import random

#Generating random seed
np.random.seed(1337)
random.seed(1337)

def train(dataset, pairs_file):
	#Getting related information to dataset
	num_writers, gen_sig_per_writer, forg_sig_per_writer = utils.get_dataset_info(dataset)

	#Getting initial dataframe with image pairs, writer and label information
	df = utils.process_pair_file(pairs_file, dataset)

	#Getting list of writers
	writers = list(np.arange(1,num_writers+1))
	
	#print(df, df.shape)
	df = fg.generate_features(df)

	print(df)
	print(df.shape)

def predict(df):
	print("Hello world predict")