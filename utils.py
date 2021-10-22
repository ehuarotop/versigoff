import os
import pandas as pd
import time
import sklearn.metrics as sk_metrics
from sklearn.model_selection import cross_validate, KFold
import numpy as np

base_datasets_dir = "../master-thesis/datasets/CEDAR"

def writeToFile(filename, str_text):
    if os.path.exists(filename):
        open_mode = "a"
    else:
        open_mode = "w"

    f = open(filename, open_mode)
    f.writelines(str_text)
    f.close()

def readTxtFile(filename):
    #Reading file recently generated
    f = open(filename)
    lines = f.readlines()

    #Removing \n from each image
    lines = [x[:-1] for x in lines]

    return lines

def get_dataset_info(dataset):
	if dataset == "CEDAR":
		#writers = list(np.arange(1,56))
		num_writers = 55
		gen_sig_per_writer = 24
		forg_sig_per_writer = 24
		dataset = "CEDAR"
	elif dataset == "BENGALI":
		#writers = list(np.arange(1,101))
		num_writers = 100
		gen_sig_per_writer = 24
		forg_sig_per_writer = 30
		dataset = "Bengali"
	elif dataset == "HINDI":
		#writers = list(np.arange(1,161))
		num_writers = 160
		gen_sig_per_writer = 24
		forg_sig_per_writer = 30
		dataset = "Hindi"

	return num_writers, gen_sig_per_writer, forg_sig_per_writer

def process_pair_file(filename, dataset):
	#Getting pairs from txt file and converting it to pd dataframe format
	img_pairs = readTxtFile(filename)
	img_pairs = [x.split(" ") for x in img_pairs]
	df = pd.DataFrame(img_pairs, columns=["img1", "img2", "label"])

	#Fixing directory for img1 and img2
	df["img1"] = df.apply(lambda x: os.path.join(base_datasets_dir, x["img1"]), axis=1)
	df["img2"] = df.apply(lambda x: os.path.join(base_datasets_dir, x["img2"]), axis=1)

	if dataset == "CEDAR":
		df["writer"] = df.apply(lambda x: os.path.basename(x["img1"]).split("_")[1], axis=1)

	return df

def custom_cross_validation(x_data, dataset, n_splits):
	if dataset == "CEDAR":
		# Generating list of 55 writers.
		idx_writers = np.arange(1,56)
	elif dataset == "Bengali":
		# Generating list of 55 writers.
		idx_writers = np.arange(1,101)
	elif dataset == "Hindi":
		# Generating list of 55 writers.
		idx_writers = np.arange(1,161)

	#Splitting writers in training and test dataset
	kf = KFold(n_splits=n_splits, shuffle=True)

	#Iterating over train and test writers for each cross validation
	for train_writers_indexs, test_writers_indexs in kf.split(idx_writers):

		#Converting indexes to integer because of they will be used as writer indexes
		train_writers_indexs = train_writers_indexs.astype(int)
		test_writers_indexs = test_writers_indexs.astype(int)

		#Generating trainging and test data
		training_data_idxs = []
		test_data_idxs = []

		#Iterating over selected writers for training and getting the actual ids associated in x_data
		for idx_writer in train_writers_indexs:
			writer_training_data_idxs = np.where(x_data[:,-1] == idx_writer)[0] #Adding [0] at the end because of np.where returns a tuple
			training_data_idxs = np.concatenate((training_data_idxs, writer_training_data_idxs), axis=0)

		#Iterating over selected writers for testing and getting the actual ids associated in x_data
		for idx_writer in test_writers_indexs:
			writer_test_data_idxs = np.where(x_data[:,-1] == idx_writer)[0] #Adding [0] at the end because of np.where returns a tuple
			test_data_idxs = np.concatenate((test_data_idxs, writer_test_data_idxs), axis=0)

		training_data_idxs = training_data_idxs.astype(int)
		test_data_idxs = test_data_idxs.astype(int)

		#Passing indexes to the cross_validate function
		yield training_data_idxs, test_data_idxs

def perform_cross_validation(classifier, x_data, y_data, logfile="", cv=10):
	start_time = time.time()

	#Performing cross validation
	scoring = 	{ 	'accuracy': sk_metrics.make_scorer(sk_metrics.accuracy_score),
					'f1': sk_metrics.make_scorer(sk_metrics.f1_score),
					'precision': sk_metrics.make_scorer(sk_metrics.precision_score),
					'recall': sk_metrics.make_scorer(sk_metrics.recall_score), 
					'roc_auc': sk_metrics.make_scorer(sk_metrics.roc_auc_score, needs_threshold=True),
					'EER': sk_metrics.make_scorer(compute_EER, needs_threshold=True)
			 	}

	# scores = cross_validate(classifier, x_data, y_data, cv=cv, scoring=scoring)
	scores = cross_validate(classifier, x_data, y_data, cv=cv, scoring=scoring)

	exec_time = time.time() - start_time
	
	str_metrics = "Cross validations scores - CLIP features\n"

	N = np.size(scores['test_accuracy'])
	for i in range(N):
		str_metrics += "Validation {0}      -->  acc={1:.5f},     f1={2:.5f},     prec={3:.5f},   recall={4:.5f},     roc_auc={5:.5f},		EER={6:.5f}\n".format( i+1, 
																	scores['test_accuracy'][i], 
																	scores['test_f1'][i],
																	scores['test_precision'][i],
																	scores['test_recall'][i],
																	scores['test_roc_auc'][i],
																	scores['test_EER'][i])

	#Printing final scores stats after locking the process
	str_metrics += "(Global) acc:       %0.4f (+/- %0.4f)\n" % (scores['test_accuracy'].mean(), scores['test_accuracy'].std())
	str_metrics += "(Global) f1:        %0.4f (+/- %0.4f)\n" % (scores['test_f1'].mean(), scores['test_f1'].std())
	str_metrics += "(Global) prec:      %0.4f (+/- %0.4f)\n" % (scores['test_precision'].mean(), scores['test_precision'].std())
	str_metrics += "(Global) recall:    %0.4f (+/- %0.4f)\n" % (scores['test_recall'].mean(), scores['test_recall'].std())
	str_metrics += "(Global) roc_auc:   %0.4f (+/- %0.4f)\n" % (scores['test_roc_auc'].mean(), scores['test_roc_auc'].std())
	str_metrics += "(Global) EER:   %0.4f (+/- %0.4f)\n" % (scores['test_EER'].mean(), scores['test_EER'].std())

	#Adding execution time to the logs
	str_metrics += "Execution time: {} seconds".format(exec_time)


	##################### Printing metrics information to terminal or saving it to a log file ##################### 
	if logfile == "":
		#just printing the metrics
		print(str_metrics)
	else:
		#Writing string to file
		writeToFile(logfile, str_metrics)

def compute_EER(y_data, predicted_y):
	# Equal Error Rate (EER) is the point where FRR (FNR) curve intersects
	# with the FAR (FPR) curve.

	#Getting true and false positive rates
	fpr, tpr, thresholds = sk_metrics.roc_curve(y_data, predicted_y)

	# Selecting the threshold closest to (FPR = 1 - TPR)
	t_idx = sorted(enumerate(abs(fpr - (1 - tpr))), key=lambda x: x[1])[0][0]
	t = thresholds[t_idx]

	#  true/predict  Positive  Negative
	#  Positive         TP        FN
	#  Negative         FP        TN

	# TPR = TP / (TP+FN)

	EER_1 = fpr[t_idx] # = FAR = FPR
	EER_2 = 1 - tpr[t_idx] # = FRR = FNR
	EER = (EER_1 + EER_2) / 2

	#print(EER_1, EER_2, EER)
	return EER