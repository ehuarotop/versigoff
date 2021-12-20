import feature_generation as fg
from sklearn.base import BaseEstimator
import pandas as pd
import pickle
import click
import utils
import numpy as np
import os

seed = 1337

def cross_dataset_validation(classifier_path, features_file, pairs_file, base_datasets_dir, logfile):

	if "CEDAR" in features_file:
		dataset = "CEDAR"
	elif "BENGALI" in features_file:
		dataset = "Bengali"
	elif "HINDI" in features_file:
		dataset = "Hindi"

	if "CEDAR" in classifier_path:
		classifier = "CEDAR"
	elif "BENGALI" in classifier_path:
		classifier = "Bengali"
	elif "HINDI" in classifier_path:
		classifier = "Hindi"

	#Getting related information to dataset
	num_writers, gen_sig_per_writer, forg_sig_per_writer = utils.get_dataset_info(dataset)

	df = utils.process_pair_file(pairs_file, dataset, base_datasets_dir)
	#Balancing the dataset
	df = utils.balance_dataset(df, seed, num_writers, dataset)
	#Generating features
	df = fg.generate_features(df, None, features_file)
	df = df[["clip_features", "handcrafted_features", "label"]]
	#Getting data for the classification task
	x_data = np.stack([np.concatenate((vec[0], vec[1])) for vec in df.values])
	y_data = df["label"].values

	clf = pickle.load(open(classifier_path, "rb"))
	scores = clf.score(x_data, y_data)

	utils.writeToFile(logfile, "{}, {}, {}\n".format(classifier, dataset, scores))

@click.command()
@click.option('--base_datasets_dir', help="base directory where all datasets are stored")
@click.option('--logfile', help="logfile to store results")
def main(base_datasets_dir, logfile):
	classifier_base_dir = "classifiers/all_samples_new"
	features_base_dir = "features/all_samples_new"
	pairs_base_dir = "pairs_files/all_samples"

	unbiased = [["CEDAR_UNBIASED_clf.pk", "CEDAR_UNBIASED_features.pk", "BENGALI_UNBIASED_features.pk", "HINDI_UNBIASED_features.pk"],
			["BENGALI_UNBIASED_clf.pk", "CEDAR_UNBIASED_features.pk", "BENGALI_UNBIASED_features.pk", "HINDI_UNBIASED_features.pk"],
			["HINDI_UNBIASED_clf.pk", "CEDAR_UNBIASED_features.pk", "BENGALI_UNBIASED_features.pk", "HINDI_UNBIASED_features.pk"]]

	unbiased_rotations = [["CEDAR_UNBIASED_ROTATIONS_clf.pk", "CEDAR_UNBIASED_ROTATIONS_features.pk", "BENGALI_UNBIASED_ROTATIONS_features.pk", "HINDI_UNBIASED_ROTATIONS_features.pk"],
						["BENGALI_UNBIASED_ROTATIONS_clf.pk", "CEDAR_UNBIASED_ROTATIONS_features.pk", "BENGALI_UNBIASED_ROTATIONS_features.pk", "HINDI_UNBIASED_ROTATIONS_features.pk"],
						["HINDI_UNBIASED_ROTATIONS_clf.pk", "CEDAR_UNBIASED_ROTATIONS_features.pk", "BENGALI_UNBIASED_ROTATIONS_features.pk", "HINDI_UNBIASED_ROTATIONS_features.pk"]]

	unbiased_scales = [["CEDAR_UNBIASED_SCALES_clf.pk", "CEDAR_UNBIASED_SCALES_features.pk", "BENGALI_UNBIASED_SCALES_features.pk", "HINDI_UNBIASED_SCALES_features.pk"],
			["BENGALI_UNBIASED_SCALES_clf.pk", "CEDAR_UNBIASED_SCALES_features.pk", "BENGALI_UNBIASED_SCALES_features.pk", "HINDI_UNBIASED_SCALES_features.pk"],
			["HINDI_UNBIASED_SCALES_clf.pk", "CEDAR_UNBIASED_SCALES_features.pk", "BENGALI_UNBIASED_SCALES_features.pk", "HINDI_UNBIASED_SCALES_features.pk"]]

	unbiased_rotation_scales = [["CEDAR_UNBIASED_ROTATION_SCALES_clf.pk", "CEDAR_UNBIASED_ROTATION_SCALES_features.pk", "BENGALI_UNBIASED_ROTATION_SCALES_features.pk", "HINDI_UNBIASED_ROTATION_SCALES_features.pk"],
			["BENGALI_UNBIASED_ROTATION_SCALES_clf.pk", "CEDAR_UNBIASED_ROTATION_SCALES_features.pk", "BENGALI_UNBIASED_ROTATION_SCALES_features.pk", "HINDI_UNBIASED_ROTATION_SCALES_features.pk"],
			["HINDI_UNBIASED_ROTATION_SCALES_clf.pk", "CEDAR_UNBIASED_ROTATION_SCALES_features.pk", "BENGALI_UNBIASED_ROTATION_SCALES_features.pk", "HINDI_UNBIASED_ROTATION_SCALES_features.pk"]]

	tests = [unbiased, unbiased_rotations, unbiased_scales, unbiased_rotation_scales]

	base_datasets_dirs = 	{
								"CEDAR_UNBIASED_features.pk": os.path.join(base_datasets_dir, "CEDAR"),
								"CEDAR_UNBIASED_ROTATIONS_features.pk": os.path.join(base_datasets_dir, "CEDAR"),
								"CEDAR_UNBIASED_SCALES_features.pk": os.path.join(base_datasets_dir, "CEDAR"),
								"CEDAR_UNBIASED_ROTATION_SCALES_features.pk": os.path.join(base_datasets_dir, "CEDAR"),
								"BENGALI_UNBIASED_features.pk": os.path.join(base_datasets_dir, "BHSig260/Bengali_trimmed"),
								"BENGALI_UNBIASED_ROTATIONS_features.pk": os.path.join(base_datasets_dir, "BHSig260/Bengali_trimmed_rotations"),
								"BENGALI_UNBIASED_SCALES_features.pk": os.path.join(base_datasets_dir, "BHSig260/Bengali_trimmed_scales"),
								"BENGALI_UNBIASED_ROTATION_SCALES_features.pk": os.path.join(base_datasets_dir, "BHSig260/Bengali_trimmed_rotation_scales"),
								"HINDI_UNBIASED_features.pk": os.path.join(base_datasets_dir, "BHSig260/Hindi_trimmed"),
								"HINDI_UNBIASED_ROTATIONS_features.pk": os.path.join(base_datasets_dir, "BHSig260/Hindi_trimmed_rotations"),
								"HINDI_UNBIASED_SCALES_features.pk": os.path.join(base_datasets_dir, "BHSig260/Hindi_trimmed_scales"),
								"HINDI_UNBIASED_ROTATION_SCALES_features.pk": os.path.join(base_datasets_dir, "BHSig260/Hindi_trimmed_rotation_scales"),
							}

	for test in tests:
		for data in test:
			classifier_path = os.path.join(classifier_base_dir, data[0])
			for feature in data[1:]:
				features_path = os.path.join(features_base_dir, feature)
				pairs = os.path.join(pairs_base_dir, feature.replace("features.pk", "pairs.txt"))
				cross_dataset_validation(classifier_path, features_path, pairs, base_datasets_dirs[feature], logfile)

if __name__ == "__main__":
	main()