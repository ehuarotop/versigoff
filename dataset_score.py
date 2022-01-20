import utils
import pickle
import classification
import click
import feature_generation as fg

seed = 1337

def dataset_score(dataset, classifier_path, features_file, pairs_file, base_datasets_dir, logfile):

	"""if "CEDAR" in features_file:
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
		classifier = "Hindi"""

	#Getting related information to dataset
	num_writers, gen_sig_per_writer, forg_sig_per_writer = utils.get_dataset_info(dataset)

	df = utils.process_pair_file(pairs_file, dataset, base_datasets_dir)
	#Balancing the dataset
	df = utils.balance_dataset(df, seed, num_writers, dataset)
	#Generating features
	df = fg.generate_features(df, None, features_file)
	df = df[["clip_features", "handcrafted_features", "label"]]

	clf = pickle.load(open(classifier_path, "rb"))
	scores = classification.score(df, clf)
	
	"""#Getting data for the classification task
	x_data = np.stack([np.concatenate((vec[0], vec[1])) for vec in df.values])
	y_data = df["label"].values

	clf = pickle.load(open(classifier_path, "rb"))
	scores = clf.score(x_data, y_data)"""

	utils.writeToFile(logfile, "{}, {}, {}\n".format(dataset, scores))

@click.command()
@click.option('--dataset', help="which dataset is being evaluated")
@click.option('--clf', help="path for the classifier to use")
@click.option('--pairs_file', help="path for pairs file")
@click.option('--features_file', help="path for features file")
@click.option('--base_datasets_dir', help="base dataset dir")
@click.option('--logfile', help="logfile to save the results")
def main(dataset, clf, pairs_file, features_file, base_datasets_dir, logfile):
	dataset_score(dataset, clf, features_file, pairs_file, base_datasets_dir, logfile)

if __name__ == "__main__":
	main()