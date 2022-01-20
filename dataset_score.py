import utils
import pickle
import classification
import click
import feature_generation as fg
import os

seed = 1337

def dataset_score(dataset, classifier_path, features_file, pairs_file, base_datasets_dir, logfile):

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

	dataset_transform = os.path.basename(features_file).split("_features.pk")[0]

	utils.writeToFile(logfile, "{}, {}\n".format(dataset_transform, scores))

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