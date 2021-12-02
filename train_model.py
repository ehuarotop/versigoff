import feature_generation as fg
import classification as clf
import pandas as pd
import click

@click.command()
#@click.option('--img_ext', default=".png", help="Image extension")
@click.option('--pairs_file', default="", help="file containing signature image pairs")
@click.option('--features_file', default="", help="indicates if already have a features file generated, '' means that the features file needs to be generated")
@click.option('--base_datasets_dir', default="", help="base directory for the dataset being processed")
@click.option('--save_classifier', is_flag=True)
@click.option('--clf_name', default="", help="name to save the classifier")
@click.option('--dataset', default="CEDAR", type=click.Choice(['CEDAR', 'Bengali', 'Hindi', 'MCYT', 'GPDS'], case_sensitive=True), help="Which dataset has to be used for training")
@click.option('--logfile', default="", help="File where training log will be saved")
@click.option('--cross_val', is_flag=True)
def main(pairs_file, features_file, base_datasets_dir, save_classifier, clf_name, dataset, logfile, cross_val):
	clf.train(dataset, pairs_file, base_datasets_dir, features_file, save_classifier, clf_name, logfile, cross_val)

if __name__ == "__main__":
	main()