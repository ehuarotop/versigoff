import feature_generation as fg
import classification as clf
import pandas as pd
import click

@click.command()
#@click.option('--img_ext', default=".png", help="Image extension")
@click.option('--pairs_file', help="file containing signature image pairs")
@click.option('--features_file', default="", help="indicates if already have a features file generated, '' means that the features file needs to be generated")
@click.option('--base_datasets_dir', help="base directory for the dataset being processed")
@click.option('--save_classifier', is_flag=True)
@click.option('--clf_name', default="", help="name to save the classifier")
@click.option('--dataset', type=click.Choice(['CEDAR', 'Bengali', 'Hindi'], case_sensitive=True), help="Which dataset has to be used for training")
@click.option('--logfile', help="File where training log will be saved")
@click.option('--cross_val', is_flag=True)
def main(pairs_file, features_file, base_datasets_dir, save_classifier, clf_name, dataset, logfile, cross_val):
	'''image_pairs = [["../master-thesis/datasets/CEDAR/full_org_hist_transform/original_1_1.png", 
					"../master-thesis/datasets/CEDAR/full_org_hist_transform/original_1_2.png"],
					["../master-thesis/datasets/CEDAR/full_org_hist_transform/original_1_1.png", 
					"../master-thesis/datasets/CEDAR/full_org_hist_transform/original_1_3.png"],
					["../master-thesis/datasets/CEDAR/full_org_hist_transform/original_1_1.png", 
					"../master-thesis/datasets/CEDAR/full_org_hist_transform/original_1_4.png"],
					["../master-thesis/datasets/CEDAR/full_org_hist_transform/original_1_1.png", 
					"../master-thesis/datasets/CEDAR/full_org_hist_transform/original_1_5.png"],
					["../master-thesis/datasets/CEDAR/full_org_hist_transform/original_1_1.png", 
					"../master-thesis/datasets/CEDAR/full_org_hist_transform/original_1_6.png"]]

	df = pd.DataFrame(image_pairs, columns=['img1', 'img2'])

	df = fg.generate_features(df)

	print(df)'''

	clf.train(dataset, pairs_file, base_datasets_dir, features_file, save_classifier, clf_name, logfile, cross_val)

if __name__ == "__main__":
	main()