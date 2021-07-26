import feature_generation as fg
import classification as clf
import pandas as pd
import click

#@click.command()
#@click.option('--img_ext', default=".png", help="Image extension")
def main():
	'''image_pairs = [["../master-thesis/datasets/CEDAR/full_org_hist_transform/original_1_1.png", 
					"../master-thesis/datasets/CEDAR/full_org_hist_transform/original_1_2.png"],
					["../master-thesis/datasets/CEDAR/full_org_hist_transform/original_1_1.png", 
					"../master-thesis/datasets/CEDAR/full_org_hist_transform/original_1_3.png"]]

	df = pd.DataFrame(image_pairs, columns=['img1', 'img2'])

	df = fg.generate_features(df)'''

	clf.train("CEDAR", "pairs_files/CEDAR_hist_transform_pairs.txt")

if __name__ == "__main__":
	main()