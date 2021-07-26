import feature_generation as fg
import pandas as pd

def main():
	#generate_features("../master-thesis/datasets/CEDAR/full_org_hist_transform/original_1_1.png")
	#Generating dataframe with filenames to be tested
	#df_clip = pd.DataFrame(["../master-thesis/datasets/CEDAR/full_org_hist_transform/original_1_1.png"], columns=['Filename'])

	image_pairs = [["../master-thesis/datasets/CEDAR/full_org_hist_transform/original_1_1.png", "../master-thesis/datasets/CEDAR/full_org_hist_transform/original_1_2.png"]]

	df = pd.DataFrame(image_pairs, columns=['img1', 'img2'])

	df = fg.generate_features(df)

	#generate_clip_features(df_clip)
	#df = generate_features(df)
	#print(len(df.iloc[0]["clip_features"]))
	#print(len(df.iloc[0]["handcrafted_features"]))

if __name__ == "__main__":
	main()