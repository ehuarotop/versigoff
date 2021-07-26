import os
import pandas as pd

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