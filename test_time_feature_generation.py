import utils
import numpy as np
import pandas as pd

seed = 1337

def feature_generation_test(dataset, pairs_file, base_datasets_dir, features_file):
    print(dataset)
    ########### Getting features from the pairs_file ###########
    #Getting related information to dataset
    num_writers, gen_sig_per_writer, forg_sig_per_writer = utils.get_dataset_info(dataset)

    #Getting initial dataframe with image pairs, writer and label information
    df = utils.process_pair_file(pairs_file, dataset, base_datasets_dir)
    #Balancing the dataset
    df = utils.balance_dataset(df, seed, num_writers, dataset)
    #If features file not exists then imgs dataframe is generated
    if not os.path.exists(features_file):
        #Getting unique image paths
        imgs = np.unique(df[['img1', 'img2']].values)
        imgs = pd.DataFrame(imgs, columns=["imagepath"])
    else:
        imgs = None

    df = fg.generate_features(df, imgs, features_file)

def main():
    feature_generation_test("CEDAR", "pairs_files/all_samples/CEDAR_UNBIASED_pairs.txt", "../master-thesis/datasets/CEDAR/", "test_CEDAR.pk")
    feature_generation_test("Bengali", "pairs_files/all_samples/BENGALI_UNBIASED_pairs.txt", "../master-thesis/datasets/BHSig260/Bengali_trimmed/", "test_BENGALI.pk")
    feature_generation_test("Hindi", "pairs_files/all_samples/HINDI_UNBIASED_pairs.txt", "../master-thesis/datasets/BHSig260/Hindi_trimmed/", "test_HINDI.pk")

if __name__ == "__main__":
    main()