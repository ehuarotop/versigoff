import os

def main():

	print("CEDAR_BIASED")
	os.system("""time python dataset_score.py --pairs_file pairs_files/all_samples/CEDAR_BIASED_pairs.txt \
		--features_file features/all_samples_new/CEDAR_BIASED_features.pk --dataset CEDAR --logfile dataset_scores.log\
		--base_datasets_dir ../master-thesis/datasets/CEDAR/ --clf classifiers/all_samples_random_forgeries/CEDAR_BIASED_clf.pk """)

	print("BENGALI_BIASED")
	os.system("""time python dataset_score.py --pairs_file pairs_files/all_samples/BENGALI_BIASED_pairs.txt \
		--features_file features/all_samples_new/BENGALI_BIASED_features.pk --dataset Bengali --logfile dataset_scores.log\
		--base_datasets_dir ../master-thesis/datasets/BHSig260/Bengali/ --clf classifiers/all_samples_random_forgeries/BENGALI_BIASED_clf.pk """)

	print("HINDI_BIASED")
	os.system("""time python dataset_score.py --pairs_file pairs_files/all_samples/HINDI_BIASED_pairs.txt \
		--features_file features/all_samples_new/HINDI_BIASED_features.pk --dataset Hindi --logfile dataset_scores.log\
		--base_datasets_dir ../master-thesis/datasets/BHSig260/Hindi/ --clf classifiers/all_samples_random_forgeries/HINDI_BIASED_clf.pk """)

	print("CEDAR_UNBIASED")
	os.system("""time python dataset_score.py --pairs_file pairs_files/all_samples/CEDAR_UNBIASED_pairs.txt \
		--features_file features/all_samples_new/CEDAR_UNBIASED_features.pk --dataset CEDAR --logfile dataset_scores.log\
		--base_datasets_dir ../master-thesis/datasets/CEDAR/ --clf classifiers/all_samples_random_forgeries/CEDAR_UNBIASED_clf.pk """)

	print("BENGALI_UNBIASED")
	os.system("""time python dataset_score.py --pairs_file pairs_files/all_samples/BENGALI_UNBIASED_pairs.txt \
		--features_file features/all_samples_new/BENGALI_UNBIASED_features.pk --dataset Bengali --logfile dataset_scores.log\
		--base_datasets_dir ../master-thesis/datasets/BHSig260/Bengali_trimmed/ --clf classifiers/all_samples_random_forgeries/BENGALI_UNBIASED_clf.pk """)

	print("HINDI_UNBIASED")
	os.system("""time python dataset_score.py --pairs_file pairs_files/all_samples/HINDI_UNBIASED_pairs.txt \
		--features_file features/all_samples_new/HINDI_UNBIASED_features.pk --dataset Hindi --logfile dataset_scores.log\
		--base_datasets_dir ../master-thesis/datasets/BHSig260/Hindi_trimmed/ --clf classifiers/all_samples_random_forgeries/HINDI_UNBIASED_clf.pk """)

	print("CEDAR_UNBIASED_ROTATIONS")
	os.system("""time python dataset_score.py --pairs_file pairs_files/all_samples/CEDAR_UNBIASED_ROTATIONS_pairs.txt \
		--features_file features/all_samples_new/CEDAR_UNBIASED_ROTATIONS_features.pk --dataset CEDAR --logfile dataset_scores.log\
		--base_datasets_dir ../master-thesis/datasets/CEDAR/ --clf classifiers/all_samples_random_forgeries/CEDAR_UNBIASED_ROTATIONS_clf.pk """)

	print("BENGALI_UNBIASED_ROTATIONS")
	os.system("""time python dataset_score.py --pairs_file pairs_files/all_samples/BENGALI_UNBIASED_ROTATIONS_pairs.txt \
		--features_file features/all_samples_new/BENGALI_UNBIASED_ROTATIONS_features.pk --dataset Bengali --logfile dataset_scores.log\
		--base_datasets_dir ../master-thesis/datasets/BHSig260/Bengali_trimmed_rotations/ --clf classifiers/all_samples_random_forgeries/BENGALI_UNBIASED_ROTATIONS_clf.pk """)

	print("HINDI_UNBIASED_ROTATIONS")
	os.system("""time python dataset_score.py --pairs_file pairs_files/all_samples/HINDI_UNBIASED_ROTATIONS_pairs.txt \
		--features_file features/all_samples_new/HINDI_UNBIASED_ROTATIONS_features.pk --dataset Hindi --logfile dataset_scores.log\
		--base_datasets_dir ../master-thesis/datasets/BHSig260/Hindi_trimmed_rotations/ --clf classifiers/all_samples_random_forgeries/HINDI_UNBIASED_ROTATIONS_clf.pk """)

	print("CEDAR_UNBIASED_SCALES")
	os.system("""time python dataset_score.py --pairs_file pairs_files/all_samples/CEDAR_UNBIASED_SCALES_pairs.txt \
		--features_file features/all_samples_new/CEDAR_UNBIASED_SCALES_features.pk --dataset CEDAR --logfile dataset_scores.log\
		--base_datasets_dir ../master-thesis/datasets/CEDAR/ --clf classifiers/all_samples_random_forgeries/CEDAR_UNBIASED_SCALES_clf.pk """)

	print("BENGALI_UNBIASED_SCALES")
	os.system("""time python dataset_score.py --pairs_file pairs_files/all_samples/BENGALI_UNBIASED_SCALES_pairs.txt \
		--features_file features/all_samples_new/BENGALI_UNBIASED_SCALES_features.pk --dataset Bengali --logfile dataset_scores.log\
		--base_datasets_dir ../master-thesis/datasets/BHSig260/Bengali_trimmed_scales/ --clf classifiers/all_samples_random_forgeries/BENGALI_UNBIASED_SCALES_clf.pk """)

	print("HINDI_UNBIASED_SCALES")
	os.system("""time python dataset_score.py --pairs_file pairs_files/all_samples/HINDI_UNBIASED_SCALES_pairs.txt \
		--features_file features/all_samples_new/HINDI_UNBIASED_SCALES_features.pk --dataset Hindi --logfile dataset_scores.log\
		--base_datasets_dir ../master-thesis/datasets/BHSig260/Hindi_trimmed_scales/ --clf classifiers/all_samples_random_forgeries/HINDI_UNBIASED_SCALES_clf.pk """)

	print("CEDAR_UNBIASED_ROTATION_SCALES")
	os.system("""time python dataset_score.py --pairs_file pairs_files/all_samples/CEDAR_UNBIASED_ROTATION_SCALES_pairs.txt \
		--features_file features/all_samples_new/CEDAR_UNBIASED_ROTATION_SCALES_features.pk --dataset CEDAR --logfile dataset_scores.log\
		--base_datasets_dir ../master-thesis/datasets/CEDAR/ --clf classifiers/all_samples_random_forgeries/CEDAR_UNBIASED_ROTATION_SCALES_clf.pk """)

	print("BENGALI_UNBIASED_ROTATION_SCALES")
	os.system("""time python dataset_score.py --pairs_file pairs_files/all_samples/BENGALI_UNBIASED_ROTATION_SCALES_pairs.txt \
		--features_file features/all_samples_new/BENGALI_UNBIASED_ROTATION_SCALES_features.pk --dataset Bengali --logfile dataset_scores.log\
		--base_datasets_dir ../master-thesis/datasets/BHSig260/Bengali_trimmed_rotation_scales/ --clf classifiers/all_samples_random_forgeries/BENGALI_UNBIASED_ROTATION_SCALES_clf.pk """)

	print("HINDI_UNBIASED_ROTATION_SCALES")
	os.system("""time python dataset_score.py --pairs_file pairs_files/all_samples/HINDI_UNBIASED_ROTATION_SCALES_pairs.txt \
		--features_file features/all_samples_new/HINDI_UNBIASED_ROTATION_SCALES_features.pk --dataset Hindi --logfile dataset_scores.log\
		--base_datasets_dir ../master-thesis/datasets/BHSig260/Hindi_trimmed_rotation_scales/ --clf classifiers/all_samples_random_forgeries/HINDI_UNBIASED_ROTATION_SCALES_clf.pk """)

if __name__ == "__main__":
	main()