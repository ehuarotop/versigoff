import os
import click

def main():

	print("CEDAR_BIASED")
	os.system("""time python train_model.py --pairs_file pairs_files/all_samples_random_forgeries/CEDAR_BIASED_pairs.txt \
		--features_file CEDAR_BIASED_features.pk --dataset CEDAR --logfile CEDAR_BIASED.log\
		--base_datasets_dir ../master-thesis/datasets/CEDAR/ --clf_name CEDAR_BIASED_clf.pk --save_classifier""")

	print("BENGALI_BIASED")
	os.system("""time python train_model.py --pairs_file pairs_files/all_samples_random_forgeries/BENGALI_BIASED_pairs.txt \
		--features_file BENGALI_BIASED_features.pk --dataset Bengali --logfile BENGALI_BIASED.log\
		--base_datasets_dir ../master-thesis/datasets/BHSig260/Bengali/ --clf_name BENGALI_BIASED_clf.pk --save_classifier""")

	print("HINDI_BIASED")
	os.system("""time python train_model.py --pairs_file pairs_files/all_samples_random_forgeries/HINDI_BIASED_pairs.txt \
		--features_file HINDI_BIASED_features.pk --dataset Hindi --logfile HINDI_BIASED.log\
		--base_datasets_dir ../master-thesis/datasets/BHSig260/Hindi/ --clf_name HINDI_BIASED_clf.pk --save_classifier""")

	print("CEDAR_UNBIASED")
	os.system("""time python train_model.py --pairs_file pairs_files/all_samples_random_forgeries/CEDAR_UNBIASED_pairs.txt \
		--features_file CEDAR_UNBIASED_features.pk --dataset CEDAR --logfile CEDAR_UNBIASED.log\
		--base_datasets_dir ../master-thesis/datasets/CEDAR/ --clf_name CEDAR_UNBIASED_clf.pk --save_classifier""")

	print("BENGALI_UNBIASED")
	os.system("""time python train_model.py --pairs_file pairs_files/all_samples_random_forgeries/BENGALI_UNBIASED_pairs.txt \
		--features_file BENGALI_UNBIASED_features.pk --dataset Bengali --logfile BENGALI_UNBIASED.log\
		--base_datasets_dir ../master-thesis/datasets/BHSig260/Bengali_trimmed/ --clf_name BENGALI_UNBIASED_clf.pk --save_classifier""")

	print("HINDI_UNBIASED")
	os.system("""time python train_model.py --pairs_file pairs_files/all_samples_random_forgeries/HINDI_UNBIASED_pairs.txt \
		--features_file HINDI_UNBIASED_features.pk --dataset Hindi --logfile HINDI_UNBIASED.log\
		--base_datasets_dir ../master-thesis/datasets/BHSig260/Hindi_trimmed/ --clf_name HINDI_UNBIASED_clf.pk --save_classifier""")

	print("CEDAR_UNBIASED_ROTATIONS")
	os.system("""time python train_model.py --pairs_file pairs_files/all_samples_random_forgeries/CEDAR_UNBIASED_ROTATIONS_pairs.txt \
		--features_file CEDAR_UNBIASED_ROTATIONS_features.pk --dataset CEDAR --logfile CEDAR_UNBIASED_ROTATIONS.log\
		--base_datasets_dir ../master-thesis/datasets/CEDAR/ --clf_name CEDAR_UNBIASED_ROTATIONS_clf.pk --save_classifier""")

	print("BENGALI_UNBIASED_ROTATIONS")
	os.system("""time python train_model.py --pairs_file pairs_files/all_samples_random_forgeries/BENGALI_UNBIASED_ROTATIONS_pairs.txt \
		--features_file BENGALI_UNBIASED_ROTATIONS_features.pk --dataset Bengali --logfile BENGALI_UNBIASED_ROTATIONS.log\
		--base_datasets_dir ../master-thesis/datasets/BHSig260/Bengali_trimmed_rotations/ --clf_name BENGALI_UNBIASED_ROTATIONS_clf.pk --save_classifier""")

	print("HINDI_UNBIASED_ROTATIONS")
	os.system("""time python train_model.py --pairs_file pairs_files/all_samples_random_forgeries/HINDI_UNBIASED_ROTATIONS_pairs.txt \
		--features_file HINDI_UNBIASED_ROTATIONS_features.pk --dataset Hindi --logfile HINDI_UNBIASED_ROTATIONS.log\
		--base_datasets_dir ../master-thesis/datasets/BHSig260/Hindi_trimmed_rotations/ --clf_name HINDI_UNBIASED_ROTATIONS_clf.pk --save_classifier""")

	print("CEDAR_UNBIASED_SCALES")
	os.system("""time python train_model.py --pairs_file pairs_files/all_samples_random_forgeries/CEDAR_UNBIASED_SCALES_pairs.txt \
		--features_file CEDAR_UNBIASED_SCALES_features.pk --dataset CEDAR --logfile CEDAR_UNBIASED_SCALES.log\
		--base_datasets_dir ../master-thesis/datasets/CEDAR/ --clf_name CEDAR_UNBIASED_SCALES_clf.pk --save_classifier""")

	print("BENGALI_UNBIASED_SCALES")
	os.system("""time python train_model.py --pairs_file pairs_files/all_samples_random_forgeries/BENGALI_UNBIASED_SCALES_pairs.txt \
		--features_file BENGALI_UNBIASED_SCALES_features.pk --dataset Bengali --logfile BENGALI_UNBIASED_SCALES.log\
		--base_datasets_dir ../master-thesis/datasets/BHSig260/Bengali_trimmed_scales/ --clf_name BENGALI_UNBIASED_SCALES_clf.pk --save_classifier""")

	print("HINDI_UNBIASED_SCALES")
	os.system("""time python train_model.py --pairs_file pairs_files/all_samples_random_forgeries/HINDI_UNBIASED_SCALES_pairs.txt \
		--features_file HINDI_UNBIASED_SCALES_features.pk --dataset Hindi --logfile HINDI_UNBIASED_SCALES.log\
		--base_datasets_dir ../master-thesis/datasets/BHSig260/Hindi_trimmed_scales/ --clf_name HINDI_UNBIASED_SCALES_clf.pk --save_classifier""")

	print("CEDAR_UNBIASED_ROTATION_SCALES")
	os.system("""time python train_model.py --pairs_file pairs_files/all_samples_random_forgeries/CEDAR_UNBIASED_ROTATION_SCALES_pairs.txt \
		--features_file CEDAR_UNBIASED_ROTATION_SCALES_features.pk --dataset CEDAR --logfile CEDAR_UNBIASED_ROTATION_SCALES.log\
		--base_datasets_dir ../master-thesis/datasets/CEDAR/ --clf_name CEDAR_UNBIASED_ROTATION_SCALES_clf.pk --save_classifier""")

	print("BENGALI_UNBIASED_ROTATION_SCALES")
	os.system("""time python train_model.py --pairs_file pairs_files/all_samples_random_forgeries/BENGALI_UNBIASED_ROTATION_SCALES_pairs.txt \
		--features_file BENGALI_UNBIASED_ROTATION_SCALES_features.pk --dataset Bengali --logfile BENGALI_UNBIASED_ROTATION_SCALES.log\
		--base_datasets_dir ../master-thesis/datasets/BHSig260/Bengali_trimmed_rotation_scales/ --clf_name BENGALI_UNBIASED_ROTATION_SCALES_clf.pk --save_classifier""")

	print("HINDI_UNBIASED_ROTATION_SCALES")
	os.system("""time python train_model.py --pairs_file pairs_files/all_samples_random_forgeries/HINDI_UNBIASED_ROTATION_SCALES_pairs.txt \
		--features_file HINDI_UNBIASED_ROTATION_SCALES_features.pk --dataset Hindi --logfile HINDI_UNBIASED_ROTATION_SCALES.log\
		--base_datasets_dir ../master-thesis/datasets/BHSig260/Hindi_trimmed_rotation_scales/ --clf_name HINDI_UNBIASED_ROTATION_SCALES_clf.pk --save_classifier""")

if __name__ == "__main__":
	main()