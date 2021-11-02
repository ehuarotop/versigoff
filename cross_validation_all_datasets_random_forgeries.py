import os
import click

def main():

	print("CEDAR_BIASED")
	os.system("""time python train_model.py --pairs_file pairs_files/all_samples_random_forgeries/CEDAR_BIASED_pairs.txt \
		--features_file CEDAR_BIASED_features.pk --dataset CEDAR --logfile CEDAR_BIASED.log\
		--base_datasets_dir ../master-thesis/datasets/CEDAR/""")

	print("BENGALI_BIASED")
	os.system("""time python train_model.py --pairs_file pairs_files/all_samples_random_forgeries/BENGALI_BIASED_pairs.txt \
		--features_file BENGALI_BIASED_features.pk --dataset Bengali --logfile BENGALI_BIASED.log\
		--base_datasets_dir ../master-thesis/datasets/BHSig260/Bengali/""")

	print("HINDI_BIASED")
	os.system("""time python train_model.py --pairs_file pairs_files/all_samples_random_forgeries/HINDI_BIASED_pairs.txt \
		--features_file HINDI_BIASED_features.pk --dataset Hindi --logfile HINDI_BIASED.log\
		--base_datasets_dir ../master-thesis/datasets/BHSig260/Hindi/""")

	print("CEDAR_UNBIASED")
	os.system("""time python train_model.py --pairs_file pairs_files/all_samples_random_forgeries/CEDAR_UNBIASED_pairs.txt \
		--features_file CEDAR_UNBIASED_features.pk --dataset CEDAR --logfile CEDAR_UNBIASED.log\
		--base_datasets_dir ../master-thesis/datasets/CEDAR/""")

	print("BENGALI_UNBIASED")
	os.system("""time python train_model.py --pairs_file pairs_files/all_samples_random_forgeries/BENGALI_UNBIASED_pairs.txt \
		--features_file BENGALI_UNBIASED_features.pk --dataset Bengali --logfile BENGALI_UNBIASED.log\
		--base_datasets_dir ../master-thesis/datasets/BHSig260/Bengali_trimmed/""")

	print("HINDI_UNBIASED")
	os.system("""time python train_model.py --pairs_file pairs_files/all_samples_random_forgeries/HINDI_UNBIASED_pairs.txt \
		--features_file HINDI_UNBIASED_features.pk --dataset Hindi --logfile HINDI_UNBIASED.log\
		--base_datasets_dir ../master-thesis/datasets/BHSig260/Hindi_trimmed/""")

	print("CEDAR_UNBIASED_ROTATIONS")
	os.system("""time python train_model.py --pairs_file pairs_files/all_samples_random_forgeries/CEDAR_UNBIASED_ROTATIONS_pairs.txt \
		--features_file CEDAR_UNBIASED_ROTATIONS_features.pk --dataset CEDAR --logfile CEDAR_UNBIASED_ROTATIONS.log\
		--base_datasets_dir ../master-thesis/datasets/CEDAR/""")

	print("BENGALI_UNBIASED_ROTATIONS")
	os.system("""time python train_model.py --pairs_file pairs_files/all_samples_random_forgeries/BENGALI_UNBIASED_ROTATIONS_pairs.txt \
		--features_file BENGALI_UNBIASED_ROTATIONS_features.pk --dataset Bengali --logfile BENGALI_UNBIASED_ROTATIONS.log\
		--base_datasets_dir ../master-thesis/datasets/BHSig260/Bengali_trimmed_rotations/""")

	print("HINDI_UNBIASED_ROTATIONS")
	os.system("""time python train_model.py --pairs_file pairs_files/all_samples_random_forgeries/HINDI_UNBIASED_ROTATIONS_pairs.txt \
		--features_file HINDI_UNBIASED_ROTATIONS_features.pk --dataset Hindi --logfile HINDI_UNBIASED_ROTATIONS.log\
		--base_datasets_dir ../master-thesis/datasets/BHSig260/Hindi_trimmed_rotations/""")

	print("CEDAR_UNBIASED_SCALES")
	os.system("""time python train_model.py --pairs_file pairs_files/all_samples_random_forgeries/CEDAR_UNBIASED_SCALES_pairs.txt \
		--features_file CEDAR_UNBIASED_SCALES_features.pk --dataset CEDAR --logfile CEDAR_UNBIASED_SCALES.log\
		--base_datasets_dir ../master-thesis/datasets/CEDAR/""")

	print("BENGALI_UNBIASED_SCALES")
	os.system("""time python train_model.py --pairs_file pairs_files/all_samples_random_forgeries/BENGALI_UNBIASED_SCALES_pairs.txt \
		--features_file BENGALI_UNBIASED_SCALES_features.pk --dataset Bengali --logfile BENGALI_UNBIASED_SCALES.log\
		--base_datasets_dir ../master-thesis/datasets/BHSig260/Bengali_trimmed_scales/""")

	print("HINDI_UNBIASED_SCALES")
	os.system("""time python train_model.py --pairs_file pairs_files/all_samples_random_forgeries/HINDI_UNBIASED_SCALES_pairs.txt \
		--features_file HINDI_UNBIASED_SCALES_features.pk --dataset Hindi --logfile HINDI_UNBIASED_SCALES.log\
		--base_datasets_dir ../master-thesis/datasets/BHSig260/Hindi_trimmed_scales/""")

	print("CEDAR_UNBIASED_ROTATION_SCALES")
	os.system("""time python train_model.py --pairs_file pairs_files/all_samples_random_forgeries/CEDAR_UNBIASED_ROTATION_SCALES_pairs.txt \
		--features_file CEDAR_UNBIASED_ROTATION_SCALES_features.pk --dataset CEDAR --logfile CEDAR_UNBIASED_ROTATION_SCALES.log\
		--base_datasets_dir ../master-thesis/datasets/CEDAR/""")

	print("BENGALI_UNBIASED_ROTATION_SCALES")
	os.system("""time python train_model.py --pairs_file pairs_files/all_samples_random_forgeries/BENGALI_UNBIASED_ROTATION_SCALES_pairs.txt \
		--features_file BENGALI_UNBIASED_ROTATION_SCALES_features.pk --dataset Bengali --logfile BENGALI_UNBIASED_ROTATION_SCALES.log\
		--base_datasets_dir ../master-thesis/datasets/BHSig260/Bengali_trimmed_rotation_scales/""")

	print("HINDI_UNBIASED_ROTATION_SCALES")
	os.system("""time python train_model.py --pairs_file pairs_files/all_samples_random_forgeries/HINDI_UNBIASED_ROTATION_SCALES_pairs.txt \
		--features_file HINDI_UNBIASED_ROTATION_SCALES_features.pk --dataset Hindi --logfile HINDI_UNBIASED_ROTATION_SCALES.log\
		--base_datasets_dir ../master-thesis/datasets/BHSig260/Hindi_trimmed_rotation_scales/""")

if __name__ == "__main__":
	main()