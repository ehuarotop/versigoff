import os
import click

def main():

	print("CEDAR_BIASED")
	os.system("""time python train_model.py --pairs_file pairs_files/all_samples_random_forgeries/CEDAR_BIASED_pairs.txt \
		--features_file CEDAR_BIASED_features.pk --dataset CEDAR --logfile CEDAR_BIASED.log\
		--base_datasets_dir ../master-thesis/datasets/CEDAR/""".format(samples))

	print("BENGALI_BIASED")
	os.system("""time python train_model.py --pairs_file pairs_files/all_samples_random_forgeries/BENGALI_BIASED_pairs.txt \
		--features_file BENGALI_BIASED_features.pk --dataset Bengali --logfile BENGALI_BIASED.log\
		--base_datasets_dir ../master-thesis/datasets/BHSig260/Bengali/""".format(samples))

	print("HINDI_BIASED")
	os.system("""time python train_model.py --pairs_file pairs_files/all_samples_random_forgeries/HINDI_BIASED_pairs.txt \
		--features_file HINDI_BIASED_features.pk --dataset Hindi --logfile HINDI_BIASED.log\
		--base_datasets_dir ../master-thesis/datasets/BHSig260/Hindi/""".format(samples))

	print("CEDAR_UNBIASED")
	os.system("""time python train_model.py --pairs_file pairs_files/all_samples_random_forgeries/CEDAR_UNBIASED_pairs.txt \
		--features_file CEDAR_UNBIASED_features.pk --dataset CEDAR --logfile CEDAR_UNBIASED.log\
		--base_datasets_dir ../master-thesis/datasets/CEDAR/""".format(samples))

	print("BENGALI_UNBIASED")
	os.system("""time python train_model.py --pairs_file pairs_files/all_samples_random_forgeries/BENGALI_UNBIASED_pairs.txt \
		--features_file BENGALI_UNBIASED_features.pk --dataset Bengali --logfile BENGALI_UNBIASED.log\
		--base_datasets_dir ../master-thesis/datasets/BHSig260/Bengali_trimmed/""".format(samples))

	print("HINDI_UNBIASED")
	os.system("""time python train_model.py --pairs_file pairs_files/all_samples_random_forgeries/HINDI_UNBIASED_pairs.txt \
		--features_file HINDI_UNBIASED_features.pk --dataset Hindi --logfile HINDI_UNBIASED.log\
		--base_datasets_dir ../master-thesis/datasets/BHSig260/Hindi_trimmed/""".format(samples))

	print("CEDAR_UNBIASED_ROTATIONS")
	os.system("""time python train_model.py --pairs_file pairs_files/all_samples_random_forgeries/CEDAR_UNBIASED_ROTATIONS_pairs.txt \
		--features_file CEDAR_UNBIASED_ROTATIONS_features.pk --dataset CEDAR --logfile CEDAR_UNBIASED_ROTATIONS.log\
		--base_datasets_dir ../master-thesis/datasets/CEDAR/""".format(samples))

	print("BENGALI_UNBIASED_ROTATIONS")
	os.system("""time python train_model.py --pairs_file pairs_files/all_samples_random_forgeries/BENGALI_UNBIASED_ROTATIONS_pairs.txt \
		--features_file BENGALI_UNBIASED_ROTATIONS_features.pk --dataset Bengali --logfile BENGALI_UNBIASED_ROTATIONS.log\
		--base_datasets_dir ../master-thesis/datasets/BHSig260/Bengali_trimmed_rotations/""".format(samples))

	print("HINDI_UNBIASED_ROTATIONS")
	os.system("""time python train_model.py --pairs_file pairs_files/all_samples_random_forgeries/HINDI_UNBIASED_ROTATIONS_pairs.txt \
		--features_file HINDI_UNBIASED_ROTATIONS_features.pk --dataset Hindi --logfile HINDI_UNBIASED_ROTATIONS.log\
		--base_datasets_dir ../master-thesis/datasets/BHSig260/Hindi_trimmed_rotations/""".format(samples))

	print("CEDAR_UNBIASED_SCALES")
	os.system("""time python train_model.py --pairs_file pairs_files/all_samples_random_forgeries/CEDAR_UNBIASED_SCALES_pairs.txt \
		--features_file CEDAR_UNBIASED_SCALES_features.pk --dataset CEDAR --logfile CEDAR_UNBIASED_SCALES.log\
		--base_datasets_dir ../master-thesis/datasets/CEDAR/""".format(samples))

	print("BENGALI_UNBIASED_SCALES")
	os.system("""time python train_model.py --pairs_file pairs_files/all_samples_random_forgeries/BENGALI_UNBIASED_SCALES_pairs.txt \
		--features_file BENGALI_UNBIASED_SCALES_features.pk --dataset Bengali --logfile BENGALI_UNBIASED_SCALES.log\
		--base_datasets_dir ../master-thesis/datasets/BHSig260/Bengali_trimmed_scales/""".format(samples))

	print("HINDI_UNBIASED_SCALES")
	os.system("""time python train_model.py --pairs_file pairs_files/all_samples_random_forgeries/HINDI_UNBIASED_SCALES_pairs.txt \
		--features_file HINDI_UNBIASED_SCALES_features.pk --dataset Hindi --logfile HINDI_UNBIASED_SCALES.log\
		--base_datasets_dir ../master-thesis/datasets/BHSig260/Hindi_trimmed_scales/""".format(samples))

	print("CEDAR_UNBIASED_ROTATION_SCALES")
	os.system("""time python train_model.py --pairs_file pairs_files/all_samples_random_forgeries/CEDAR_UNBIASED_ROTATION_SCALES_pairs.txt \
		--features_file CEDAR_UNBIASED_ROTATION_SCALES_features.pk --dataset CEDAR --logfile CEDAR_UNBIASED_ROTATION_SCALES.log\
		--base_datasets_dir ../master-thesis/datasets/CEDAR/""".format(samples))

	print("BENGALI_UNBIASED_ROTATION_SCALES")
	os.system("""time python train_model.py --pairs_file pairs_files/all_samples_random_forgeries/BENGALI_UNBIASED_ROTATION_SCALES_pairs.txt \
		--features_file BENGALI_UNBIASED_ROTATION_SCALES_features.pk --dataset Bengali --logfile BENGALI_UNBIASED_ROTATION_SCALES.log\
		--base_datasets_dir ../master-thesis/datasets/BHSig260/Bengali_trimmed_rotation_scales/""".format(samples))

	print("HINDI_UNBIASED_ROTATION_SCALES")
	os.system("""time python train_model.py --pairs_file pairs_files/all_samples_random_forgeries/HINDI_UNBIASED_ROTATION_SCALES_pairs.txt \
		--features_file HINDI_UNBIASED_ROTATION_SCALES_features.pk --dataset Hindi --logfile HINDI_UNBIASED_ROTATION_SCALES.log\
		--base_datasets_dir ../master-thesis/datasets/BHSig260/Hindi_trimmed_rotation_scales/""".format(samples))

if __name__ == "__main__":
	main()