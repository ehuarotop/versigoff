import os
import click

@click.command()
@click.option('--n_samples', default="", help="number of samples to be considered in the signature pairs generation")
def main(n_samples):

	if n_samples == "":
		samples = "all_samples"
	elif n_samples == "all_new":
		samples = "all_samples_new"
	else:
		samples = "{}_samples".format(n_samples)

	print("CEDAR_BIASED")
	os.system("""time python train_model.py --pairs_file pairs_files/all_samples/CEDAR_BIASED_pairs.txt \
		--features_file features/{0}/CEDAR_BIASED_features.pk --dataset CEDAR \
		--base_datasets_dir ../master-thesis/datasets/CEDAR/ --clf_name CEDAR_BIASED_clf.pk --save_classifier""".format(samples))

	print("BENGALI_BIASED")
	os.system("""time python train_model.py --pairs_file pairs_files/all_samples/BENGALI_BIASED_pairs.txt \
		--features_file features/{0}/BENGALI_BIASED_features.pk --dataset Bengali \
		--base_datasets_dir ../master-thesis/datasets/BHSig260/Bengali/ --clf_name BENGALI_BIASED_clf.pk --save_classifier""".format(samples))

	print("HINDI_BIASED")
	os.system("""time python train_model.py --pairs_file pairs_files/all_samples/HINDI_BIASED_pairs.txt \
		--features_file features/{0}/HINDI_BIASED_features.pk --dataset Hindi \
		--base_datasets_dir ../master-thesis/datasets/BHSig260/Hindi/ --clf_name HINDI_BIASED_clf.pk --save_classifier""".format(samples))

	print("CEDAR_UNBIASED")
	os.system("""time python train_model.py --pairs_file pairs_files/all_samples/CEDAR_UNBIASED_pairs.txt \
		--features_file features/{0}/CEDAR_UNBIASED_features.pk --dataset CEDAR \
		--base_datasets_dir ../master-thesis/datasets/CEDAR/ --clf_name CEDAR_UNBIASED_clf.pk --save_classifier""".format(samples))

	print("BENGALI_UNBIASED")
	os.system("""time python train_model.py --pairs_file pairs_files/all_samples/BENGALI_UNBIASED_pairs.txt \
		--features_file features/{0}/BENGALI_UNBIASED_features.pk --dataset Bengali \
		--base_datasets_dir ../master-thesis/datasets/BHSig260/Bengali_trimmed/ --clf_name BENGALI_UNBIASED_clf.pk --save_classifier""".format(samples))

	print("HINDI_UNBIASED")
	os.system("""time python train_model.py --pairs_file pairs_files/all_samples/HINDI_UNBIASED_pairs.txt \
		--features_file features/{0}/HINDI_UNBIASED_features.pk --dataset Hindi \
		--base_datasets_dir ../master-thesis/datasets/BHSig260/Hindi_trimmed/ --clf_name HINDI_UNBIASED_clf.pk --save_classifier""".format(samples))

	print("CEDAR_UNBIASED_ROTATIONS")
	os.system("""time python train_model.py --pairs_file pairs_files/all_samples/CEDAR_UNBIASED_ROTATIONS_pairs.txt \
		--features_file features/{0}/CEDAR_UNBIASED_ROTATIONS_features.pk --dataset CEDAR \
		--base_datasets_dir ../master-thesis/datasets/CEDAR/ --clf_name CEDAR_UNBIASED_ROTATIONS_clf.pk --save_classifier""".format(samples))

	print("BENGALI_UNBIASED_ROTATIONS")
	os.system("""time python train_model.py --pairs_file pairs_files/all_samples/BENGALI_UNBIASED_ROTATIONS_pairs.txt \
		--features_file features/{0}/BENGALI_UNBIASED_ROTATIONS_features.pk --dataset Bengali \
		--base_datasets_dir ../master-thesis/datasets/BHSig260/Bengali_trimmed_rotations/ --clf_name BENGALI_UNBIASED_ROTATIONS_clf.pk --save_classifier""".format(samples))

	print("HINDI_UNBIASED_ROTATIONS")
	os.system("""time python train_model.py --pairs_file pairs_files/all_samples/HINDI_UNBIASED_ROTATIONS_pairs.txt \
		--features_file features/{0}/HINDI_UNBIASED_ROTATIONS_features.pk --dataset Hindi \
		--base_datasets_dir ../master-thesis/datasets/BHSig260/Hindi_trimmed_rotations/ --clf_name HINDI_UNBIASED_ROTATIONS_clf.pk --save_classifier""".format(samples))

	print("CEDAR_UNBIASED_SCALES")
	os.system("""time python train_model.py --pairs_file pairs_files/all_samples/CEDAR_UNBIASED_SCALES_pairs.txt \
		--features_file features/{0}/CEDAR_UNBIASED_SCALES_features.pk --dataset CEDAR \
		--base_datasets_dir ../master-thesis/datasets/CEDAR/ --clf_name CEDAR_UNBIASED_SCALES_clf.pk --save_classifier""".format(samples))

	print("BENGALI_UNBIASED_SCALES")
	os.system("""time python train_model.py --pairs_file pairs_files/all_samples/BENGALI_UNBIASED_SCALES_pairs.txt \
		--features_file features/{0}/BENGALI_UNBIASED_SCALES_features.pk --dataset Bengali \
		--base_datasets_dir ../master-thesis/datasets/BHSig260/Bengali_trimmed_scales/ --clf_name BENGALI_UNBIASED_SCALES_clf.pk --save_classifier""".format(samples))

	print("HINDI_UNBIASED_SCALES")
	os.system("""time python train_model.py --pairs_file pairs_files/all_samples/HINDI_UNBIASED_SCALES_pairs.txt \
		--features_file features/{0}/HINDI_UNBIASED_SCALES_features.pk --dataset Hindi \
		--base_datasets_dir ../master-thesis/datasets/BHSig260/Hindi_trimmed_scales/ --clf_name HINDI_UNBIASED_SCALES_clf.pk --save_classifier""".format(samples))

	print("CEDAR_UNBIASED_ROTATION_SCALES")
	os.system("""time python train_model.py --pairs_file pairs_files/all_samples/CEDAR_UNBIASED_ROTATION_SCALES_pairs.txt \
		--features_file features/{0}/CEDAR_UNBIASED_ROTATION_SCALES_features.pk --dataset CEDAR \
		--base_datasets_dir ../master-thesis/datasets/CEDAR/ --clf_name CEDAR_UNBIASED_ROTATION_SCALES_clf.pk --save_classifier""".format(samples))

	print("BENGALI_UNBIASED_ROTATION_SCALES")
	os.system("""time python train_model.py --pairs_file pairs_files/all_samples/BENGALI_UNBIASED_ROTATION_SCALES_pairs.txt \
		--features_file features/{0}/BENGALI_UNBIASED_ROTATION_SCALES_features.pk --dataset Bengali \
		--base_datasets_dir ../master-thesis/datasets/BHSig260/Bengali_trimmed_rotation_scales/ --clf_name BENGALI_UNBIASED_ROTATION_SCALES_clf.pk --save_classifier""".format(samples))

	print("HINDI_UNBIASED_ROTATION_SCALES")
	os.system("""time python train_model.py --pairs_file pairs_files/all_samples/HINDI_UNBIASED_ROTATION_SCALES_pairs.txt \
		--features_file features/{0}/HINDI_UNBIASED_ROTATION_SCALES_features.pk --dataset Hindi \
		--base_datasets_dir ../master-thesis/datasets/BHSig260/Hindi_trimmed_rotation_scales/ --clf_name HINDI_UNBIASED_ROTATION_SCALES_clf.pk --save_classifier""".format(samples))

if __name__ == "__main__":
	main()