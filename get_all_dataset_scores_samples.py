import os
import click

@click.command()
@click.option('--n_samples', default="", help="number of samples to be considered in the signature pairs generation")
def main(n_samples):

	print("CEDAR_BIASED")
	os.system("""time python dataset_score.py --pairs_file pairs_files/{0}_samples_complement/CEDAR_BIASED_pairs.txt \
		--features_file features/all_samples_new/CEDAR_BIASED_features.pk --dataset CEDAR --logfile dataset_scores.log\
		--base_datasets_dir ../master-thesis/datasets/CEDAR/ --clf classifiers/{0}_samples/CEDAR_BIASED_clf.pk """).format(n_samples)

	print("BENGALI_BIASED")
	os.system("""time python dataset_score.py --pairs_file pairs_files/{0}_samples_complement/BENGALI_BIASED_pairs.txt \
		--features_file features/all_samples_new/BENGALI_BIASED_features.pk --dataset Bengali --logfile dataset_scores.log\
		--base_datasets_dir ../master-thesis/datasets/BHSig260/Bengali/ --clf classifiers/{0}_samples/BENGALI_BIASED_clf.pk """).format(n_samples)

	print("HINDI_BIASED")
	os.system("""time python dataset_score.py --pairs_file pairs_files/{0}_samples_complement/HINDI_BIASED_pairs.txt \
		--features_file features/all_samples_new/HINDI_BIASED_features.pk --dataset Hindi --logfile dataset_scores.log\
		--base_datasets_dir ../master-thesis/datasets/BHSig260/Hindi/ --clf classifiers/{0}_samples/HINDI_BIASED_clf.pk """).format(n_samples)

	print("CEDAR_UNBIASED")
	os.system("""time python dataset_score.py --pairs_file pairs_files/{0}_samples_complement/CEDAR_UNBIASED_pairs.txt \
		--features_file features/all_samples_new/CEDAR_UNBIASED_features.pk --dataset CEDAR --logfile dataset_scores.log\
		--base_datasets_dir ../master-thesis/datasets/CEDAR/ --clf classifiers/{0}_samples/CEDAR_UNBIASED_clf.pk """).format(n_samples)

	print("BENGALI_UNBIASED")
	os.system("""time python dataset_score.py --pairs_file pairs_files/{0}_samples_complement/BENGALI_UNBIASED_pairs.txt \
		--features_file features/all_samples_new/BENGALI_UNBIASED_features.pk --dataset Bengali --logfile dataset_scores.log\
		--base_datasets_dir ../master-thesis/datasets/BHSig260/Bengali_trimmed/ --clf classifiers/{0}_samples/BENGALI_UNBIASED_clf.pk """).format(n_samples)

	print("HINDI_UNBIASED")
	os.system("""time python dataset_score.py --pairs_file pairs_files/{0}_samples_complement/HINDI_UNBIASED_pairs.txt \
		--features_file features/all_samples_new/HINDI_UNBIASED_features.pk --dataset Hindi --logfile dataset_scores.log\
		--base_datasets_dir ../master-thesis/datasets/BHSig260/Hindi_trimmed/ --clf classifiers/{0}_samples/HINDI_UNBIASED_clf.pk """).format(n_samples)

	print("CEDAR_UNBIASED_ROTATIONS")
	os.system("""time python dataset_score.py --pairs_file pairs_files/{0}_samples_complement/CEDAR_UNBIASED_ROTATIONS_pairs.txt \
		--features_file features/all_samples_new/CEDAR_UNBIASED_ROTATIONS_features.pk --dataset CEDAR --logfile dataset_scores.log\
		--base_datasets_dir ../master-thesis/datasets/CEDAR/ --clf classifiers/{0}_samples/CEDAR_UNBIASED_ROTATIONS_clf.pk """).format(n_samples)

	print("BENGALI_UNBIASED_ROTATIONS")
	os.system("""time python dataset_score.py --pairs_file pairs_files/{0}_samples_complement/BENGALI_UNBIASED_ROTATIONS_pairs.txt \
		--features_file features/all_samples_new/BENGALI_UNBIASED_ROTATIONS_features.pk --dataset Bengali --logfile dataset_scores.log\
		--base_datasets_dir ../master-thesis/datasets/BHSig260/Bengali_trimmed_rotations/ --clf classifiers/{0}_samples/BENGALI_UNBIASED_ROTATIONS_clf.pk """).format(n_samples)

	print("HINDI_UNBIASED_ROTATIONS")
	os.system("""time python dataset_score.py --pairs_file pairs_files/{0}_samples_complement/HINDI_UNBIASED_ROTATIONS_pairs.txt \
		--features_file features/all_samples_new/HINDI_UNBIASED_ROTATIONS_features.pk --dataset Hindi --logfile dataset_scores.log\
		--base_datasets_dir ../master-thesis/datasets/BHSig260/Hindi_trimmed_rotations/ --clf classifiers/{0}_samples/HINDI_UNBIASED_ROTATIONS_clf.pk """).format(n_samples)

	print("CEDAR_UNBIASED_SCALES")
	os.system("""time python dataset_score.py --pairs_file pairs_files/{0}_samples_complement/CEDAR_UNBIASED_SCALES_pairs.txt \
		--features_file features/all_samples_new/CEDAR_UNBIASED_SCALES_features.pk --dataset CEDAR --logfile dataset_scores.log\
		--base_datasets_dir ../master-thesis/datasets/CEDAR/ --clf classifiers/{0}_samples/CEDAR_UNBIASED_SCALES_clf.pk """).format(n_samples)

	print("BENGALI_UNBIASED_SCALES")
	os.system("""time python dataset_score.py --pairs_file pairs_files/{0}_samples_complement/BENGALI_UNBIASED_SCALES_pairs.txt \
		--features_file features/all_samples_new/BENGALI_UNBIASED_SCALES_features.pk --dataset Bengali --logfile dataset_scores.log\
		--base_datasets_dir ../master-thesis/datasets/BHSig260/Bengali_trimmed_scales/ --clf classifiers/{0}_samples/BENGALI_UNBIASED_SCALES_clf.pk """).format(n_samples)

	print("HINDI_UNBIASED_SCALES")
	os.system("""time python dataset_score.py --pairs_file pairs_files/{0}_samples_complement/HINDI_UNBIASED_SCALES_pairs.txt \
		--features_file features/all_samples_new/HINDI_UNBIASED_SCALES_features.pk --dataset Hindi --logfile dataset_scores.log\
		--base_datasets_dir ../master-thesis/datasets/BHSig260/Hindi_trimmed_scales/ --clf classifiers/{0}_samples/HINDI_UNBIASED_SCALES_clf.pk """).format(n_samples)

	print("CEDAR_UNBIASED_ROTATION_SCALES")
	os.system("""time python dataset_score.py --pairs_file pairs_files/{0}_samples_complement/CEDAR_UNBIASED_ROTATION_SCALES_pairs.txt \
		--features_file features/all_samples_new/CEDAR_UNBIASED_ROTATION_SCALES_features.pk --dataset CEDAR --logfile dataset_scores.log\
		--base_datasets_dir ../master-thesis/datasets/CEDAR/ --clf classifiers/{0}_samples/CEDAR_UNBIASED_ROTATION_SCALES_clf.pk """).format(n_samples)

	print("BENGALI_UNBIASED_ROTATION_SCALES")
	os.system("""time python dataset_score.py --pairs_file pairs_files/{0}_samples_complement/BENGALI_UNBIASED_ROTATION_SCALES_pairs.txt \
		--features_file features/all_samples_new/BENGALI_UNBIASED_ROTATION_SCALES_features.pk --dataset Bengali --logfile dataset_scores.log\
		--base_datasets_dir ../master-thesis/datasets/BHSig260/Bengali_trimmed_rotation_scales/ --clf classifiers/{0}_samples/BENGALI_UNBIASED_ROTATION_SCALES_clf.pk """).format(n_samples)

	print("HINDI_UNBIASED_ROTATION_SCALES")
	os.system("""time python dataset_score.py --pairs_file pairs_files/{0}_samples_complement/HINDI_UNBIASED_ROTATION_SCALES_pairs.txt \
		--features_file features/all_samples_new/HINDI_UNBIASED_ROTATION_SCALES_features.pk --dataset Hindi --logfile dataset_scores.log\
		--base_datasets_dir ../master-thesis/datasets/BHSig260/Hindi_trimmed_rotation_scales/ --clf classifiers/{0}_samples/HINDI_UNBIASED_ROTATION_SCALES_clf.pk """).format(n_samples)

if __name__ == "__main__":
	main()