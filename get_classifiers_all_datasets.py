import os
import click

@click.command()
@click.option('--features_dir', default="", help="number of samples to be considered in the signature pairs generation")
def main():
	os.system("""python train_model.py --features_file {} \
				--clf_name CEDAR_BIASED_clf.pk --save_classifier""".format(os.path.join(features_dir, "CEDAR_BIASED_features.pk")))
	os.system("""python train_model.py --features_file {} \
				--clf_name CEDAR_UNBIASED_clf.pk --save_classifier""".format(os.path.join(features_dir, "CEDAR_UNBIASED_features.pk")))
	os.system("""python train_model.py --features_file {} \
				--clf_name CEDAR_UNBIASED_ROTATIONS_clf.pk --save_classifier""".format(os.path.join(features_dir, "CEDAR_UNBIASED_ROTATIONS_features.pk")))
	os.system("""python train_model.py --features_file {} \
				--clf_name CEDAR_UNBIASED_SCALES_clf.pk --save_classifier""".format(os.path.join(features_dir, "CEDAR_UNBIASED_SCALES_features.pk")))
	os.system("""python train_model.py --features_file {} \
				--clf_name CEDAR_UNBIASED_ROTATION_SCALES_clf.pk --save_classifier""".format(os.path.join(features_dir, "CEDAR_UNBIASED_ROTATION_SCALES_features.pk")))
	os.system("""python train_model.py --features_file {} \
				--clf_name BENGALI_BIASED_clf.pk --save_classifier""".format(os.path.join(features_dir, "BENGALI_BIASED_features.pk")))
	os.system("""python train_model.py --features_file {} \
				--clf_name BENGALI_UNBIASED_clf.pk --save_classifier""".format(os.path.join(features_dir, "BENGALI_UNBIASED_features.pk")))
	os.system("""python train_model.py --features_file {} \
				--clf_name BENGALI_UNBIASED_ROTATIONS_clf.pk --save_classifier""".format(os.path.join(features_dir, "BENGALI_UNBIASED_ROTATIONS_features.pk")))
	os.system("""python train_model.py --features_file {} \
				--clf_name BENGALI_UNBIASED_SCALES_clf.pk --save_classifier""".format(os.path.join(features_dir, "BENGALI_UNBIASED_SCALES_features.pk")))
	os.system("""python train_model.py --features_file {} \
				--clf_name BENGALI_UNBIASED_ROTATION_SCALES_clf.pk --save_classifier""".format(os.path.join(features_dir, "BENGALI_UNBIASED_ROTATION_SCALES_features.pk")))
	os.system("""python train_model.py --features_file {} \
				--clf_name HINDI_BIASED_clf.pk --save_classifier""".format(os.path.join(features_dir, "HINDI_BIASED_features.pk")))
	os.system("""python train_model.py --features_file {} \
				--clf_name HINDI_UNBIASED_clf.pk --save_classifier""".format(os.path.join(features_dir, "HINDI_UNBIASED_features.pk")))
	os.system("""python train_model.py --features_file {} \
				--clf_name HINDI_UNBIASED_ROTATIONS_clf.pk --save_classifier""".format(os.path.join(features_dir, "HINDI_UNBIASED_ROTATIONS_features.pk")))
	os.system("""python train_model.py --features_file {} \
				--clf_name HINDI_UNBIASED_SCALES_clf.pk --save_classifier""".format(os.path.join(features_dir, "HINDI_UNBIASED_SCALES_features.pk")))
	os.system("""python train_model.py --features_file {} \
				--clf_name HINDI_UNBIASED_ROTATION_SCALES_clf.pk --save_classifier""".format(os.path.join(features_dir, "HINDI_UNBIASED_ROTATION_SCALES_features.pk")))

if __name__ == "__main__":
	main()