import os
import click

@click.command()
@click.option('--n_samples', default="", help="number of samples to be considered in the signature pairs generation")
def main(n_samples):
	if n_samples != "":
		samples_option = "--n_samples {}".format(n_samples)
	else:
		samples_option = ""

	#HINDI
	os.system("""python ../pairs_generator.py --dataset Hindi --pairs_file HINDI_BIASED_pairs.txt {}""".format(samples_option))
	os.system("""python ../pairs_generator.py --dataset Hindi --pairs_file HINDI_UNBIASED_pairs.txt {}""".format( samples_option))
	os.system("""python ../pairs_generator.py --dataset Hindi --pairs_file HINDI_UNBIASED_ROTATIONS_pairs.txt {}""".format(samples_option))
	os.system("""python ../pairs_generator.py --dataset Hindi --pairs_file HINDI_UNBIASED_SCALES_pairs.txt {}""".format(samples_option))
	os.system("""python ../pairs_generator.py --dataset Hindi --pairs_file HINDI_UNBIASED_ROTATION_SCALES_pairs.txt {}""".format(samples_option))

	#BENGALI
	os.system("""python ../pairs_generator.py --dataset Bengali --pairs_file BENGALI_BIASED_pairs.txt {}""".format(samples_option))
	os.system("""python ../pairs_generator.py --dataset Bengali --pairs_file BENGALI_UNBIASED_pairs.txt {}""".format(samples_option))
	os.system("""python ../pairs_generator.py --dataset Bengali --pairs_file BENGALI_UNBIASED_ROTATIONS_pairs.txt {}""".format(samples_option))
	os.system("""python ../pairs_generator.py --dataset Bengali --pairs_file BENGALI_UNBIASED_SCALES_pairs.txt {}""".format(samples_option))
	os.system("""python ../pairs_generator.py --dataset Bengali --pairs_file BENGALI_UNBIASED_ROTATION_SCALESpairs.txt {}""".format(samples_option))

	#CEDAR
	os.system("""python ../pairs_generator.py --dataset CEDAR \
					--pairs_file CEDAR_BIASED_pairs.txt --image_dir_genuine ../../master-thesis/datasets/CEDAR/full_org\
					--image_dir_forgery ../../master-thesis/datasets/CEDAR/full_forg {}""".format(samples_option))
	os.system("""python ../pairs_generator.py --dataset CEDAR \
					--pairs_file CEDAR_UNBIASED_pairs.txt --image_dir_genuine ../../master-thesis/datasets/CEDAR/full_org_hist_transform\
					--image_dir_forgery ../../master-thesis/datasets/CEDAR/full_forg {}""".format(samples_option))
	os.system("""python ../pairs_generator.py --dataset CEDAR \
					--pairs_file CEDAR_UNBIASED_ROTATIONS_pairs.txt --image_dir_genuine ../../master-thesis/datasets/CEDAR/full_org_rotations_1_new\
					--image_dir_forgery ../../master-thesis/datasets/CEDAR/full_forg_rotations1_new {}""".format(samples_option))
	os.system("""python ../pairs_generator.py --dataset CEDAR \
					--pairs_file CEDAR_UNBIASED_SCALES_pairs.txt --image_dir_genuine ../../master-thesis/datasets/CEDAR/full_org_scales_1\
					--image_dir_forgery ../../master-thesis/datasets/CEDAR/full_forg_scales_1 {}""".format(samples_option))
	os.system("""python ../pairs_generator.py --dataset CEDAR \
					--pairs_file CEDAR_BIASED_ROTATION_SCALES_pairs.txt --image_dir_genuine ../../master-thesis/datasets/CEDAR/full_org_rotation_scales_1\
					--image_dir_forgery ../../master-thesis/datasets/CEDAR/full_forg_rotation_scales_1 {}""".format(samples_option))

if __name__ == "__main__":
	main()