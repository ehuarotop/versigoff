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
	os.system("""python ../pairs_generator_random_forgeries.py --dataset Hindi --pairs_file HINDI_BIASED_pairs.txt {}""".format(samples_option))
	os.system("""python ../pairs_generator_random_forgeries.py --dataset Hindi --pairs_file HINDI_UNBIASED_pairs.txt {} --transform trim""".format( samples_option))
	os.system("""python ../pairs_generator_random_forgeries.py --dataset Hindi --pairs_file HINDI_UNBIASED_ROTATIONS_pairs.txt {} --transform trim_rotation""".format(samples_option))
	os.system("""python ../pairs_generator_random_forgeries.py --dataset Hindi --pairs_file HINDI_UNBIASED_SCALES_pairs.txt {} --transform trim_scale""".format(samples_option))
	os.system("""python ../pairs_generator_random_forgeries.py --dataset Hindi --pairs_file HINDI_UNBIASED_ROTATION_SCALES_pairs.txt {} --transform trim_rotation_scale""".format(samples_option))

	#BENGALI
	os.system("""python ../pairs_generator_random_forgeries.py --dataset Bengali --pairs_file BENGALI_BIASED_pairs.txt {}""".format(samples_option))
	os.system("""python ../pairs_generator_random_forgeries.py --dataset Bengali --pairs_file BENGALI_UNBIASED_pairs.txt {} --transform trim""".format(samples_option))
	os.system("""python ../pairs_generator_random_forgeries.py --dataset Bengali --pairs_file BENGALI_UNBIASED_ROTATIONS_pairs.txt {} --transform trim_rotation""".format(samples_option))
	os.system("""python ../pairs_generator_random_forgeries.py --dataset Bengali --pairs_file BENGALI_UNBIASED_SCALES_pairs.txt {} --transform trim_scale""".format(samples_option))
	os.system("""python ../pairs_generator_random_forgeries.py --dataset Bengali --pairs_file BENGALI_UNBIASED_ROTATION_SCALES_pairs.txt {} --transform trim_rotation_scale""".format(samples_option))

	#CEDAR
	os.system("""python ../pairs_generator_random_forgeries.py --dataset CEDAR \
					--pairs_file CEDAR_BIASED_pairs.txt --image_dir_genuine ../../master-thesis/datasets/CEDAR/full_org\
					{}""".format(samples_option))
	os.system("""python ../pairs_generator_random_forgeries.py --dataset CEDAR \
					--pairs_file CEDAR_UNBIASED_pairs.txt --image_dir_genuine ../../master-thesis/datasets/CEDAR/full_org_hist_transform\
					{}""".format(samples_option))
	os.system("""python ../pairs_generator_random_forgeries.py --dataset CEDAR \
					--pairs_file CEDAR_UNBIASED_ROTATIONS_pairs.txt --image_dir_genuine ../../master-thesis/datasets/CEDAR/full_org_rotations_1_new\
					{} --transform rotation""".format(samples_option))
	os.system("""python ../pairs_generator_random_forgeries.py --dataset CEDAR \
					--pairs_file CEDAR_UNBIASED_SCALES_pairs.txt --image_dir_genuine ../../master-thesis/datasets/CEDAR/full_org_scales_1\
					{} --transform scale""".format(samples_option))
	os.system("""python ../pairs_generator_random_forgeries.py --dataset CEDAR \
					--pairs_file CEDAR_BIASED_ROTATION_SCALES_pairs.txt --image_dir_genuine ../../master-thesis/datasets/CEDAR/full_org_rotation_scales_1\
					{} --transform rotation_scale""".format(samples_option))

if __name__ == "__main__":
	main()