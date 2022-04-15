import os
import click
import random
import numpy as np

seed = 1337
np.random.seed((int)(seed))
random.seed((int)(seed))

@click.command()
@click.option('--dataset', help='dataset name')
@click.option('--pairs_file', help='pairs file, where pairs will be saved')
@click.option('--image_dir_genuine', default="", help='image_directory where to look for genuine signatures')
@click.option('--image_dir_forgery', default="", help='image_directory where to look for forgery signatures')
@click.option('--n_samples', default="", help="number of samples to be considered in the signature pairs generation")
@click.option('--transform', default="", help='type of transform applied over the dataset (to get the name), could be: rotation, scale, rotation_scale, trim')
def main(dataset, pairs_file, image_dir_genuine, image_dir_forgery, n_samples, transform):
	if dataset == "Bengali":
		#Setting number of writers and number of genuine and forged signatures for each writer.
		num_writers = 100
		gen_sig_per_writer = 24
		forg_sig_per_writer = 30

		#Getting writers directories.
		writers_dir = ["{:03d}".format(i) for i in range(1, num_writers+1)]

		dataset_lines = []

		if n_samples != "":
			gen_sigs_per_writer = random.sample(list(range(1, gen_sig_per_writer+1)), (int)(n_samples))
			forg_sigs_per_writer = random.sample(list(range(1, forg_sig_per_writer+1)), (int)(n_samples))

			#uncomment to generate n_samples complement
			#gen_sigs_per_writer = list(set(list(range(1, gen_sig_per_writer+1)))-set(gen_sigs_per_writer))
			#forg_sigs_per_writer = list(set(list(range(1, forg_sig_per_writer+1)))-set(forg_sigs_per_writer))
		else:
			gen_sigs_per_writer = list(range(1, gen_sig_per_writer+1))
			forg_sigs_per_writer = list(range(1, forg_sig_per_writer+1))

		#generating pairs for each writer
		for ix, writer_dir in enumerate(writers_dir):
			if transform == "":
				#Generating list of genuine and forgeries image names
				gen_signatures = ["B-S-{ix_writer}-G-{:02d}.tif".format(i, ix_writer=ix+1) for i in gen_sigs_per_writer]
				forg_signatures = ["B-S-{ix_writer}-F-{:02d}.tif".format(i, ix_writer=ix+1) for i in forg_sigs_per_writer]
			elif transform == "trim":
				#Generating list of genuine and forgeries image names
				gen_signatures = ["B-S-{ix_writer}-G-{:02d}_trimmed.png".format(i, ix_writer=ix+1) for i in gen_sigs_per_writer]
				forg_signatures = ["B-S-{ix_writer}-F-{:02d}_trimmed.png".format(i, ix_writer=ix+1) for i in forg_sigs_per_writer]
			elif transform == "trim_rotation":
				#Generating list of genuine and forgeries image names
				gen_signatures = ["B-S-{ix_writer}-G-{:02d}_trimmed_rotation1.png".format(i, ix_writer=ix+1) for i in gen_sigs_per_writer]
				forg_signatures = ["B-S-{ix_writer}-F-{:02d}_trimmed_rotation1.png".format(i, ix_writer=ix+1) for i in forg_sigs_per_writer]
			elif transform == "trim_scale":
				#Generating list of genuine and forgeries image names
				gen_signatures = ["B-S-{ix_writer}-G-{:02d}_trimmed_scale1.png".format(i, ix_writer=ix+1) for i in gen_sigs_per_writer]
				forg_signatures = ["B-S-{ix_writer}-F-{:02d}_trimmed_scale1.png".format(i, ix_writer=ix+1) for i in forg_sigs_per_writer]
			elif transform == "trim_rotation_scale":
				#Generating list of genuine and forgeries image names
				gen_signatures = ["B-S-{ix_writer}-G-{:02d}_trimmed_rotation_scale1.png".format(i, ix_writer=ix+1) for i in gen_sigs_per_writer]
				forg_signatures = ["B-S-{ix_writer}-F-{:02d}_trimmed_rotation_scale1.png".format(i, ix_writer=ix+1) for i in forg_sigs_per_writer]

			
			writer_lines = []

			#Generating pairs (genuine / forged)
			for f_sig in forg_signatures:
				for g_sig in gen_signatures:
					line = os.path.join(writer_dir, g_sig) + " " + os.path.join(writer_dir, f_sig) + " 0"
					writer_lines.append(line)

			#Generating pairs (genuine / genuine)
			for ix, g_sig in enumerate(gen_signatures):
				for i in range(ix+1, len(gen_sigs_per_writer)):
					line = os.path.join(writer_dir, gen_signatures[i]) + " " + os.path.join(writer_dir, g_sig) + " 1"
					writer_lines.append(line)

			#Concatenating previous generated lines with the lines for the current writer
			dataset_lines += writer_lines

		#Writing lines to a txt file for Bengali pairs
		with open(pairs_file, 'w') as filehandle:
			filehandle.writelines("%s\n" % line for line in dataset_lines)
	elif dataset == "Hindi":
		#Setting number of writers and number of genuine and forged signatures for each writer.
		num_writers = 160
		gen_sig_per_writer = 24
		forg_sig_per_writer = 30

		#Getting writers directories.
		writers_dir = ["{:03d}".format(i) for i in range(1, num_writers+1)]

		dataset_lines = []

		if n_samples != "":
			gen_sigs_per_writer = random.sample(list(range(1, gen_sig_per_writer+1)), (int)(n_samples))
			forg_sigs_per_writer = random.sample(list(range(1, forg_sig_per_writer+1)), (int)(n_samples))

			#uncomment to generate n_samples complement
			#gen_sigs_per_writer = list(set(list(range(1, gen_sig_per_writer+1)))-set(gen_sigs_per_writer))
			#forg_sigs_per_writer = list(set(list(range(1, forg_sig_per_writer+1)))-set(forg_sigs_per_writer))
		else:
			gen_sigs_per_writer = list(range(1, gen_sig_per_writer+1))
			forg_sigs_per_writer = list(range(1, forg_sig_per_writer+1))

		#Generating pairs for each writer
		for ix, writer_dir in enumerate(writers_dir):
			if ix+1 not in [11,17,18,35,76,87,93,123]:
				if transform == "":
					#Generating list of genuine and forgeries image names
					gen_signatures = ["H-S-{ix_writer}-G-{:02d}.tif".format(i, ix_writer=ix+1) for i in gen_sigs_per_writer]
					forg_signatures = ["H-S-{ix_writer}-F-{:02d}.tif".format(i, ix_writer=ix+1) for i in forg_sigs_per_writer]
				elif transform == "trim":
					#Generating list of genuine and forgeries image names
					gen_signatures = ["H-S-{ix_writer}-G-{:02d}_trimmed.png".format(i, ix_writer=ix+1) for i in gen_sigs_per_writer]
					forg_signatures = ["H-S-{ix_writer}-F-{:02d}_trimmed.png".format(i, ix_writer=ix+1) for i in forg_sigs_per_writer]
				elif transform == "trim_rotation":
					#Generating list of genuine and forgeries image names
					gen_signatures = ["H-S-{ix_writer}-G-{:02d}_trimmed_rotation1.png".format(i, ix_writer=ix+1) for i in gen_sigs_per_writer]
					forg_signatures = ["H-S-{ix_writer}-F-{:02d}_trimmed_rotation1.png".format(i, ix_writer=ix+1) for i in forg_sigs_per_writer]
				elif transform == "trim_scale":
					#Generating list of genuine and forgeries image names
					gen_signatures = ["H-S-{ix_writer}-G-{:02d}_trimmed_scale1.png".format(i, ix_writer=ix+1) for i in gen_sigs_per_writer]
					forg_signatures = ["H-S-{ix_writer}-F-{:02d}_trimmed_scale1.png".format(i, ix_writer=ix+1) for i in forg_sigs_per_writer]
				elif transform == "trim_rotation_scale":
					#Generating list of genuine and forgeries image names
					gen_signatures = ["H-S-{ix_writer}-G-{:02d}_trimmed_rotation_scale1.png".format(i, ix_writer=ix+1) for i in gen_sigs_per_writer]
					forg_signatures = ["H-S-{ix_writer}-F-{:02d}_trimmed_rotation_scale1.png".format(i, ix_writer=ix+1) for i in forg_sigs_per_writer]
			elif ix+1==123 :
				if transform == "":
					#Generating list of genuine and forgeries image names
					gen_signatures = ["H-S-{ix_writer}-G-{:02d}.tif".format(i, ix_writer=ix+2) for i in gen_sigs_per_writer]
					forg_signatures = ["H-S-{ix_writer}-F-{:02d}.tif".format(i, ix_writer=ix+1) for i in forg_sigs_per_writer]
				elif transform == "trim":
					#Generating list of genuine and forgeries image names
					gen_signatures = ["H-S-{ix_writer}-G-{:02d}_trimmed.png".format(i, ix_writer=ix+2) for i in gen_sigs_per_writer]
					forg_signatures = ["H-S-{ix_writer}-F-{:02d}_trimmed.png".format(i, ix_writer=ix+1) for i in forg_sigs_per_writer]
				elif transform == "trim_rotation":
					#Generating list of genuine and forgeries image names
					gen_signatures = ["H-S-{ix_writer}-G-{:02d}_trimmed_rotation1.png".format(i, ix_writer=ix+2) for i in gen_sigs_per_writer]
					forg_signatures = ["H-S-{ix_writer}-F-{:02d}_trimmed_rotation1.png".format(i, ix_writer=ix+1) for i in forg_sigs_per_writer]
				elif transform == "trim_scale":
					#Generating list of genuine and forgeries image names
					gen_signatures = ["H-S-{ix_writer}-G-{:02d}_trimmed_scale1.png".format(i, ix_writer=ix+2) for i in gen_sigs_per_writer]
					forg_signatures = ["H-S-{ix_writer}-F-{:02d}_trimmed_scale1.png".format(i, ix_writer=ix+1) for i in forg_sigs_per_writer]
				elif transform == "trim_rotation_scale":
					#Generating list of genuine and forgeries image names
					gen_signatures = ["H-S-{ix_writer}-G-{:02d}_trimmed_rotation_scale1.png".format(i, ix_writer=ix+2) for i in gen_sigs_per_writer]
					forg_signatures = ["H-S-{ix_writer}-F-{:02d}_trimmed_rotation_scale1.png".format(i, ix_writer=ix+1) for i in forg_sigs_per_writer]
			else:
				if transform == "":
					#Generating list of genuine and forgeries image names
					gen_signatures = ["H-S-{:03d}-G-{:02d}.tif".format(ix+1, i) for i in gen_sigs_per_writer]
					forg_signatures = ["H-S-{:03d}-F-{:02d}.tif".format(ix+1, i) for i in forg_sigs_per_writer]
				elif transform == "trim":
					#Generating list of genuine and forgeries image names
					gen_signatures = ["H-S-{:03d}-G-{:02d}_trimmed.png".format(ix+1, i) for i in gen_sigs_per_writer]
					forg_signatures = ["H-S-{:03d}-F-{:02d}_trimmed.png".format(ix+1, i) for i in forg_sigs_per_writer]
				elif transform == "trim_rotation":
					#Generating list of genuine and forgeries image names
					gen_signatures = ["H-S-{:03d}-G-{:02d}_trimmed_rotation1.png".format(ix+1, i) for i in gen_sigs_per_writer]
					forg_signatures = ["H-S-{:03d}-F-{:02d}_trimmed_rotation1.png".format(ix+1, i) for i in forg_sigs_per_writer]
				elif transform == "trim_scale":
					#Generating list of genuine and forgeries image names
					gen_signatures = ["H-S-{:03d}-G-{:02d}_trimmed_scale1.png".format(ix+1, i) for i in gen_sigs_per_writer]
					forg_signatures = ["H-S-{:03d}-F-{:02d}_trimmed_scale1.png".format(ix+1, i) for i in forg_sigs_per_writer]
				elif transform == "trim_rotation_scale":
					#Generating list of genuine and forgeries image names
					gen_signatures = ["H-S-{:03d}-G-{:02d}_trimmed_rotation_scale1.png".format(ix+1, i) for i in gen_sigs_per_writer]
					forg_signatures = ["H-S-{:03d}-F-{:02d}_trimmed_rotation_scale1.png".format(ix+1, i) for i in forg_sigs_per_writer]

			writer_lines = []

			#Generating pairs (genuine / forged)
			for f_sig in forg_signatures:
				for g_sig in gen_signatures:
					line = os.path.join(writer_dir, g_sig) + " " + os.path.join(writer_dir, f_sig) + " 0"
					writer_lines.append(line)

			#Generating pairs (genuine / genuine)
			for ix, g_sig in enumerate(gen_signatures):
				for i in range(ix+1, len(gen_sigs_per_writer)):
					line = os.path.join(writer_dir, gen_signatures[i]) + " " + os.path.join(writer_dir, g_sig) + " 1"
					writer_lines.append(line)

			#Concatenating previous generated lines with the lines for the current writer
			dataset_lines += writer_lines

		#Writing lines to a txt file for Bengali pairs
		with open(pairs_file, 'w') as filehandle:
			filehandle.writelines("%s\n" % line for line in dataset_lines)

	elif dataset == "CEDAR":
		num_writers = 55
		gen_sig_per_writer = 24
		forg_sig_per_writer = 24

		dataset_lines = []

		writers = list(range(1, num_writers+1))

		if n_samples != "":
			gen_sigs_per_writer = random.sample(list(range(1, gen_sig_per_writer+1)), (int)(n_samples))
			forg_sigs_per_writer = random.sample(list(range(1, forg_sig_per_writer+1)), (int)(n_samples))
			
			#uncomment to generate n_samples complement
			#gen_sigs_per_writer = list(set(list(range(1, gen_sig_per_writer+1)))-set(gen_sigs_per_writer))
			#forg_sigs_per_writer = list(set(list(range(1, forg_sig_per_writer+1)))-set(forg_sigs_per_writer))
		else:
			gen_sigs_per_writer = list(range(1, gen_sig_per_writer+1))
			forg_sigs_per_writer = list(range(1, forg_sig_per_writer+1))

		for ix_writer in writers:
			writer_lines = []

			if transform == "rotation":
				gen_signatures = ["original_{0}_{1}_rotation1.png".format(ix_writer,i) for i in gen_sigs_per_writer]
				forg_signatures = ["forgeries_{0}_{1}_rotation1.png".format(ix_writer,i) for i in forg_sigs_per_writer]
			elif transform == "scale":
				gen_signatures = ["original_{0}_{1}_scale1.png".format(ix_writer,i) for i in gen_sigs_per_writer]
				forg_signatures = ["forgeries_{0}_{1}_scale1.png".format(ix_writer,i) for i in forg_sigs_per_writer]
			elif transform == "rotation_scale":
				gen_signatures = ["original_{0}_{1}_rotation_scale1.png".format(ix_writer,i) for i in gen_sigs_per_writer]
				forg_signatures = ["forgeries_{0}_{1}_rotation_scale1.png".format(ix_writer,i) for i in forg_sigs_per_writer]
			else:
				gen_signatures = ["original_{0}_{1}.png".format(ix_writer,i) for i in gen_sigs_per_writer]
				forg_signatures = ["forgeries_{0}_{1}.png".format(ix_writer,i) for i in forg_sigs_per_writer]

			#Generating pairs (genuine / forged)
			for f_sig in forg_signatures:
				for g_sig in gen_signatures:
					line = os.path.join(os.path.basename(image_dir_genuine), g_sig) + " " + os.path.join(os.path.basename(image_dir_forgery), f_sig) + " 0"
					writer_lines.append(line)

			#Generating pairs (genuine / genuine)
			for ix, g_sig in enumerate(gen_signatures):

				for i in range(ix+1, len(gen_sigs_per_writer)):
					line = os.path.join(os.path.basename(image_dir_genuine), gen_signatures[i]) + " " + os.path.join(os.path.basename(image_dir_genuine), g_sig) + " 1"
					writer_lines.append(line)

			#Concatenating previous generated lines with the lines for the current writer
			dataset_lines += writer_lines

		#Writing lines to a txt file for CEDAR pairs
		with open(pairs_file, 'w') as filehandle:
			filehandle.writelines("%s\n" % line for line in dataset_lines)
	elif dataset == "MCYT":
		num_writers = 75
		gen_sig_per_writer = 15
		forg_sig_per_writer = 15

		dataset_lines = []

		writers_dir = [os.path.basename(i[0]) for i in os.walk("../master-thesis/datasets/MCYT")]
		writers_dir = [i for i in writers_dir if i != "MCYT"]

		for ix, writer_dir in enumerate(writers_dir):
			gen_signatures = []
			forg_signatures = []
			absolute_writer_dir = os.path.join("../master-thesis/datasets/MCYT", writer_dir)

			for root, dirs, files in os.walk(absolute_writer_dir):
				gen_signatures = [file for file in files if "v" in file.lower()]
				forg_signatures = [file for file in files if "f" in file]

			writer_lines = []

			#Generating pairs (genuine / forged)
			for f_sig in forg_signatures:
				for g_sig in gen_signatures:
					line = os.path.join(writer_dir, g_sig) + " " + os.path.join(writer_dir, f_sig) + " 0"
					writer_lines.append(line)

			#Generating pairs (genuine / genuine)
			for ix, g_sig in enumerate(gen_signatures):
				for i in range(ix+1, len(gen_signatures)):
					line = os.path.join(writer_dir, gen_signatures[i]) + " " + os.path.join(writer_dir, g_sig) + " 1"
					writer_lines.append(line)

			#Concatenating previous generated lines with the lines for the current writer
			dataset_lines += writer_lines

		#Writing lines to a txt file for CEDAR pairs
		with open(pairs_file, 'w') as filehandle:
			filehandle.writelines("%s\n" % line for line in dataset_lines)

	elif dataset == "GPDS":
		num_writers = 4000
		gen_sig_per_writer = 24
		forg_sig_per_writer = 30

		dataset_lines = []

		writers_dir = ["{:03d}".format(i) for i in range(1, num_writers+1)]
		gen_sigs_per_writer = list(range(1, gen_sig_per_writer+1))
		forg_sigs_per_writer = list(range(1, forg_sig_per_writer+1))

		#Generating pairs for each writer
		for ix, writer_dir in enumerate(writers_dir):
			#Generating list of genuine and forgeries image names
			gen_signatures = ["c-{:03d}-{:02d}.jpg".format(ix+1, i) for i in gen_sigs_per_writer]
			forg_signatures = ["cf-{:03d}-{:02d}.jpg".format(ix+1, i) for i in forg_sigs_per_writer]

			writer_lines = []

			#Generating pairs (genuine / forged)
			for f_sig in forg_signatures:
				for g_sig in gen_signatures:
					line = os.path.join(writer_dir, g_sig) + " " + os.path.join(writer_dir, f_sig) + " 0"
					writer_lines.append(line)

			#Generating pairs (genuine / genuine)
			for ix, g_sig in enumerate(gen_signatures):
				for i in range(ix+1, len(gen_sigs_per_writer)):
					line = os.path.join(writer_dir, gen_signatures[i]) + " " + os.path.join(writer_dir, g_sig) + " 1"
					writer_lines.append(line)

			#Concatenating previous generated lines with the lines for the current writer
			dataset_lines += writer_lines

		#Writing lines to a txt file for CEDAR pairs
		with open(pairs_file, 'w') as filehandle:
			filehandle.writelines("%s\n" % line for line in dataset_lines)


if __name__ == "__main__":
	main()
