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
@click.option('--transform', default="", help='type of transform applied over the dataset (to get the name), could be: rotation, scale, rotation_scale, trim')
def main(dataset, pairs_file, image_dir_genuine, transform):
	if dataset == "CEDAR":
		num_writers = 55
		gen_sig_per_writer = 24
		forg_sig_per_writer = 24

		dataset_lines = []

		writers = list(range(1, num_writers+1))

		gen_sigs_per_writer = list(range(1, gen_sig_per_writer+1))

		for index, ix_writer in enumerate(writers):
			writer_lines = []

			if transform != "":
				gen_signatures = ["original_{0}_{1}_{2}1.png".format(ix_writer,i, transform) for i in gen_sigs_per_writer]
			else:
				gen_signatures = ["original_{0}_{1}.png".format(ix_writer,i) for i in gen_sigs_per_writer]

			#Genuine-random forgery pair generation
			random_forgers = list(range(1, num_writers+1)) #Generating all writers list
			del random_forgers[index] #removing the current writer

			for i in random_forgers:
				for g_sig in gen_signatures:
					random_signature = random.choice(gen_sigs_per_writer)

					if transform != "":
						f_sig = "original_{0}_{1}_{2}1.png".format(i, random_signature, transform)
					else:
						f_sig = "original_{0}_{1}.png".format(i, random_signature)

					line = os.path.join(os.path.basename(image_dir_genuine), g_sig) + " " + os.path.join(os.path.basename(image_dir_genuine), f_sig) + " 0"
					writer_lines.append(line)

			#Generating pairs (genuine / genuine)
			for ix, g_sig in enumerate(gen_signatures):
				#Genuine-genuine pair generation working as before
				for i in range(ix+1, len(gen_sigs_per_writer)):
					line = os.path.join(os.path.basename(image_dir_genuine), gen_signatures[i]) + " " + os.path.join(os.path.basename(image_dir_genuine), g_sig) + " 1"
					writer_lines.append(line)

			#Concatenating previous generated lines with the lines for the current writer
			dataset_lines += writer_lines

		#Writing lines to a txt file for CEDAR pairs
		with open(pairs_file, 'w') as filehandle:
			filehandle.writelines("%s\n" % line for line in dataset_lines)

	if dataset == "Bengali":
		#Setting number of writers and number of genuine and forged signatures for each writer.
		num_writers = 100
		gen_sig_per_writer = 24
		forg_sig_per_writer = 30

		#Getting writers directories.
		writers_dir = ["{:03d}".format(i) for i in range(1, num_writers+1)]

		dataset_lines = []

		gen_sigs_per_writer = list(range(1, gen_sig_per_writer+1))

		#generating pairs for each writer
		for ix, writer_dir in enumerate(writers_dir):

			if transform == "":
				#Generating list of genuine and forgeries image names
				gen_signatures = ["B-S-{ix_writer}-G-{:02d}.tif".format(i, ix_writer=ix+1) for i in gen_sigs_per_writer]
			elif transform == "trim":
				#Generating list of genuine and forgeries image names
				gen_signatures = ["B-S-{ix_writer}-G-{:02d}_trimmed.png".format(i, ix_writer=ix+1) for i in gen_sigs_per_writer]
			elif transform == "trim_rotation":
				#Generating list of genuine and forgeries image names
				gen_signatures = ["B-S-{ix_writer}-G-{:02d}_trimmed_rotation1.png".format(i, ix_writer=ix+1) for i in gen_sigs_per_writer]
			elif transform == "trim_scale":
				#Generating list of genuine and forgeries image names
				gen_signatures = ["B-S-{ix_writer}-G-{:02d}_trimmed_scale1.png".format(i, ix_writer=ix+1) for i in gen_sigs_per_writer]
			elif transform == "trim_rotation_scale":
				#Generating list of genuine and forgeries image names
				gen_signatures = ["B-S-{ix_writer}-G-{:02d}_trimmed_rotation_scale1.png".format(i, ix_writer=ix+1) for i in gen_sigs_per_writer]
			
			writer_lines = []

			#Genuine-random forgery pair generation
			random_forgers = ["{}".format(i) for i in range(1, num_writers+1)] #Generating all writers list
			del random_forgers[ix] #removing the current writer

			for i in random_forgers:
				for g_sig in gen_signatures:
					random_signature = random.choice(gen_sigs_per_writer)

					if transform == "":
						#Generating list of genuine and forgeries image names
						f_sig = "B-S-{ix_writer}-G-{:02d}.tif".format(random_signature, ix_writer=i)
					elif transform == "trim":
						#Generating list of genuine and forgeries image names
						f_sig = "B-S-{ix_writer}-G-{:02d}_trimmed.png".format(random_signature, ix_writer=i)
					elif transform == "trim_rotation":
						#Generating list of genuine and forgeries image names
						f_sig = "B-S-{ix_writer}-G-{:02d}_trimmed_rotation1.png".format(random_signature, ix_writer=i)
					elif transform == "trim_scale":
						#Generating list of genuine and forgeries image names
						f_sig = "B-S-{ix_writer}-G-{:02d}_trimmed_scale1.png".format(random_signature, ix_writer=i)
					elif transform == "trim_rotation_scale":
						#Generating list of genuine and forgeries image names
						f_sig = "B-S-{ix_writer}-G-{:02d}_trimmed_rotation_scale1.png".format(random_signature, ix_writer=i)

					line = os.path.join(writer_dir, g_sig) + " " + os.path.join("{:03d}".format((int)(i)), f_sig) + " 0"
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

		gen_sigs_per_writer = list(range(1, gen_sig_per_writer+1))

		#Generating pairs for each writer
		for ix, writer_dir in enumerate(writers_dir):
			if ix+1 not in [11,17,18,35,76,87,93,123]:
				if transform == "":
					#Generating list of genuine and forgeries image names
					gen_signatures = ["H-S-{ix_writer}-G-{:02d}.tif".format(i, ix_writer=ix+1) for i in gen_sigs_per_writer]
				elif transform == "trim":
					#Generating list of genuine and forgeries image names
					gen_signatures = ["H-S-{ix_writer}-G-{:02d}_trimmed.png".format(i, ix_writer=ix+1) for i in gen_sigs_per_writer]
				elif transform == "trim_rotation":
					#Generating list of genuine and forgeries image names
					gen_signatures = ["H-S-{ix_writer}-G-{:02d}_trimmed_rotation1.png".format(i, ix_writer=ix+1) for i in gen_sigs_per_writer]
				elif transform == "trim_scale":
					#Generating list of genuine and forgeries image names
					gen_signatures = ["H-S-{ix_writer}-G-{:02d}_trimmed_scale1.png".format(i, ix_writer=ix+1) for i in gen_sigs_per_writer]
				elif transform == "trim_rotation_scale":
					#Generating list of genuine and forgeries image names
					gen_signatures = ["H-S-{ix_writer}-G-{:02d}_trimmed_rotation_scale1.png".format(i, ix_writer=ix+1) for i in gen_sigs_per_writer]
			elif ix+1==123 :
				if transform == "":
					#Generating list of genuine and forgeries image names
					gen_signatures = ["H-S-{ix_writer}-G-{:02d}.tif".format(i, ix_writer=ix+2) for i in gen_sigs_per_writer]
				elif transform == "trim":
					#Generating list of genuine and forgeries image names
					gen_signatures = ["H-S-{ix_writer}-G-{:02d}_trimmed.png".format(i, ix_writer=ix+2) for i in gen_sigs_per_writer]
				elif transform == "trim_rotation":
					#Generating list of genuine and forgeries image names
					gen_signatures = ["H-S-{ix_writer}-G-{:02d}_trimmed_rotation1.png".format(i, ix_writer=ix+2) for i in gen_sigs_per_writer]
				elif transform == "trim_scale":
					#Generating list of genuine and forgeries image names
					gen_signatures = ["H-S-{ix_writer}-G-{:02d}_trimmed_scale1.png".format(i, ix_writer=ix+2) for i in gen_sigs_per_writer]
				elif transform == "trim_rotation_scale":
					#Generating list of genuine and forgeries image names
					gen_signatures = ["H-S-{ix_writer}-G-{:02d}_trimmed_rotation_scale1.png".format(i, ix_writer=ix+2) for i in gen_sigs_per_writer]
			else:
				if transform == "":
					#Generating list of genuine and forgeries image names
					gen_signatures = ["H-S-{:03d}-G-{:02d}.tif".format(ix+1, i) for i in gen_sigs_per_writer]
				elif transform == "trim":
					#Generating list of genuine and forgeries image names
					gen_signatures = ["H-S-{:03d}-G-{:02d}_trimmed.png".format(ix+1, i) for i in gen_sigs_per_writer]
				elif transform == "trim_rotation":
					#Generating list of genuine and forgeries image names
					gen_signatures = ["H-S-{:03d}-G-{:02d}_trimmed_rotation1.png".format(ix+1, i) for i in gen_sigs_per_writer]
				elif transform == "trim_scale":
					#Generating list of genuine and forgeries image names
					gen_signatures = ["H-S-{:03d}-G-{:02d}_trimmed_scale1.png".format(ix+1, i) for i in gen_sigs_per_writer]
				elif transform == "trim_rotation_scale":
					#Generating list of genuine and forgeries image names
					gen_signatures = ["H-S-{:03d}-G-{:02d}_trimmed_rotation_scale1.png".format(ix+1, i) for i in gen_sigs_per_writer]

			writer_lines = []

			#Genuine-random forgery pair generation
			random_forgers = [i for i in range(1, num_writers+1)] #Generating all writers list
			del random_forgers[ix] #removing the current writer

			#Generating pairs (genuine / forged)
			for i in random_forgers:
				for g_sig in gen_signatures:
					random_signature = random.choice(gen_sigs_per_writer)

					if i not in [11,17,18,35,76,87,93,123]:
						if transform == "":
							#Generating list of genuine and forgeries image names
							f_sig = "H-S-{ix_writer}-G-{:02d}.tif".format(random_signature, ix_writer=i)
						elif transform == "trim":
							#Generating list of genuine and forgeries image names
							f_sig = "H-S-{ix_writer}-G-{:02d}_trimmed.png".format(random_signature, ix_writer=i)
						elif transform == "trim_rotation":
							#Generating list of genuine and forgeries image names
							f_sig = "H-S-{ix_writer}-G-{:02d}_trimmed_rotation1.png".format(random_signature, ix_writer=i)
						elif transform == "trim_scale":
							#Generating list of genuine and forgeries image names
							f_sig = "H-S-{ix_writer}-G-{:02d}_trimmed_scale1.png".format(random_signature, ix_writer=i)
						elif transform == "trim_rotation_scale":
							#Generating list of genuine and forgeries image names
							f_sig = "H-S-{ix_writer}-G-{:02d}_trimmed_rotation_scale1.png".format(random_signature, ix_writer=i)
					elif i==123 :
						if transform == "":
							#Generating list of genuine and forgeries image names
							f_sig = "H-S-{ix_writer}-G-{:02d}.tif".format(random_signature, ix_writer=124)
						elif transform == "trim":
							#Generating list of genuine and forgeries image names
							f_sig = "H-S-{ix_writer}-G-{:02d}_trimmed.png".format(random_signature, ix_writer=124)
						elif transform == "trim_rotation":
							#Generating list of genuine and forgeries image names
							f_sig = "H-S-{ix_writer}-G-{:02d}_trimmed_rotation1.png".format(random_signature, ix_writer=124)
						elif transform == "trim_scale":
							#Generating list of genuine and forgeries image names
							f_sig = "H-S-{ix_writer}-G-{:02d}_trimmed_scale1.png".format(random_signature, ix_writer=124)
						elif transform == "trim_rotation_scale":
							#Generating list of genuine and forgeries image names
							f_sig = "H-S-{ix_writer}-G-{:02d}_trimmed_rotation_scale1.png".format(random_signature, ix_writer=124)
					else:
						if transform == "":
							#Generating list of genuine and forgeries image names
							f_sig = "H-S-{:03d}-G-{:02d}.tif".format(i, random_signature)
						elif transform == "trim":
							#Generating list of genuine and forgeries image names
							f_sig = "H-S-{:03d}-G-{:02d}_trimmed.png".format(i, random_signature)
						elif transform == "trim_rotation":
							#Generating list of genuine and forgeries image names
							f_sig = "H-S-{:03d}-G-{:02d}_trimmed_rotation1.png".format(i, random_signature)
						elif transform == "trim_scale":
							#Generating list of genuine and forgeries image names
							f_sig = "H-S-{:03d}-G-{:02d}_trimmed_scale1.png".format(i, random_signature)
						elif transform == "trim_rotation_scale":
							#Generating list of genuine and forgeries image names
							f_sig = "H-S-{:03d}-G-{:02d}_trimmed_rotation_scale1.png".format(i, random_signature)

					line = os.path.join(writer_dir, g_sig) + " " + os.path.join("{:03d}".format(i), f_sig) + " 0"
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

if __name__ == "__main__":
	main()