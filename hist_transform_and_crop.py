import click
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os

@click.command()
@click.option('--dataset', help="dataset which will be transformed")
def main():
    perform_transform(dataset)

def perform_transform(dataset):
    if dataset == "MCYT":
        ############# MCYT #############
        writers_dir = [os.path.basename(i[0]) for i in os.walk("../master-thesis/datasets/MCYT")]
        writers_dir = [i for i in writers_dir if i != "MCYT"]

        for writer_dir in writers_dir:
            absolute_writer_dir = os.path.join("../master-thesis/datasets/MCYT", writer_dir)
            os.system("mkdir -p {}".format(absolute_writer_dir.replace("MCYT", "MCYT_UNBIASED")))
            for root, dirs, files in os.walk(absolute_writer_dir):
                for file in files:
                    if os.path.splitext(file)[1] == ".bmp":
                        org_file = os.path.join(absolute_writer_dir, file)
                        print(org_file)
                        new_file = org_file.replace("MCYT", "MCYT_UNBIASED")
                        #print("bash code/versigoff/trim.sh {} {}".format(org_file, new_file))
                        os.system("bash code/versigoff/trim.sh {} {}".format(org_file, new_file))

                        #Applying histogram transformation (to remove background)
                        img = cv.imread(new_file, cv.IMREAD_GRAYSCALE)
                        img = np.clip(1.23*(img - 0.4) + 0.35, 0, 255)

                        cv.imwrite(new_file, img)

if __name__ == "__main__":
    main()