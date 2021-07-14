import feature_generation as fg

def main():
	img_height, img_width, height, width, histograms = fg.get_r2_histogram("../master-thesis/datasets/CEDAR/full_org_hist_transform/original_1_1.png", [6,11,16,21,26])
	print(img_height, img_width, height, width, histograms)

if __name__ == "__main__":
	main()