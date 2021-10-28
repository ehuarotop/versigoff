import utils
import os
import xlsxwriter
import click
import numpy as np

@click.command()
@click.option('--logdir', help="log directory")
@click.option('--method', default="versigoff", help="which classifier was used")
@click.option('--file', default="results.xlsx", help="filename where information will be saved")
def main(logdir, method, file):
	logs = utils.get_log_files_list(logdir)

	logs = sorted(logs)

	#Creating excel file to save this results
	workbook = xlsxwriter.Workbook(os.path.join(logdir, file))
	worksheet = workbook.add_worksheet()

	headers = ["logfile", "accuracy", "std", "F1-score", "std", "precision", "std", "recall", "std", "ROC_AUC", "std"]

	col=0
	for header in headers:
		worksheet.write(0, col, header)
		col += 1 

	row = 1

	for log in logs:
		col = 0
		lines = utils.readTxtFile(log)

		results = []

		if method == "versigoff":
			for line in lines:
				if "Global" in line:
					metric_std = line.split(":")[1].lstrip(' ')
					metric = (float)(metric_std.split("(+/-")[0].strip(" "))
					std = (float)(metric_std.split("(+/-")[1].split(")")[0].strip(" "))
					results.append(metric)
					results.append(std)

				if "Execution time:" in line:
					seconds = (float)(line.split("Execution time: ")[1].split(" second")[0])
					results.append(seconds)
		elif method == "signet":
			accs = []
			for line in lines:
				if "Accuracy on test set: " in line:
					accuracy = (float)(line.split("Accuracy on test set: ")[1].split("%")[0])
					accs.append(accuracy)

			#Adding accuracy and std to the current line
			results.append(round(np.mean(np.array(accs)), 4))
			results.append(round(np.std(np.array(accs)), 4))

		results = [os.path.basename(log)] + results

		print(results)

		for result in results:
			worksheet.write(row, col, result)
			col += 1

		row += 1

	workbook.close()

if __name__ == "__main__":
	main()
