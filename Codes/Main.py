import os, argparse
import Make_TF_Data, Model, Predict, Utilities

def main(args):
	if args.strtopt == 'a' or input ("Get MovieLens Dataset? (y/n): ") == 'y':
		os.system("chmod +x " + args.job_dir + "Codes/Get_Dataset.sh")
		os.system("./" + args.job_dir + "Codes/Get_Dataset.sh")
	if args.strtopt == 'a' or input("Run Make_TF_Data.py? (y/n): ") == 'y':
		Make_TF_Data.main(args.mode, args.job_dir)
	if args.strtopt == 'a' or input("Run Model.py? (y/n): ") == 'y':
		Model.main(args.job_dir, args.epochs, args.batch_size, args.learning_rate, args.patience)
	if args.strtopt == 'a' or input("Run Predict.py? (y/n): ") == 'y':
		Predict.main(args.job_dir, args.batch_size, args.option, args.n_movies)
	if args.strtopt == 'a' or input ("Run Metrics Tester? (y/n): ") == 'y':
		Utilities.main()

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = "Options")
	parser.add_argument("-m", "--mode", type = int, default = 1, choices = [1, 2], help = "1 -> All known loss, 2 -> Single withheld loss")
	parser.add_argument("-so", "--strtopt", type = str, default = "", choices = ["a", ""], help = "a -> Run All: Make, Build and Predict")
	parser.add_argument("-jb", "--job_dir", type = str, default = "", help = "Base Directory")
	parser.add_argument("-e", "--epochs", type = int, default = 1000, help = "Number of epochs")
	parser.add_argument("-bs", "--batch_size", type = int, default = 256, help = "Batch Size")
	parser.add_argument("-lr", "--learning_rate", type = float, default = 0.001, help = "Learning Rate for Adam")
	parser.add_argument("-p", "--patience", type = int, default = 50, help = "Early Stopping Patience")
	parser.add_argument("-o", "--option", type = int, default = 3, choices = [1, 2, 3], help = "1 -> Metrics only, 2 -> Top N Movies only, 3 -> Both")
	parser.add_argument("-n", "--n_movies", type = int, default = 10, help = "Number of movies to be predicted for each user")
	args = parser.parse_args()
	main(args)