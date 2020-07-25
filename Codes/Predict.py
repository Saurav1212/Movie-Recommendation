import os, argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from prettytable import PrettyTable
from tensorflow.keras.models import load_model
from Utilities import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.keras.backend.set_floatx('float64')

def get_metrics(bdir, model, testds, batch_size):
	print ("--> Getting Metrics")
	filelendf = pd.read_csv(bdir + 'Dataset/file_length.dat', engine = 'python', sep = ':', index_col = 0)
	rmse, mae = 0.0, 0.0
	for subset in testds:
		predictions = model.predict(subset)
		rmse += numpy_rmse(subset[0][1].numpy(), predictions, subset[0][2].numpy(), subset[0][3].numpy())
		mae += numpy_mae(subset[0][1].numpy(), predictions, subset[0][2].numpy(), subset[0][3].numpy())
	print ("RMSE: ", rmse / filelendf.loc['Test']['Length'])
	print ("MAE: ", mae / filelendf.loc['Test']['Length'])

def get_topn_movies(bdir, model, testds, n):
	print ("--> Top {n} Movies".format(n = n))
	keymoviedf = pd.read_csv(bdir + 'Dataset/key_movie.dat', engine = 'python', sep = ':', index_col = 0)
	for subset in testds:
		predictions = model.predict(subset)
		for prediction in predictions:
			indices = np.argsort(prediction)
			top_n = indices[:-n:-1]
			t = PrettyTable(['Rank', 'Movie', 'Predicted Rating'])
			for i in range(len(top_n)):
				t.add_row([i + 1, str(keymoviedf.loc[top_n[i]]['Title']), "{:.2f}".format(prediction[top_n[i]])])
			print (t)

def main(bdir, batch_size, option, n):
	print ("***Predict.py***")
	model = load_model(bdir + 'model.h5', custom_objects = {'rmse': rmse, 'mae': mae})
	inp = open(bdir + 'Dataset/movie_num.txt', 'r')
	movie_num = int(inp.read())
	inp.close()
	testds = dataset_input(bdir, 'Test', movie_num, batch_size)
	if option != 2:
		get_metrics(bdir, model, testds, batch_size)
	if option != 1:
		get_topn_movies(bdir, model, testds, n)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = "Options")
	parser.add_argument("-jb", "--job_dir", type = str, default = "", help = "Base Directory")
	parser.add_argument("-bs", "--batch_size", type = int, default = 256, help = "Batch Size")
	parser.add_argument("-o", "--option", type = int, default = 3, choices = [1, 2, 3], help = "1 -> Metrics only, 2 -> Top N Movies only, 3 -> Both")
	parser.add_argument("-n", "--n_movies", type = int, default = 10, help = "Number of movies to be predicted for each user")
	args = parser.parse_args()
	main(args.job_dir, args.batch_size, args.option, args.n_movies)