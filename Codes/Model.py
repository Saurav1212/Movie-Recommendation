import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, argparse
import pandas as pd
from math import floor, log
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from Utilities import mae, rmse, extract_fn, dataset_input

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.keras.backend.set_floatx('float64')

def layer_creator(n, x):
	if n == 512:
		x = Dense(512, activation = "relu")(x)
	else:
		x = Dense(n, activation = "relu")(x)
		x = layer_creator(n/2, x)
		x = Dense(n, activation = "relu")(x)
	return x

def model(train_ds, val_ds, bdir, movie_num, epochs, batch_size, learning_rate, patience):
	os.system("rm " + bdir + "log.csv")
	filelendf = pd.read_csv(bdir + 'Dataset/file_length.dat', engine = 'python', sep = ':', index_col = 0)
	print ("--> Starting Training with learning rate = {learning_rate} for epochs = {epochs} with patience = {patience}".format(learning_rate = learning_rate, epochs = epochs, patience = patience))
	adam = Adam(learning_rate = learning_rate)

	Ip = Input(shape = (movie_num, ), name = "Input")
	Op = Input(shape = (movie_num, ), name = "Target")
	Weight = Input(shape = (movie_num, ), name = "Weight")
	Count = Input(shape = (1, ), name = "Count")

	n = pow(2, floor(log(movie_num)/log(2)))
	if n < 512:
		print ("Insufficient number of movies for a good model")
		exit()
	else:
		x = layer_creator(n, Ip)

	Output = Dense(movie_num, activation = "relu", name = "Output")(x)

	model = Model(inputs = [Ip, Op, Weight, Count], outputs = Output)
	model.add_loss(rmse(Op, Output, Weight, Count))
	model.add_metric(mae(Op, Output, Weight, Count), aggregation = 'mean', name = 'mae')
	model.compile(optimizer = adam, loss = None, metrics = None)
	#print (model.summary())

	es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = patience)
	cl = CSVLogger(bdir + 'log.csv', append = True, separator = ',')
	mc = ModelCheckpoint(bdir + 'model.h5', monitor = 'val_loss', verbose = 1, save_best_only = True)

	history = model.fit(train_ds, epochs = epochs, steps_per_epoch = (filelendf.loc['Train']['Length'] // batch_size), validation_data = val_ds, callbacks = [es, cl, mc])

	print ("--> Plotting Loss")
	#print(history.history.keys())
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'val'], loc = 'upper left')
	plt.savefig(bdir + 'loss.png', bbox_inches = 'tight')

def get_dataset(bdir, movie_num, batch_size):
	print ("--> Getting TfRecords")
	train_ds = dataset_input(bdir, 'Train', movie_num, batch_size)
	val_ds = dataset_input(bdir, 'Validation', movie_num, batch_size)
	return train_ds, val_ds

def main(bdir, epochs, batch_size, learning_rate, patience):
	print ("***Model.py***")
	inp = open(bdir + 'Dataset/movie_num.txt', 'r')
	movie_num = int(inp.read())
	inp.close()
	train_ds, val_ds = get_dataset(bdir, movie_num, batch_size)
	model(train_ds, val_ds, bdir, movie_num, epochs, batch_size, learning_rate, patience)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = "Options")
	parser.add_argument("-jb", "--job_dir", type = str, default = "", help = "Base Directory")
	parser.add_argument("-e", "--epochs", type = int, default = 1000, help = "Number of epochs")
	parser.add_argument("-bs", "--batch_size", type = int, default = 256, help = "Batch Size")
	parser.add_argument("-lr", "--learning_rate", type = float, default = 0.001, help = "Learning Rate for Adam")
	parser.add_argument("-p", "--patience", type = int, default = 50, help = "Early Stopping Patience")
	args = parser.parse_args()
	main(args.job_dir, args.epochs, args.batch_size, args.learning_rate, args.patience)