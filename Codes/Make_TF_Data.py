import os, argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from numpy import argmax
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import StratifiedShuffleSplit

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
mode = 0
file_length = dict()

def get_data(bdir):
	print ("--> Getting Data")
	ratingsdf = pd.read_csv(bdir + 'Dataset/ratings.dat', engine = "python", sep = "::", names = ['UserID', 'MovieID', 'Rating', 'Timestamp'], usecols = ['UserID', 'MovieID', 'Rating'])
	moviedf = pd.read_csv(bdir + 'Dataset/movies.dat', engine = "python", sep = "::", names = ['MovieID', 'Title', 'Genre'], index_col = 'MovieID', usecols = ['MovieID', 'Title'])
	return ratingsdf, moviedf

def process(bdir, ratingsdf, moviedf):
	print ("--> Processing Data")

	matrix = ratingsdf.pivot_table(values = 'Rating', index = 'UserID', columns = 'MovieID', fill_value = 0.0)
	matrix = matrix.astype('float32')
	User = matrix.T.to_dict('list')

	movies = list(set(ratingsdf['MovieID']))
	with open(bdir + 'Dataset/movie_num.txt', 'w') as out:
		out.write(str(len(movies)))
	lb = LabelBinarizer()
	y = lb.fit_transform(movies)
	Movie = dict(zip(movies, y))

	moviedf = moviedf.to_dict('dict')['Title']
	Movie_rev = {k:moviedf[movies[k]] for k in range(len(movies))}
	pd.DataFrame.from_dict(Movie_rev, orient = "index").to_csv(bdir + 'Dataset/key_movie.dat', sep = ':', header = ['Title'])
	return User, Movie

def tf_writer(dataset, bdir, filename, User, Movie):
	print ("--> Writing " + filename + ".tfrecords")
	file_length[filename] = len(dataset)
	writer = tf.io.TFRecordWriter(bdir + 'TfRecords/' + filename + '.tfrecords')
	for index, data in dataset.iterrows():
		Ip = User[data['UserID']][:]
		Ip[argmax(Movie[data['MovieID']])] = 0.0
		if mode == 1:
			Op = User[data['UserID']][:]
		elif mode == 2:
			Op = Movie[data['MovieID']][:]
			Op[argmax(Movie[data['MovieID']])] = data['Rating']
		else:
			print ("Invalid mode")
		feature = {'Input' : tf.train.Feature(float_list = tf.train.FloatList(value = Ip)), 'Output': tf.train.Feature(float_list = tf.train.FloatList(value = Op))}
		example = tf.train.Example(features = tf.train.Features(feature = feature))
		writer.write(example.SerializeToString())

def tf_writer_prep(bdir, ratingsdf, User, Movie):
	print ("--> Preparing Data")
	sss = StratifiedShuffleSplit(n_splits = 5, test_size = 0.2, random_state = 12)
	for train_index, valtest_index in sss.split(np.zeros(len(ratingsdf)), ratingsdf['Rating']):
		pass
	sss = StratifiedShuffleSplit(n_splits = 5, test_size = 0.5, random_state = 12)
	for val_index, test_index in sss.split(np.zeros(len(valtest_index)), ratingsdf['Rating'].iloc[valtest_index]):
		pass
	os.system("rm -r " + bdir + "TfRecords")
	tf_writer(ratingsdf.iloc[train_index], bdir, 'Train', User, Movie)
	tf_writer(ratingsdf.iloc[val_index], bdir, 'Validation', User, Movie)
	tf_writer(ratingsdf.iloc[test_index], bdir, 'Test', User, Movie)
	pd.DataFrame.from_dict(file_length, orient = "index").to_csv(bdir + 'Dataset/file_length.dat', sep = ':', header = ['Length'])

def main(mode, bdir):
	globals()['mode'] = mode
	print ("***Make_TF_Data.py (Mode: {mode})***".format(mode = mode))
	ratingsdf, moviedf = get_data(bdir)
	User, Movie = process(bdir, ratingsdf, moviedf)
	tf_writer_prep(bdir, ratingsdf, User, Movie)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = "Options")
	parser.add_argument("-m", "--mode", type = int, default = 1, choices = [1, 2], help = "1 -> All known loss, 2 -> Single withheld loss")
	parser.add_argument("-jb", "--job_dir", type = str, default = "", help = "Base Directory")
	args = parser.parse_args()
	main(args.mode, args.job_dir)