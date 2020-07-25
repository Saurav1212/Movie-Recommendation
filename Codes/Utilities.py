import os
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def mae(y_true, y_pred, weights, count):
	return tf.math.reduce_mean(tf.math.divide_no_nan(tf.math.reduce_sum(tf.abs((y_true - y_pred) * weights), axis = -1), tf.reshape(count, [-1])))

def rmse(y_true, y_pred, weights, count):
	return tf.math.reduce_mean(tf.math.sqrt(tf.math.divide_no_nan(tf.math.reduce_sum(tf.square((y_true - y_pred) * weights), axis = -1), tf.reshape(count, [-1]))))

def numpy_mae(y_true, y_pred, weights, count):
	return np.sum(np.divide(np.sum(np.abs((y_true - y_pred) * weights), axis = -1), count))

def numpy_rmse(y_true, y_pred, weights, count):
	return np.sum(np.sqrt(np.divide(np.sum(np.square((y_true - y_pred) * weights), axis = -1), count)))

def extract_fn(data_record, movie_num):
	features = {'Input': tf.io.FixedLenFeature([movie_num], tf.float32), 'Output': tf.io.FixedLenFeature([movie_num], tf.float32)}
	sample = tf.io.parse_single_example(data_record, features)
	Ip = sample['Input']
	Op = sample['Output']
	zeros = tf.zeros_like(Op)
	mask = tf.not_equal(Op, zeros)
	Weight = tf.where(mask, tf.fill(Op.shape, 1.0), Op)
	Count = tf.cast(tf.math.count_nonzero(Op), tf.float32)
	return (Ip, Op, Weight, Count), Op

def dataset_input(bdir, fname, movie_num, batch_size):
	inp = open(bdir + 'Dataset/movie_num.txt', 'r')
	movie_num = int(inp.read())
	inp.close()
	dataset = tf.data.TFRecordDataset([bdir + 'TfRecords/' + fname + '.tfrecords'])
	dataset = dataset.map(lambda x: extract_fn(x, movie_num))
	if fname == 'Train':
		dataset = dataset.shuffle(batch_size * 10)
		dataset = dataset.repeat()
	dataset = dataset.batch(batch_size)
	return dataset

def numpy_metrics_tester():
	print ("***Numpy Metrics Tester***")
	a = np.array([[1, 3, 0], [0, 2, 5]])
	b = np.array([[3, 1, 4], [5, 4, 0]])
	c = np.array([[1, 1, 0], [0, 1, 1]])
	d = np.array([[2], [2]])
	print ("Y_true: ", a)
	print ("Y_pred: ", b)
	print ("Weights: ", c)
	print ("Count: ", d)
	print ("Diff: ", a - b)
	print ("Weighted Abs Diff: ", (a - b) * c)
	print ("Square: ", np.square((a - b) * c))
	print ("Sum: ", np.sum(np.square((a - b) * c), axis = -1))
	print ("Mean: ", np.divide(np.sum(np.square((a - b) * c), axis = -1), np.reshape(d, [-1])))
	print ("Sqrt: ", np.sqrt(np.divide(np.sum(np.square((a - b) * c), axis = -1), np.reshape(d, [-1]))))
	print ("RMSE: ", np.sum(np.sqrt(np.divide(np.sum(np.square((a - b) * c), axis = -1), np.reshape(d, [-1])))) / 2)
	print ("Abs: ", np.abs((a - b) * c))
	print ("Sum: ", np.sum(np.abs((a - b) * c), axis = -1))
	print ("Mean: ", np.divide(np.sum(np.abs((a - b) * c), axis = -1), np.reshape(d, [-1])))
	print ("MAE: ", np.sum(np.divide(np.sum(np.abs((a - b) * c), axis = -1), np.reshape(d, [-1]))) / 2)

def tf_metrics_tester():
	print ("***Tensorflow Metrics Tester***")
	a = tf.constant([[1, 0, 0], [0, 2, 0]], dtype = tf.float64)
	b = tf.constant([[3, 1, 4], [5, 4, 0]], dtype = tf.float64)
	c = tf.constant([[1, 0, 0], [0, 1, 0]], dtype = tf.float64)
	d = tf.constant([[1], [1]], dtype = tf.float64)
	print ("Y_true: ", a)
	print ("Y_pred: ", b)
	print ("Weights: ", c)
	print ("Count: ", d)
	print ("Diff: ", a - b)
	print ("Abs Diff: ", tf.abs(a - b))
	print ("Weighted Abs Diff: ", tf.abs(a - b) * c)
	print ("Sum: ", tf.math.reduce_sum(tf.abs(a - b) * c, axis = -1))
	print ("MAE: ", tf.math.divide_no_nan(tf.math.reduce_sum(tf.abs(a - b) * c, axis = -1), tf.reshape(d, [-1])))
	print ("Mean over batch: ", tf.math.reduce_mean(tf.math.divide_no_nan(tf.math.reduce_sum(tf.abs(a - b) * c, axis = -1), tf.reshape(d, [-1]))))
	print ("Square: ", tf.square(a - b))
	print ("Weighted Square: ", tf.square(a - b) * c)
	print ("Sum: ", tf.math.reduce_sum(tf.square(a - b) * c, axis = -1))
	print ("Mean: ", tf.math.divide_no_nan(tf.math.reduce_sum(tf.square(a - b) * c, axis = -1), tf.reshape(d, [-1])))
	print ("RMSE: ", tf.math.sqrt(tf.math.divide_no_nan(tf.math.reduce_sum(tf.square(a - b) * c, axis = -1), tf.reshape(d, [-1]))))
	print ("Mean over bath: ", tf.math.reduce_mean(tf.math.sqrt(tf.math.divide_no_nan(tf.math.reduce_sum(tf.square(a - b) * c, axis = -1), tf.reshape(d, [-1])))))

def main():
	numpy_metrics_tester()
	tf_metrics_tester()

if __name__ == "__main__":
	main()