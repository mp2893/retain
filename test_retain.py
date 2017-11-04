#################################################################
# Code written by Edward Choi (mp2893@gatech.edu)
# For bug report, please contact author using the email address
#################################################################

import sys, random
import numpy as np
import cPickle as pickle
from collections import OrderedDict
import argparse

import theano
import theano.tensor as T
from theano import config

def sigmoid(x):
  return 1. / (1. + np.exp(-x))

def numpy_floatX(data):
	return np.asarray(data, dtype=config.floatX)

def load_embedding(infile):
	Wemb = np.array(pickle.load(open(infile, 'rb'))).astype(config.floatX)
	return Wemb

def load_params(options):
	params = OrderedDict()
	weights = np.load(options['modelFile'])
	for k,v in weights.iteritems():
		params[k] = v
	if len(options['embFile']) > 0: params['W_emb'] = np.array(pickle.load(open(options['embFile'], 'rb'))).astype(config.floatX)
	return params

def init_tparams(params, options):
	tparams = OrderedDict()
	for key, value in params.iteritems():
		tparams[key] = theano.shared(value, name=key)
	return tparams

def _slice(_x, n, dim):
	if _x.ndim == 3:
		return _x[:, :, n*dim:(n+1)*dim]
	return _x[:, n*dim:(n+1)*dim]

def gru_layer(tparams, emb, name, hiddenDimSize):
	timesteps = emb.shape[0]
	if emb.ndim == 3: n_samples = emb.shape[1]
	else: n_samples = 1

	def stepFn(wx, h, U_gru):
		uh = T.dot(h, U_gru)
		r = T.nnet.sigmoid(_slice(wx, 0, hiddenDimSize) + _slice(uh, 0, hiddenDimSize))
		z = T.nnet.sigmoid(_slice(wx, 1, hiddenDimSize) + _slice(uh, 1, hiddenDimSize))
		h_tilde = T.tanh(_slice(wx, 2, hiddenDimSize) + r * _slice(uh, 2, hiddenDimSize))
		h_new = z * h + ((1. - z) * h_tilde)
		return h_new

	Wx = T.dot(emb, tparams['W_gru_'+name]) + tparams['b_gru_'+name]
	results, updates = theano.scan(fn=stepFn, sequences=[Wx], outputs_info=T.alloc(numpy_floatX(0.0), n_samples, hiddenDimSize), non_sequences=[tparams['U_gru_'+name]], name='gru_layer', n_steps=timesteps)

	return results
	
def build_model(tparams, options):
	alphaHiddenDimSize = options['alphaHiddenDimSize']
	betaHiddenDimSize = options['betaHiddenDimSize']

	x = T.tensor3('x', dtype=config.floatX)

	reverse_emb_t = x[::-1]
	reverse_h_a = gru_layer(tparams, reverse_emb_t, 'a', alphaHiddenDimSize)[::-1] * 0.5
	reverse_h_b = gru_layer(tparams, reverse_emb_t, 'b', betaHiddenDimSize)[::-1] * 0.5

	preAlpha = T.dot(reverse_h_a, tparams['w_alpha']) + tparams['b_alpha']
	preAlpha = preAlpha.reshape((preAlpha.shape[0], preAlpha.shape[1]))
	alpha = (T.nnet.softmax(preAlpha.T)).T

	beta = T.tanh(T.dot(reverse_h_b, tparams['W_beta']) + tparams['b_beta'])
	
	return x, alpha, beta

def padMatrixWithTime(seqs, times, options):
	lengths = np.array([len(seq) for seq in seqs]).astype('int32')
	n_samples = len(seqs)
	maxlen = np.max(lengths)

	x = np.zeros((maxlen, n_samples, options['inputDimSize'])).astype(config.floatX)
	t = np.zeros((maxlen, n_samples)).astype(config.floatX)
	for idx, (seq,time) in enumerate(zip(seqs,times)):
		for xvec, subseq in zip(x[:,idx,:], seq):
			xvec[subseq] = 1.
		t[:lengths[idx], idx] = time

	if options['useLogTime']: t = np.log(t + 1.)

	return x, t, lengths

def padMatrixWithoutTime(seqs, options):
	lengths = np.array([len(seq) for seq in seqs]).astype('int32')
	n_samples = len(seqs)
	maxlen = np.max(lengths)

	x = np.zeros((maxlen, n_samples, options['inputDimSize'])).astype(config.floatX)
	for idx, seq in enumerate(seqs):
		for xvec, subseq in zip(x[:,idx,:], seq):
			xvec[subseq] = 1.

	return x, lengths

def load_data_debug(seqFile, labelFile, timeFile=''):
	sequences = np.array(pickle.load(open(seqFile, 'rb')))
	labels = np.array(pickle.load(open(labelFile, 'rb')))
	if len(timeFile) > 0:
		times = np.array(pickle.load(open(timeFile, 'rb')))

	dataSize = len(labels)
	np.random.seed(0)
	ind = np.random.permutation(dataSize)
	nTest = int(0.15 * dataSize)
	nValid = int(0.10 * dataSize)

	test_indices = ind[:nTest]
	valid_indices = ind[nTest:nTest+nValid]
	train_indices = ind[nTest+nValid:]

	train_set_x = sequences[train_indices]
	train_set_y = labels[train_indices]
	test_set_x = sequences[test_indices]
	test_set_y = labels[test_indices]
	valid_set_x = sequences[valid_indices]
	valid_set_y = labels[valid_indices]
	train_set_t = None
	test_set_t = None
	valid_set_t = None

	if len(timeFile) > 0:
		train_set_t = times[train_indices]
		test_set_t = times[test_indices]
		valid_set_t = times[valid_indices]

	def len_argsort(seq):
		return sorted(range(len(seq)), key=lambda x: len(seq[x]))

	train_sorted_index = len_argsort(train_set_x)
	train_set_x = [train_set_x[i] for i in train_sorted_index]
	train_set_y = [train_set_y[i] for i in train_sorted_index]

	valid_sorted_index = len_argsort(valid_set_x)
	valid_set_x = [valid_set_x[i] for i in valid_sorted_index]
	valid_set_y = [valid_set_y[i] for i in valid_sorted_index]

	test_sorted_index = len_argsort(test_set_x)
	test_set_x = [test_set_x[i] for i in test_sorted_index]
	test_set_y = [test_set_y[i] for i in test_sorted_index]

	if len(timeFile) > 0:
		train_set_t = [train_set_t[i] for i in train_sorted_index]
		valid_set_t = [valid_set_t[i] for i in valid_sorted_index]
		test_set_t = [test_set_t[i] for i in test_sorted_index]

	train_set = (train_set_x, train_set_y, train_set_t)
	valid_set = (valid_set_x, valid_set_y, valid_set_t)
	test_set = (test_set_x, test_set_y, test_set_t)

	return train_set, valid_set, test_set

def load_data(dataFile, labelFile, timeFile):
	test_set_x = np.array(pickle.load(open(dataFile, 'rb')))
	test_set_y = np.array(pickle.load(open(labelFile, 'rb')))
	test_set_t = None
	if len(timeFile) > 0:
		test_set_t = np.array(pickle.load(open(timeFile, 'rb')))

	def len_argsort(seq):
		return sorted(range(len(seq)), key=lambda x: len(seq[x]))

	sorted_index = len_argsort(test_set_x)
	test_set_x = [test_set_x[i] for i in sorted_index]
	test_set_y = [test_set_y[i] for i in sorted_index]
	if len(timeFile) > 0:
		test_set_t = [test_set_t[i] for i in sorted_index]
	
	test_set = (test_set_x, test_set_y, test_set_t)

	return test_set

def print2file(buf, outFile):
	outfd = open(outFile, 'a')
	outfd.write(buf + '\n')
	outfd.close()

def train_RETAIN(
	modelFile='model.npz',
	seqFile='seqFile.txt',
	labelFile='labelFile.txt',
	outFile='outFile.txt',
	timeFile='timeFile.txt',
	typeFile='types.txt',
	useLogTime=True,
	embFile='embFile.txt',
	logEps=1e-8
):
	options = locals().copy()

	if len(timeFile) > 0: useTime = True
	else: useTime = False
	options['useTime'] = useTime

	if len(embFile) > 0: useFixedEmb = True
	else: useFixedEmb = False
	options['useFixedEmb'] = useFixedEmb
	
	print 'Loading the parameters ... ',
	params = load_params(options)
	tparams = init_tparams(params, options)

	options['alphaHiddenDimSize'] = params['w_alpha'].shape[0]
	options['betaHiddenDimSize'] = params['W_beta'].shape[0]
	options['inputDimSize'] = params['W_emb'].shape[0]

	print 'Building the model ... ',
	x, alpha, beta =  build_model(tparams, options)
	get_result = theano.function(inputs=[x], outputs=[alpha, beta], name='get_result')

	print 'Loading data ... ',
	testSet = load_data(seqFile, labelFile, timeFile)
	print 'done'

	types = pickle.load(open(typeFile, 'rb'))
	rtypes = dict([(v,k) for k,v in types.iteritems()])

	print 'Contribution calculation start!!'
	count = 0
	outfd = open(outFile, 'w')
	for index in range(len(testSet[0])):
		if count % 100 == 0: print 'processed %d patients' % count
		count += 1
		batchX = [testSet[0][index]]
		label = testSet[1][index]

		if useTime: 
			batchT = [testSet[2][index]]
			x, t, lengths = padMatrixWithTime(batchX, batchT, options)
		else:
			x, lengths = padMatrixWithoutTime(batchX, options)

		n_timesteps = x.shape[0]
		n_samples = x.shape[1]
		emb = np.dot(x, params['W_emb'])

		if useTime:
			temb = np.concatenate([emb, t.reshape((n_timesteps,n_samples,1))], axis=2)
		else:
			temb = emb

		alpha, beta = get_result(temb)
		alpha = alpha[:,0]
		beta = beta[:,0,:]

		ct = (alpha[:,None] * beta * emb[:,0,:]).sum(axis=0)
		y_t = sigmoid(np.dot(ct, params['w_output']) + params['b_output'])

		buf = ''
		patient = batchX[0]
		for i in range(len(patient)):
			visit = patient[i]
			buf += '-------------- visit_index:%d ---------------\n' % i
			for j in range(len(visit)):
				code = visit[j]
				contribution = np.dot(params['w_output'].flatten(), alpha[i] * beta[i] * params['W_emb'][code])
				buf += '%s:%f  ' % (rtypes[code], contribution)
			buf += '\n------------------------------------\n'
		buf += 'patient_index:%d, label:%d, score:%f\n\n' % (index, label, y_t)
		outfd.write(buf + '\n')
	outfd.close()
	
def parse_arguments(parser):
	parser.add_argument('model_file', type=str, metavar='<model_file>', help='The path to the Numpy-compressed file containing the model parameters.')
	parser.add_argument('seq_file', type=str, metavar='<visit_file>', help='The path to the cPickled file containing visit information of patients')
	parser.add_argument('label_file', type=str, metavar='<label_file>', help='The path to the cPickled file containing label information of patients')
	parser.add_argument('type_file', type=str, metavar='<type_file>', help='The path to the cPickled dictionary for mapping medical code strings to integers')
	parser.add_argument('out_file', metavar='<out_file>', help='The path to the output models. The models will be saved after every epoch')
	parser.add_argument('--time_file', type=str, default='', help='The path to the cPickled file containing durations between visits of patients. If you are not using duration information, do not use this option')
	parser.add_argument('--use_log_time', type=int, default=1, choices=[0,1], help='Use logarithm of time duration to dampen the impact of the outliers (0 for false, 1 for true) (default value: 1)')
	parser.add_argument('--embed_file', type=str, default='', help='The path to the cPickled file containing the representation vectors of medical codes. If you are not using medical code representations, do not use this option')
	args = parser.parse_args()
	return args

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	args = parse_arguments(parser)

	train_RETAIN(
		modelFile=args.model_file,
		seqFile=args.seq_file, 
		labelFile=args.label_file, 
		typeFile=args.type_file, 
		outFile=args.out_file, 
		timeFile=args.time_file, 
		useLogTime=args.use_log_time,
		embFile=args.embed_file
	)
