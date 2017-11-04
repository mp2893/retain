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
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from sklearn.metrics import roc_auc_score

_TEST_RATIO = 0.2
_VALIDATION_RATIO = 0.1

def unzip(zipped):
	new_params = OrderedDict()
	for key, value in zipped.iteritems():
		new_params[key] = value.get_value()
	return new_params

def numpy_floatX(data):
	return np.asarray(data, dtype=config.floatX)

def get_random_weight(dim1, dim2, left=-0.1, right=0.1):
	return np.random.uniform(left, right, (dim1, dim2)).astype(config.floatX)

def load_embedding(infile):
	Wemb = np.array(pickle.load(open(infile, 'rb'))).astype(config.floatX)
	return Wemb

def init_params(options):
	params = OrderedDict()
	timeFile = options['timeFile']
	embFile = options['embFile']
	embDimSize = options['embDimSize']
	inputDimSize = options['inputDimSize']
	alphaHiddenDimSize= options['alphaHiddenDimSize']
	betaHiddenDimSize= options['betaHiddenDimSize']
	numClass = options['numClass']

	if len(embFile) > 0: 
		print 'using external code embedding'
		params['W_emb'] = load_embedding(embFile)
		embDimSize = params['W_emb'].shape[1]
	else: 
		print 'using randomly initialized code embedding'
		params['W_emb'] = get_random_weight(inputDimSize, embDimSize)

	gruInputDimSize = embDimSize
	if len(timeFile) > 0: gruInputDimSize = embDimSize + 1

	params['W_gru_a'] = get_random_weight(gruInputDimSize, 3*alphaHiddenDimSize)
	params['U_gru_a'] = get_random_weight(alphaHiddenDimSize, 3*alphaHiddenDimSize)
	params['b_gru_a'] = np.zeros(3 * alphaHiddenDimSize).astype(config.floatX)

	params['W_gru_b'] = get_random_weight(gruInputDimSize, 3*betaHiddenDimSize)
	params['U_gru_b'] = get_random_weight(betaHiddenDimSize, 3*betaHiddenDimSize)
	params['b_gru_b'] = np.zeros(3 * betaHiddenDimSize).astype(config.floatX)

	params['w_alpha'] = get_random_weight(alphaHiddenDimSize, 1)
	params['b_alpha'] = np.zeros(1).astype(config.floatX)
	params['W_beta'] = get_random_weight(betaHiddenDimSize, embDimSize)
	params['b_beta'] = np.zeros(embDimSize).astype(config.floatX)
	params['w_output'] = get_random_weight(embDimSize, numClass)
	params['b_output'] = np.zeros(numClass).astype(config.floatX)

	return params

def load_params(options):
	return np.load(options['modelFile'])

def init_tparams(params, options):
	tparams = OrderedDict()
	for key, value in params.iteritems():
		if not options['embFineTune'] and key == 'W_emb': continue
		tparams[key] = theano.shared(value, name=key)
	return tparams

def dropout_layer(state_before, use_noise, trng, keep_prob=0.5):
	proj = T.switch(
                use_noise,
                state_before * trng.binomial(state_before.shape, p=keep_prob, n=1, dtype=state_before.dtype) / keep_prob,
                state_before)
	return proj

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
	
def build_model(tparams, options, W_emb=None):
	keep_prob_emb = options['keepProbEmb']
	keep_prob_context = options['keepProbContext']
	alphaHiddenDimSize = options['alphaHiddenDimSize']
	betaHiddenDimSize = options['betaHiddenDimSize']

	trng = RandomStreams(1234)
	use_noise = theano.shared(numpy_floatX(0.))
	useTime = options['useTime']

	x = T.tensor3('x', dtype=config.floatX)
	t = T.matrix('t', dtype=config.floatX)
	y = T.vector('y', dtype=config.floatX)
	lengths = T.ivector('lengths')

	n_timesteps = x.shape[0]
	n_samples = x.shape[1]

	if options['embFineTune']: emb = T.dot(x, tparams['W_emb'])
	else: emb = T.dot(x, W_emb)

	if keep_prob_emb < 1.0: emb = dropout_layer(emb, use_noise, trng, keep_prob_emb)

	if useTime: temb = T.concatenate([emb, t.reshape([n_timesteps,n_samples,1])], axis=2) #Adding the time element to the embedding
	else: temb = emb
	
	def attentionStep(att_timesteps):
		reverse_emb_t = temb[:att_timesteps][::-1]
		reverse_h_a = gru_layer(tparams, reverse_emb_t, 'a', alphaHiddenDimSize)[::-1] * 0.5
		reverse_h_b = gru_layer(tparams, reverse_emb_t, 'b', betaHiddenDimSize)[::-1] * 0.5

		preAlpha = T.dot(reverse_h_a, tparams['w_alpha']) + tparams['b_alpha']
		preAlpha = preAlpha.reshape((preAlpha.shape[0], preAlpha.shape[1]))
		alpha = (T.nnet.softmax(preAlpha.T)).T

		beta = T.tanh(T.dot(reverse_h_b, tparams['W_beta']) + tparams['b_beta'])
		c_t = (alpha[:,:,None] * beta * emb[:att_timesteps]).sum(axis=0)
		return c_t

	counts = T.arange(n_timesteps)+ 1
	c_t, updates = theano.scan(fn=attentionStep, sequences=[counts], outputs_info=None, name='attention_layer', n_steps=n_timesteps)
        if keep_prob_context < 1.0: c_t = dropout_layer(c_t, use_noise, trng, keep_prob_context)

	preY = T.nnet.sigmoid(T.dot(c_t, tparams['w_output']) + tparams['b_output'])
	preY = preY.reshape((preY.shape[0], preY.shape[1]))
	indexRow = T.arange(n_samples)
	y_hat = preY.T[indexRow, lengths - 1]

	logEps = options['logEps']
	cross_entropy = -(y * T.log(y_hat + logEps) + (1. - y) * T.log(1. - y_hat + logEps))
	cost_noreg = T.mean(cross_entropy)

	cost = cost_noreg + options['L2_output'] * (tparams['w_output']**2).sum() + options['L2_alpha'] * (tparams['w_alpha']**2).sum() + options['L2_beta'] * (tparams['W_beta']**2).sum()
	
	if options['embFineTune']: cost += options['L2_emb'] * (tparams['W_emb']**2).sum()

	if useTime: return use_noise, x, y, t, lengths, cost_noreg, cost, y_hat
	else: return use_noise, x, y, lengths, cost_noreg, cost, y_hat

def adadelta(tparams, grads, x, y, lengths, cost, options, t=None):
	zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_grad' % k) for k, p in tparams.iteritems()]
	running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_rup2' % k) for k, p in tparams.iteritems()]
	running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_rgrad2' % k) for k, p in tparams.iteritems()]

	zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
	rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) for rg2, g in zip(running_grads2, grads)]

	if len(options['timeFile']) > 0:
		f_grad_shared = theano.function([x, y, t, lengths], cost, updates=zgup + rg2up, name='adadelta_f_grad_shared')
	else:
		f_grad_shared = theano.function([x, y, lengths], cost, updates=zgup + rg2up, name='adadelta_f_grad_shared')

	updir = [-T.sqrt(ru2 + 1e-6) / T.sqrt(rg2 + 1e-6) * zg for zg, ru2, rg2 in zip(zipped_grads, running_up2, running_grads2)]
	ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2)) for ru2, ud in zip(running_up2, updir)]
	param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

	f_update = theano.function([], [], updates=ru2up + param_up, on_unused_input='ignore', name='adadelta_f_update')

	return f_grad_shared, f_update

def adam(cost, tparams, lr=0.0002, b1=0.1, b2=0.001, e=1e-8):
	updates = []
	grads = T.grad(cost, wrt=tparams.values())
	i = theano.shared(numpy_floatX(0.))
	i_t = i + 1.
	fix1 = 1. - (1. - b1)**i_t
	fix2 = 1. - (1. - b2)**i_t
	lr_t = lr * (T.sqrt(fix2) / fix1)
	for p, g in zip(tparams.values(), grads):
		m = theano.shared(p.get_value() * 0.)
		v = theano.shared(p.get_value() * 0.)
		m_t = (b1 * g) + ((1. - b1) * m)
		v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
		g_t = m_t / (T.sqrt(v_t) + e)
		p_t = p - (lr_t * g_t)
		updates.append((m, m_t))
		updates.append((v, v_t))
		updates.append((p, p_t))
	updates.append((i, i_t))
	return updates

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

def load_data_simple(seqFile, labelFile, timeFile=''):
	sequences = np.array(pickle.load(open(seqFile, 'rb')))
	labels = np.array(pickle.load(open(labelFile, 'rb')))
	if len(timeFile) > 0:
		times = np.array(pickle.load(open(timeFile, 'rb')))

	dataSize = len(labels)
	np.random.seed(0)
	ind = np.random.permutation(dataSize)
	nTest = int(_TEST_RATIO * dataSize)
	nValid = int(_VALIDATION_RATIO * dataSize)

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


def load_data(seqFile, labelFile, timeFile):
	train_set_x = pickle.load(open(seqFile+'.train', 'rb'))
	valid_set_x = pickle.load(open(seqFile+'.valid', 'rb'))
	test_set_x = pickle.load(open(seqFile+'.test', 'rb'))
	train_set_y = pickle.load(open(labelFile+'.train', 'rb'))
	valid_set_y = pickle.load(open(labelFile+'.valid', 'rb'))
	test_set_y = pickle.load(open(labelFile+'.test', 'rb'))
	train_set_t = None
	valid_set_t = None
	test_set_t = None

	if len(timeFile) > 0:
		train_set_t = pickle.load(open(timeFile+'.train', 'rb'))
		valid_set_t = pickle.load(open(timeFile+'.valid', 'rb'))
		test_set_t = pickle.load(open(timeFile+'.test', 'rb'))

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

def calculate_auc(test_model, dataset, options):
	batchSize = options['batchSize']
	useTime = options['useTime']
	
	n_batches = int(np.ceil(float(len(dataset[0])) / float(batchSize)))
	scoreVec = []
	for index in xrange(n_batches):
		batchX = dataset[0][index*batchSize:(index+1)*batchSize]
		if useTime:
			batchT = dataset[2][index*batchSize:(index+1)*batchSize]
			x, t, lengths = padMatrixWithTime(batchX, batchT, options)
			scores = test_model(x, t, lengths)
		else:
			x, lengths = padMatrixWithoutTime(batchX, options)
			scores = test_model(x, lengths)
		scoreVec.extend(list(scores))
	labels = dataset[1]
	auc = roc_auc_score(list(labels), list(scoreVec))
	return auc

def calculate_cost(test_model, dataset, options):
	batchSize = options['batchSize']
	useTime = options['useTime']
	
	costSum = 0.0
	dataCount = 0
	
	n_batches = int(np.ceil(float(len(dataset[0])) / float(batchSize)))
	for index in xrange(n_batches):
		batchX = dataset[0][index*batchSize:(index+1)*batchSize]
		if useTime:
			batchT = dataset[2][index*batchSize:(index+1)*batchSize]
			x, t, lengths = padMatrixWithTime(batchX, batchT, options)
			y = np.array(dataset[1][index*batchSize:(index+1)*batchSize]).astype(config.floatX)
			scores = test_model(x, y, t, lengths)
		else:
			x, lengths = padMatrixWithoutTime(batchX, options)
			y = np.array(dataset[1][index*batchSize:(index+1)*batchSize]).astype(config.floatX)
			scores = test_model(x, y, lengths)
		costSum += scores * len(batchX)
		dataCount += len(batchX)
	return costSum / dataCount

def print2file(buf, outFile):
	outfd = open(outFile, 'a')
	outfd.write(buf + '\n')
	outfd.close()

def train_RETAIN(
	seqFile='seqFile.txt',
	inputDimSize=20000,
	labelFile='labelFile.txt',
	numClass=1,
	outFile='outFile.txt',
	timeFile='',
	modelFile='model.npz',
	useLogTime=True,
	embFile='embFile.txt',
	embDimSize=128,
	embFineTune=True,
	alphaHiddenDimSize=128,
	betaHiddenDimSize=128,
	batchSize=100,
	max_epochs=10,
	L2_output=0.001,
	L2_emb=0.001,
	L2_alpha=0.001,
	L2_beta=0.001,
	keepProbEmb=0.5,
	keepProbContext=0.5,
	logEps=1e-8,
	solver='adadelta',
	simpleLoad=False,
	verbose=False
):
	options = locals().copy()

	if len(timeFile) > 0: useTime = True
	else: useTime = False
	options['useTime'] = useTime
	
	print 'Initializing the parameters ... ',
	params = init_params(options)
	if len(modelFile) > 0: params = load_params(options)
	tparams = init_tparams(params, options)

	print 'Building the model ... ',
	if useTime and embFineTune:
		print 'using time information, fine-tuning code representations'
		use_noise, x, y, t, lengths, cost_noreg, cost, y_hat =  build_model(tparams, options)
		if solver=='adadelta':
			grads = T.grad(cost, wrt=tparams.values())
			f_grad_shared, f_update = adadelta(tparams, grads, x, y, lengths, cost, options, t)
		elif solver=='adam':
			updates = adam(cost, tparams)
			update_model = theano.function(inputs=[x, y, t, lengths], outputs=cost, updates=updates, name='update_model')
		get_prediction = theano.function(inputs=[x, t, lengths], outputs=y_hat, name='get_prediction')
		get_cost = theano.function(inputs=[x, y, t, lengths], outputs=cost_noreg, name='get_cost')
	elif useTime and not embFineTune:
		print 'using time information, not fine-tuning code representations'
		W_emb = theano.shared(params['W_emb'], name='W_emb')
		use_noise, x, y, t, lengths, cost_noreg, cost, y_hat =  build_model(tparams, options, W_emb)
		if solver=='adadelta':
			grads = T.grad(cost, wrt=tparams.values())
			f_grad_shared, f_update = adadelta(tparams, grads, x, y, lengths, cost, options, t)
		elif solver=='adam':
			updates = adam(cost, tparams)
			update_model = theano.function(inputs=[x, y, t, lengths], outputs=cost, updates=updates, name='update_model')
		get_prediction = theano.function(inputs=[x, t, lengths], outputs=y_hat, name='get_prediction')
		get_cost = theano.function(inputs=[x, y, t, lengths], outputs=cost_noreg, name='get_cost')
	elif not useTime and embFineTune:
		print 'not using time information, fine-tuning code representations'
		use_noise, x, y, lengths, cost_noreg, cost, y_hat =  build_model(tparams, options)
		if solver=='adadelta':
			grads = T.grad(cost, wrt=tparams.values())
			f_grad_shared, f_update = adadelta(tparams, grads, x, y, lengths, cost, options)
		elif solver=='adam':
			updates = adam(cost, tparams)
			update_model = theano.function(inputs=[x, y, lengths], outputs=cost, updates=updates, name='update_model')
		get_prediction = theano.function(inputs=[x, lengths], outputs=y_hat, name='get_prediction')
		get_cost = theano.function(inputs=[x, y, lengths], outputs=cost_noreg, name='get_cost')
	elif not useTime and not embFineTune:
		print 'not using time information, not fine-tuning code representations'
		W_emb = theano.shared(params['W_emb'], name='W_emb')
		use_noise, x, y, lengths, cost_noreg, cost, y_hat =  build_model(tparams, options, W_emb)
		if solver=='adadelta':
			grads = T.grad(cost, wrt=tparams.values())
			f_grad_shared, f_update = adadelta(tparams, grads, x, y, lengths, cost, options)
		elif solver=='adam':
			updates = adam(cost, tparams)
			update_model = theano.function(inputs=[x, y, lengths], outputs=cost, updates=updates, name='update_model')
		get_prediction = theano.function(inputs=[x, lengths], outputs=y_hat, name='get_prediction')
		get_cost = theano.function(inputs=[x, y, lengths], outputs=cost_noreg, name='get_cost')

	print 'Loading data ... ',
	if simpleLoad:
		trainSet, validSet, testSet = load_data_simple(seqFile, labelFile, timeFile)
	else:
		trainSet, validSet, testSet = load_data(seqFile, labelFile, timeFile)
	n_batches = int(np.ceil(float(len(trainSet[0])) / float(batchSize)))
	print 'done'

	bestValidAuc = 0.0
	bestTestAuc = 0.0
	bestValidEpoch = 0
	logFile = outFile + '.log'
	print 'Optimization start !!'
	for epoch in xrange(max_epochs):
		iteration = 0
		costVector = []
		for index in random.sample(range(n_batches), n_batches):
			use_noise.set_value(1.)
			batchX = trainSet[0][index*batchSize:(index+1)*batchSize]
			y = np.array(trainSet[1][index*batchSize:(index+1)*batchSize]).astype(config.floatX)

			if useTime:
				batchT = trainSet[2][index*batchSize:(index+1)*batchSize]
				x, t, lengths = padMatrixWithTime(batchX, batchT, options)
				if solver=='adadelta':
					costValue = f_grad_shared(x, y, t, lengths)
					f_update()
				elif solver=='adam':
					costValue = update_model(x, y, t, lengths)
			else:
				x, lengths = padMatrixWithoutTime(batchX, options)
				if solver=='adadelta':
					costValue = f_grad_shared(x, y, lengths)
					f_update()
				elif solver=='adam':
					costValue = update_model(x, y, lengths)
			costVector.append(costValue)
			if (iteration % 10 == 0) and verbose: 
				print 'Epoch:%d, Iteration:%d/%d, Train_Cost:%f' % (epoch, iteration, n_batches, costValue)
			iteration += 1

		use_noise.set_value(0.)
		trainCost = np.mean(costVector)
		validAuc = calculate_auc(get_prediction, validSet, options)
		buf = 'Epoch:%d, Train_cost:%f, Validation_AUC:%f' % (epoch, trainCost, validAuc)
		print buf
		print2file(buf, logFile)
		if validAuc > bestValidAuc: 
			bestValidAuc = validAuc
			bestValidEpoch = epoch
			bestTestAuc = calculate_auc(get_prediction, testSet, options)
			buf = 'Currently the best validation AUC found. Test AUC:%f at epoch:%d' % (bestTestAuc, epoch)
			print buf
			print2file(buf, logFile)
			tempParams = unzip(tparams)
			np.savez_compressed(outFile + '.' + str(epoch), **tempParams)
	buf = 'The best validation & test AUC:%f, %f at epoch:%d' % (bestValidAuc, bestTestAuc, bestValidEpoch)
	print buf
	print2file(buf, logFile)
	
def parse_arguments(parser):
	parser.add_argument('seq_file', type=str, metavar='<visit_file>', help='The path to the Pickled file containing visit information of patients')
	parser.add_argument('n_input_codes', type=int, metavar='<n_input_codes>', help='The number of unique input medical codes')
	parser.add_argument('label_file', type=str, metavar='<label_file>', help='The path to the Pickled file containing label information of patients')
	#parser.add_argument('n_output_codes', type=int, metavar='<n_output_codes>', help='The number of unique label medical codes')
	parser.add_argument('out_file', metavar='<out_file>', help='The path to the output models. The models will be saved after every epoch')
	parser.add_argument('--time_file', type=str, default='', help='The path to the Pickled file containing durations between visits of patients. If you are not using duration information, do not use this option')
	parser.add_argument('--model_file', type=str, default='', help='The path to the Numpy-compressed file containing the model parameters. Use this option if you want to re-train an existing model')
	parser.add_argument('--use_log_time', type=int, default=1, choices=[0,1], help='Use logarithm of time duration to dampen the impact of the outliers (0 for false, 1 for true) (default value: 1)')
	parser.add_argument('--embed_file', type=str, default='', help='The path to the Pickled file containing the representation vectors of medical codes. If you are not using medical code representations, do not use this option')
	parser.add_argument('--embed_size', type=int, default=128, help='The size of the visit embedding. If you are not providing your own medical code vectors, you can specify this value (default value: 128)')
	parser.add_argument('--embed_finetune', type=int, default=1, choices=[0,1], help='If you are using randomly initialized code representations, always use this option. If you are using an external medical code representations, and you want to fine-tune them as you train RETAIN, use this option (0 for false, 1 for true) (default value: 1)')
	parser.add_argument('--alpha_hidden_dim_size', type=int, default=128, help='The size of the hidden layers of the GRU responsible for generating alpha weights (default value: 128)')
	parser.add_argument('--beta_hidden_dim_size', type=int, default=128, help='The size of the hidden layers of the GRU responsible for generating beta weights (default value: 128)')
	parser.add_argument('--batch_size', type=int, default=100, help='The size of a single mini-batch (default value: 100)')
	parser.add_argument('--n_epochs', type=int, default=10, help='The number of training epochs (default value: 10)')
	parser.add_argument('--L2_output', type=float, default=0.001, help='L2 regularization for the final classifier weight w (default value: 0.001)')
	parser.add_argument('--L2_emb', type=float, default=0.001, help='L2 regularization for the input embedding weight W_emb (default value: 0.001)')
	parser.add_argument('--L2_alpha', type=float, default=0.001, help='L2 regularization for the alpha generating weight w_alpha (default value: 0.001).')
	parser.add_argument('--L2_beta', type=float, default=0.001, help='L2 regularization for the input embedding weight W_beta (default value: 0.001)')
	parser.add_argument('--keep_prob_emb', type=float, default=0.5, help='Decides how much you want to keep during the dropout between the embedded input and the alpha & beta generation process (default value: 0.5)')
	parser.add_argument('--keep_prob_context', type=float, default=0.5, help='Decides how much you want to keep during the dropout between the context vector c_i and the final classifier (default value: 0.5)')
	parser.add_argument('--log_eps', type=float, default=1e-8, help='A small value to prevent log(0) (default value: 1e-8)')
	parser.add_argument('--solver', type=str, default='adadelta', choices=['adadelta','adam'], help='Select which solver to train RETAIN: adadelta, or adam. (default: adadelta)')
	parser.add_argument('--simple_load', action='store_true', help='Use an alternative way to load the dataset. Instead of you having to provide a trainign set, validation set, test set, this will automatically divide the dataset. (default false)')
	parser.add_argument('--verbose', action='store_true', help='Print output after every 100 mini-batches (default false)')
	args = parser.parse_args()
	return args

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	args = parse_arguments(parser)

	train_RETAIN(
		seqFile=args.seq_file, 
		inputDimSize=args.n_input_codes, 
		labelFile=args.label_file, 
		#numClass=args.n_output_codes, 
		outFile=args.out_file, 
		timeFile=args.time_file, 
		modelFile=args.model_file,
		useLogTime=args.use_log_time,
		embFile=args.embed_file, 
		embDimSize=args.embed_size, 
		embFineTune=args.embed_finetune, 
		alphaHiddenDimSize=args.alpha_hidden_dim_size,
		betaHiddenDimSize=args.beta_hidden_dim_size,
		batchSize=args.batch_size, 
		max_epochs=args.n_epochs, 
		L2_output=args.L2_output, 
		L2_emb=args.L2_emb, 
		L2_alpha=args.L2_alpha, 
		L2_beta=args.L2_beta, 
		keepProbEmb=args.keep_prob_emb, 
		keepProbContext=args.keep_prob_context, 
		logEps=args.log_eps, 
		solver=args.solver,
		simpleLoad=args.simple_load,
		verbose=args.verbose
	)
