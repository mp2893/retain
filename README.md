RETAIN
=========================================

Doctor AI is a automatic diagnosis machine that predicts medical codes that occur in the next visit, while also predicting the time duration until the next visit.

####Relevant Publications

RETAIN implements an algorithm introduced in the following:

	RETAIN: Interpretable Predictive Model in Healthcare using Reverse Time Attention Mechanism
	Edward Choi, Mohammad Taha Bahadori, Andy Schuetz, Walter F. Stewart, Jimeng Sun
	arXiv preprint arXiv:1511.05942

####Notice

The RETAIN paper formulates the model as being able to make prediction at each timestep (e.g. try to predict what diagnoses the patient will receive at each visit), and treats sequence classification (e.g. Given a patient record, will he be diagnosed with heart failure in the future?) as a special case, since sequence classification makes the prediction at the last timestep only.

This code, however, is implemented to perform the sequence classification task. For example, you can use this code to predict whether the given patient is a heart failure patient or not. Or you can predict whether this patient will be readmitted in the future. The more general version of RETAIN will be released shortly in the future.
	
####Running RETAIN

**STEP 1: Installation**  

1. Install [python](https://www.python.org/), [Theano](http://deeplearning.net/software/theano/index.html). We use Python 2.7, Theano 0.8. Theano can be easily installed in Ubuntu as suggested [here](http://deeplearning.net/software/theano/install_ubuntu.html#install-ubuntu)

2. If you plan to use GPU computation, install [CUDA](https://developer.nvidia.com/cuda-downloads)

3. Download/clone the RETAIN code  

**STEP 2: Preparing training data**  

1. RETAIN's training dataset needs to be a Python cPickled list of list of list. The outermost list corresponds to patients, the intermediate to the visit sequence each patient made, and the innermost to the medical codes (e.g. diagnosis codes, medication codes, procedure codes, etc.) that occurred within each visit.
First, medical codes need to be converted to an integer. Then a single visit can be seen as a list of integers. Then a patient can be seen as a list of visits.
For example, [5,8,15] means the patient was assigned with code 5, 8, and 15 at a certain visit.
If a patient made two visits [1,2,3] and [4,5,6,7], it can be converted to a list of list [[1,2,3], [4,5,6,7]].
Multiple patients can be represented as [[[1,2,3], [4,5,6,7]], [[2,4], [8,3,1], [3]]], which means there are two patients where the first patient made two visits and the second patient made three visits.
This list of list of list needs to be pickled using cPickle. We will refer to this file as the "visit file".

2. The total number of unique medical codes is required to run Doctor AI.
For example, if the dataset is using 14,000 diagnosis codes and 11,000 procedure codes, the total number is 25,000. 

3. The label dataset (let us call this "label file") needs to be a Python cPickled list. Each element corresponds to the true label of each patient. For example, 1 can be the case patient and 0 can be the control patient. If there are two patients where only the first patient is a case, then we should have [1,0].

4. The "visit file" and "label file" need to have 3 sets respectively: training set, validation set, and test set.
The file extension must be ".train", ".valid", and ".test" respectivley.  
For example, if you want to use a file named "my\_visit\_sequences" as the "visit file", then RETAIN will try to load "my\_visit\_sequences.train", "my\_visit\_sequences.valid", and "my\_visit\_sequences.test".  
This is also true for the "label file"

5. You can use the time information regarding the visits as an additional source of information. Let us call this "time file".
Note that the time information could be anything: duration between consecutive visits, cumulative number of days since the first visit, etc.
"time file" needs to be prepared as a Python cPickled list of list. Each list corresponds to patients and the time information of each visit.
For example, given a "visit file" [[[1,2,3], [4,5,6,7]], [[2,4], [8,3,1], [3]]], its corresponding "time file" could look like [[0, 15], [0, 45, 23]], if we are using the duration between the consecutive visits. (of course the numbers are fake, and I've set the duration for the first visit to zero.)
Use `--time\_file` option to use "time file"
Remember that the ".train", ".valid", ".test" rule also applies to the "time file" as well.

**Additional: Using your own medical code representations**  
RETAIN internally learns vector representation of medical codes while training. These vectors are initialized with random values of course.  
You can, however, also provide medical code representations, if you have one. (They can be easily trained by using Skip-gram like algorithms.)
If you want to provide the medical code representations, it has to be a list of list (basically a matrix) of N rows and M columns where N is the number of unique codes in your "visit file" and M is the size of the code representations.
Specify the path to your code representation file using `--embed\_file`.  
Additionally even if you provided your own medical code representations, you can re-train (a.k.a fine-tune) them as you train RETAIN.
Use `--embed\_finetune` option to do this. If you are not providing your own medical code representations, RETAIN will use randomly initialized one, which obviously requires this fine-tuning process. Since the default is to use the fine-tuning, you do not need to worry about this.

**STEP 3: Running RETAIN**  

1. The minimum input you need to run RETAIN is the "visit file", the number of unique medical codes in the "visit file", 
the "label file", and the output path. The output path is where the learned weights and the log will be saved.  
`python retain.py <visit file> <# codes in the visit file> <label file> <output path>`  

2. Specifying `--verbose` option will print training process after each 10 mini-batches.

3. You can specify the size of the embedding W\_emb, the size of the GRU hidden layer for generating alpha, and the size of the GRU hidden layer for generating beta.
The respective commands are `--embed\_size`, `--alpha\_hidden\_dim\_size`, and `--beta\_hidden\_dim\_size`.
For example `--alpha\_hidden\_dim\_size 128` will tell RETAIN to use a GRU with 128-dimensional hidden layer for generating alpha.

4. Dropouts are applied to two places: 1) to the input embedding, 2) to the context vector c\_i. The respective dropout rates can be adjusted using `--dropout_embed` and `--dropout_context`. Dropout values affect the performance so it is recommended to tune them for your data.

5. L2 regularizations can be applied to W\_emb, w\_alpha, W\_beta, and w\_output.

6. Additional options can be specified such as the size of the batch size, the number of epochs, etc. Detailed information can be accessed by `python retain.py --help`

7. My personal recommendation: use mild regularization (0.0001 ~ 0.001) on all four weights, and use moderate dropout on the context vector only. But this entirely depends on your data, so you should always tune the hyperparameters by yourself.

**STEP 4: Getting your results**  

RETAIN checks the AUC of the validation set after each epoch, and if it is higher than all previous value, it will save the current model. The model file is generated by [numpy.savez_compressed](http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.savez_compressed.html).

**Step 5: Testing your model**

1. Using the file "test\_retain.py", you can calculate the contributions of each medical code at each visit. First you need to have a trained model that was saved by numpy.savez\_compressed. Note that you need to know the configuration with which you trained RETAIN (use of "time file", use of "--use\_log\_time", value of "--embed\_size", etc.)

2. Again, you need the "visit file" and "label file" prepared in the same way. This time, however, you do not need to follow the ".train", ".valid", ".test" rule. The testing script will try to load the file name as given.

3. Using additional options such as "--embed\_size" and "--use\_log\_time", you should use exactly the same configuration with which you trained the model. For more detailed information, use "--help" option.

4. The minimum input to run the testing script is the "model file", "visit file", "label file", and "output file". "output file" will be where the contributions will be stored.
`python test_retain.py <model file> <visit file> <label file> <output file>`
