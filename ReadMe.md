# ReadMe for MetadataClassifier
## Overview
This relates to WP2.

## Description
This repo contains mechanisms for further training a pre-trained BERT model using health metadata and TPOLE 
classifications, and for using the resultant state dictionary to make a first pass at classifying new metadata. 
It also includes a bespoke Naïve Bayes classifier for comparison.  

Metadata in this context refers to fields with descriptors and associated source name (e.g. name of a source database 
table or database view). The input for training is such metadata that has been pre-processed, with the final format 
being rows comprising category (based on TPOLE plus 'key') and processed text, separated by a comma. The 'text' value of 
each row should have no punctuation. 

A full list of available pre-trained BERT models can be found at: 
https://huggingface.co/transformers/v3.3.1/pretrained_models.html

## Usage
Primary scripts:
<ul>
<li>bert_train.py</li>
<li>bert_train_harness.py</li>
<li>bert_predict.py</li>
<li>naive_bayes.py</li>
</ul>

Examples below assume the script is being run from the main application directory.

Help text available by running <code>py <i>script_name</i>.py -h</code> in the terminal.

### bert_train.py.
Train model against data and set parameters as below.

Arguments:

<code>-sdd</code>, <code>--src_data_dir</code>: Path to source data directory. 
Default: DEFAULT_SRC_DIR, as defined in the script.  

<code>-dfn</code>, <code>--data_file_name</code>: Source data file name. Default: DEFAULT_DATA_FILE_NAME,
as defined in the script.

<code>-tfn</code>, <code>--test_file_name</code>: Optional source test file name. Default: empty string.

<code>-l</code>, <code>--labels_file_name</code>: Source labels file name. Default: DEFAULT_LABELS_FILE_NAME, as 
defined in the script.

<code>-td</code>, <code>--target_dir</code>: Working directory for saving files etc. 
Default: parent directory of the script.

<code>-bm</code>, <code>--bert_model</code>: Pre-trained BERT model. Default: bert-base-uncased

<code>-do</code>, <code>--dropout</code>: Dropout. Default: 0.5

<code>-lr</code>, <code>--learning_rate</code>: Learning rate. Default: 1e-6

<code>-bs</code>, <code>--batch_size</code>: Batch size. Default: 5

<code>-ep</code>, <code>--num_epochs</code>: Number of training epochs. Default: 100

<code>-d</code>, <code>--device</code>: Compute device to use. Default: -1, for cpu 

<code>--save_prefix</code>: Path prefix to save models (optional). 

<code>--to_archive</code>: Flag to create an archive file of results. 
(True if flag present, default False if not). 

<code>-am</code>, <code>--to_archive_model</code>: Flag to create an archive file of the model; assume to_archive is 
true. Default: False.

Example:<br />
```commandline
python bert_train.py -sdd .data -dfn serums_fcrb_Tokenized.csv -l serums_fcrb_labels.txt -lr 1e-6 -bs 2 -ep 3 -d 0 --save_prefix xx --to_archive --to_archive_model
```

The outputs include:
<ul>
<li>images folder, containing graphs (Category frequencies, Accuracy and Loss)</li>
<li>labels.txt - list of categories derived from the source data</li>
<li>models.txt - model definition(s)</li>
<li>test_result.txt - final result based on test data not used in the training</li>
<li>train_val_results.txt - data captured during training and used to generate the graphs</li>
<li>*_model_*.pt - trained model</li>
<li>*_state_dict_*.pt - state dictionary of the trained model</li>
</ul>

### bert_train_harness.py
Harness for running bert_train.py through all permutations of variations in selected arguments.

Arguments:

<code>-sdd</code>, <code>--src_data_dir</code>: Path to source data directory. 
Default: DEFAULT_SRC_DIR, as defined in the script.  

<code>-dfn</code>, <code>--data_file_names</code>: List of source data file names. 
Default: DEFAULT_DATA_FILE_NAMES, as defined in the script.

<code>-dfn</code>, <code>--test_file_names</code>: Optional list of source test file names. If supplied, must
be paired up with and in same order as source data file names. Default: empty list.

<code>-l</code>, <code>--labels_file_name</code>: List of source labels file names, corresponding to 
data file names. 
Default: DEFAULT_LABELS_FILE_NAME, as defined in the script.

<code>-td</code>, <code>--target_dir</code>: Working directory for saving files etc. 
Default: parent directory of the script.

<code>-bm</code>, <code>--bert_model</code>: List of pre-trained BERT models. Default: ['bert-base-uncased']

<code>-do</code>, <code>--dropout</code>: List of dropout rates. Default: [0.5]

<code>-lr</code>, <code>--learning_rate</code>: List of learning rates. Default: ['1e-6']

<code>-bs</code>, <code>--batch_size</code>: List of batch sizes. Default: ['5']

<code>-ep</code>, <code>--num_epochs</code>: List of numbers of training epochs. Default: ['100']

<code>-d</code>, <code>--device</code>: Compute device to use. Default: -1, for cpu 

<code>-r</code>, <code>--to_run_sub_scripts</code>: Whether to run each sub-script; otherwise just print the command.
Default: False (if flag not present).

<code>--to_archive</code>: Flag to create an archive file of results. 
Default: False (if flag not present).

<code>-am</code>, <code>--to_archive_model</code>: Flag to create an archive file of the model; assume to_archive is 
true. Default: False.

Outputs as per bert_train.py.

### bert_predict.py
Classifies meta data. Requires:
- data to be classified
- trained model's state dictionary
- labels list as used for training the model
- the name of the pre-trained BERT model
The data must have 'category' and 'text' columns. It can optionally have other columns too.

Arguments:

<code>-sdd</code>, <code>--src_data_dir</code>: Path to source data directory. 
Default: TEST_SRC_DIR, as defined in the script.  

<code>-dfn</code>, <code>--data_file_name</code>: Source data file name. 
Default: TEST_DATA_FILE_NAME, as defined in the script.

<code>-l</code>, <code>--labels_file_name</code>: List of source labels file names, corresponding to 
data file names. 
Default: TEST_LABELS_FILE_NAME, as defined in the script.

<code>-sdp</code>, <code>--state_dict_path</code>: Saved state dictionary of the trained model. 

<code>-bm</code>, <code>--bert_model</code>: Pre-trained BERT model used in training. Default: ['bert-base-uncased']

<code>-td</code>, <code>--target_dir</code>: Working directory for saving files etc. 
Default: parent directory of this script.

<code>-d</code>, <code>--device</code>: Compute device to use. Default: -1, for cpu 

<code>--save_prefix</code>: Path prefix to save outputs (optional). 

<code>--to_archive</code>: Flag to create an archive file of outputs. 
(True if flag present, default False if not). 

### naive_bayes.py
Naïve Bayes classifier. Takes same data file inputs as <code>bert_train.py</code>. 

The main data file is used for building the model, and the test data file is used for validation. There is no separation
between validation and test in this instance because there is no re-training of the model - the classifier
is a static model. Consequently, the classifier runs once only against a specific set of data.

Laplacian probability is used to accommodate instances where validation data contains terms that are not in the
original vocabulary (and which would otherwise result in a zero probability simultaneously for  being present and 
not being present).

The output is information regarding training and validation accuracy, plus saved model vocabularies. It is not
envisaged that these would be used.

Arguments:

<code>-s</code>, <code>--src_path</code>: Source path for processing. 
Default: DATA_DIR, as defined in the script.  

<code>-od</code>, <code>--output_dir</code>: Working directory for saving files etc. 
Default: Parent directory of the script.

<code>-td</code>, <code>--train_data</code>: Training data. 

<code>-vd</code>, <code>--val_data</code>: Validation data. 

<code>-c</code>, <code>--classes</code>: All classes. 
Default: tpole_labels.txt.

<code>-sl</code>, <code>--save_label</code>: Optional label to add to save name for easier identification.

<code>-sm</code>, <code>--to_save_model</code>: Flag for whether to save the model vocabulary.

<code>-v</code>, <code>--verbose</code>: Flag for whether to print lots of text data.



## General Python project basics
### Tools and technologies used:
<ul>
<li>PyCharm 2021.2.3 - 2021.3.2</li>
<li>python 3.8.10 - packages as listed in <code>requirements.txt</code></li>
</ul>

### Set up
Assumes NVIDIA CUDA 11.3 and NVIDIA CuDNN already installed.

Python package requirements are defined in <code>requirements.txt</code>. We used a virtual environment for installing these
to reduce the risk of package dependency issues.

One way of installing requirements:
```commandline
python -m pip install --upgrade pip
pip install -U pip setuptools wheel
pip install transformers
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install -r requirements.txt
```
For further PyTorch details and dependencies, see https://pytorch.org/get-started/locally/

Code was developed in a local repository with root at MetadataClassifier level, and pushed to a deeper GitHub repo 
via GitHub Desktop.

### Reference
Initial example code for using BERT from Text Classification with BERT in PyTorch | by Ruben
(https://towardsdatascience.com/text-classification-with-bert-in-pytorch-887965e5820f)
