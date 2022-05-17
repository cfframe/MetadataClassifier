import argparse
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
import shutil

import torch

from src.bert_model import BertClassifier
from src.config import Config
from src.file_tools import FileTools
from src.misc_tools import MiscTools
from src.model_tools import ModelTools
from src.plot_helper import PlotHelper

"""
Description: Train a BERT based model on labelled SERUMS data based on meta-data with field descriptors.

Example usage:
python bert_train.py -sdd .data -dfn serums_fcrb_Tokenized.csv -l serums_fcrb_labels.txt -lr 1e-6 -bs 2 -ep 3 -d 0 --save_prefix xx --to_archive --to_archive_model

"""

DEFAULT_SRC_DIR = '.data'
DEFAULT_DATA_FILE_NAME = 'serums_fcrb_Tokenized.csv'
DEFAULT_LABELS_FILE_NAME = 'serums_fcrb_labels.txt'
MIN_EPOCHS = 3


def parse_args():
    parser = argparse.ArgumentParser(description='Further train pre-trained BERT model with labelled serums metadata. ')
    parser.add_argument('-sdd', '--src_data_dir', type=str, default=DEFAULT_SRC_DIR, help='Path to source data directory.')
    parser.add_argument('-dfn', '--data_file_name', type=str, default=DEFAULT_DATA_FILE_NAME, help='Source data file name.')
    parser.add_argument('-tfn', '--test_file_name', type=str, default='', help='Optional separate source test data file name.')
    parser.add_argument('-l', '--labels_file_name', type=str, default=DEFAULT_LABELS_FILE_NAME,
                        help='Source labels file name.')
    parser.add_argument('-td', '--target_dir', type=str, default=Path(__file__).parent,
                        help='Working directory for saving files etc. Default: parent directory of this script.')
    parser.add_argument('-bm', '--bert_model', type=str, default='bert-base-uncased', help='Pre-trained BERT model.')
    parser.add_argument('-do', '--dropout', type=float, default='0.5',
                        help='Dropout.')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-6, help='Learning rate (default: 0.000001)')
    parser.add_argument('-bs', '--batch_size', type=int, default=5, help='Batch size (default: 5)')
    parser.add_argument('-ep', '--num_epochs', type=int, default=100, help='Number of training epochs (default: 100)')
    parser.add_argument('-d', '--device', type=int, default=-1, help='Compute device to use (default: -1, for any cpu)')
    parser.add_argument('--save_prefix', help='Path prefix to save models (optional)')
    parser.add_argument('--to_archive', action='store_true', help='Flag to create an archive file of results.')
    parser.add_argument('-am', '--to_archive_model', action='store_true',
                        help='Flag to create an archive file of the model; assume to_archive is true (default: false)')

    args = parser.parse_args()

    return args


def main():

    args = parse_args()
    args.eval_only = False
    config = Config(args)

    if config.use_cuda:
        torch.cuda.set_device(config.device)

    if config.num_epochs < MIN_EPOCHS:
        print(f'Too few epochs (num_epochs=={config.num_epochs}). Exiting.')
        exit()

    print(f"Start : {config.start_time.strftime('%y%m%d_%H:%M:%S')}")

    # Save labels used in training to output dir for ref
    target_path = os.path.join(config.output_dir, 'labels.txt')
    header = ['index', 'category']
    MiscTools.save_dictionary_with_headers_to_file(target_path, config.labels_dict, header)

    df = pd.read_csv(config.src_data_file_path)
    df.head()

    df.groupby(['category']).size().plot.barh()
    plt.title = 'Category Frequencies'
    target_path = os.path.join(config.output_dir, 'images', plt.title.replace(' ', '_') + '.svg')
    # Need bbox_inches='tight' below to ensure y-label is not cut off
    plt.savefig(target_path, bbox_inches='tight')

    # Preprocessing data
    # If separate test data not provided, split df into training, validation, test at 80:10:10
    # If separate test data available, split main df into training, validation at 80:10 ~ 89:11  (assumes test came from
    # same original source, and is 10% of that)
    np.random.seed(112)
    if len(config.test_file_name) == 0:
        df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42),
                                             [int(.8 * len(df)), int(.9 * len(df))])
    else:
        df_train, df_val = np.split(df.sample(frac=1, random_state=42),
                                    [int(.89 * len(df))])
        df_test = pd.read_csv(config.src_test_file_path)

    print(f'df_train: {len(df_train)}; df_val: {len(df_val)}; df_test: {len(df_test)}')

    model = BertClassifier(dropout=config.dropout, category_count=len(config.labels_dict), bert_model=config.bert_model)
    if config.use_cuda:
        model.cuda()

    MiscTools.save_model_specs_to_file(config.output_dir, [model])
    results, saved_model_train_result, saved_train_probabilities_by_label, saved_val_probabilities_by_label \
        = ModelTools.train_model(model, df_train, df_val, df_test, config)

    # Probabilities for each label
    columns = [key for key in config.labels_dict.keys()]
    columns.append('True.Label')
    columns.append('Predicted.Label')

    # - all training results
    file_name = 'train_probabilities.txt'
    MiscTools.append_data(output_dir=config.output_dir, data=saved_train_probabilities_by_label,
                          file_name=file_name, header=columns)

    # - all validation results
    file_name = 'val_probabilities.txt'
    MiscTools.append_data(output_dir=config.output_dir, data=saved_val_probabilities_by_label,
                          file_name=file_name, header=columns)

    # Plot and save results
    results_df = pd.DataFrame(data=results[1:], columns=results[0])

    PlotHelper.basic_run_plot(df=results_df, output_dir=os.path.join(config.output_dir, 'images'))

    MiscTools.append_data(output_dir=config.output_dir, data=results, file_name='train_val_results.txt')

    results_path = ''

    # Create archive of output directory
    if config.to_archive:
        now = datetime.datetime.now()
        print(f"About to archive: {now.strftime('%y%m%d_%H:%M:%S')}")

        # Create a new folder for the model and move it there. Then archive the rest of the files.
        shutil.move(config.old_model_dir, config.new_model_dir)

        results_path = FileTools.make_datetime_named_archive(src_path_to_archive=config.output_dir,
                                                             base_target_path=config.output_dir,
                                                             format='zip',  datestamp=now)

        # Create archive of trained model - assumes archiving results
        if config.to_archive_model:
            FileTools.make_datetime_named_archive(config.new_model_dir, config.new_model_dir, 'zip', now)

    end_time = datetime.datetime.now()
    print(f"Elapsed time : {(end_time - config.start_time)}")

    print(f"End : {end_time.strftime('%y%m%d_%H:%M:%S')}")

    # Append test result to file external to the main output dir - doing this way so can collate results
    # from several script runs
    tokenized = 'n' if 'NotTokenized' in config.data_file_name else 'y'
    results_archive_file_name = 'No archive file' if results_path == '' else Path(results_path).name
    append_timestamp = datetime.datetime.now().strftime('%y%m%d_%H%M%S')

    data = [[append_timestamp,
             results_archive_file_name, tokenized,
             config.bert_model, config.dropout,
             config.learning_rate, config.batch_size, config.num_epochs,
             saved_model_train_result.epoch,
             saved_model_train_result.validation_loss,
             saved_model_train_result.validation_accuracy,
             saved_model_train_result.test_accuracy]]
    header = ['Append time',
              'Results archive file name', 'IsTokenized',
              'BERT model', 'Dropout',
              'Learning rate', 'Batch size', 'Total epochs',
              'Saved model epoch',
              'Saved model validation loss',
              'Saved model validation accuracy',
              'Saved model test accuracy']
    MiscTools.append_data(output_dir=Path(config.output_dir).parent, data=data, file_name='test_results.csv',
                          header=header)


if __name__ == '__main__':
    main()
