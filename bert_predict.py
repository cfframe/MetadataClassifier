import argparse
import datetime
import os
import pandas as pd
from pathlib import Path
import shutil

import torch

from src.bert_dataset import Dataset
from src.bert_model import BertClassifier
from src.config import Config
from src.misc_tools import MiscTools
from src.model_tools import ModelTools
from src.file_tools import FileTools

"""
Description: Use the state dictionary of a trained model to infer SERUMS labels from pre-prepared table metadata/
field descriptors.

Requires these files:
- New data (can be pre-labelled)
- Saved model
- Labels list - must match the list used in training the model

Example usage:
python bert_predict.py -sdd .data -dfn serums_trial_data.csv -l serums_fcrb_labels.txt -sdp C:\temp\Serums\ClassifierResults\fNT_lr1e-05_bs2_ep15_state_dict_epoch15.pt -d 0 --save_prefix yy --to_archive
"""

TEST_SRC_DIR = '.data'
TEST_DATA_FILE_NAME = 'serums_trial_data.csv'
TEST_LABELS_FILE_NAME = 'serums_fcrb_labels.txt'


def parse_args():
    parser = argparse.ArgumentParser(description='Infer SERUMS labels for table metadata.')
    parser.add_argument('-sdd', '--src_data_dir', type=str, default=TEST_SRC_DIR, help='Path to source data directory.')
    parser.add_argument('-dfn', '--data_file_name', type=str, default=TEST_DATA_FILE_NAME, help='Source data file name.')
    parser.add_argument('-l', '--labels_file_name', type=str, default=TEST_LABELS_FILE_NAME,
                        help='Source labels file name.')
    parser.add_argument('-sdp', '--state_dict_path', type=str, help='State dictionary file path.')
    parser.add_argument('-bm', '--bert_model', type=str, default='bert-base-uncased',
                        help='Pre-trained BERT model (default: bert-base-uncased).')
    parser.add_argument('-td', '--target_dir', type=str, default=Path(__file__).parent,
                        help='Working directory for saving files etc. Default: parent directory of this script.')
    parser.add_argument('-d', '--device', type=int, default=-1, help='Compute device to use (default: -1, for any cpu)')
    parser.add_argument('--save_prefix', help='Path prefix to save outputs (optional)')
    parser.add_argument('--to_archive', action='store_true', help='Flag to create an archive file of outputs.')

    args = parser.parse_args()

    return args


def main():
    ###
    # Load model
    # Load evaluation data
    # Evaluate model against evaluation data
    # Save results
    # ###

    args = parse_args()
    args.eval_only = True
    config = Config(args)

    print(f"Start : {config.start_time.strftime('%y%m%d_%H:%M:%S')}")

    # Check for key items
    # TODO Create fn that takes list of paths and prints missing ones.
    if not Path(config.src_data_file_path).exists():
        print(f'Exiting: file not found, at {config.src_data_file_path}.')
        exit()

    if not Path(config.state_dict_path).exists():
        print(f'Exiting: file not found, at {config.state_dict_path}.')
        exit()

    # Copy source files to output dir
    shutil.copy(config.src_data_file_path, config.output_dir)
    shutil.copy(config.labels_path, config.output_dir)

    # Read in data, insert dummy category column
    df = pd.read_csv(config.src_data_file_path)
    # Insert dummy category if column doesn't already exist
    was_labelled = True
    if 'category' not in df.columns:
        was_labelled = False
        df.insert(0, 'category', 'key')

    # Load
    model = BertClassifier(category_count=len(config.labels_dict), bert_model=config.bert_model)
    model.load_state_dict(torch.load(config.state_dict_path))
    model.eval()

    if config.use_cuda:
        model.cuda()

    er = ModelTools.evaluate(model, df, config)

    if not was_labelled:
        er.predictions.drop(['category', 'is_match'], axis=1, inplace=True)

    target_path = os.path.join(config.output_dir, 'predictions.csv')
    er.predictions.to_csv(target_path, index=False)
    print(f'Predictions saved to {target_path}')

    # Probabilities for each label
    columns = [key for key in config.labels_dict.keys()]
    columns.append('True.Label')
    columns.append('Predicted.Label')

    file_name = 'test_probabilities.txt'
    MiscTools.append_data(output_dir=config.output_dir, data=er.probabilities,
                          file_name=file_name, header=columns)

    if config.to_archive:
        now = datetime.datetime.now()
        print(f"About to archive: {now.strftime('%y%m%d_%H:%M:%S')}")

        results_path = FileTools.make_datetime_named_archive(src_path_to_archive=config.output_dir,
                                                             base_target_path=config.output_dir,
                                                             format='zip',  datestamp=now)
        print(f'Archived: {results_path}')

    end_time = datetime.datetime.now()
    print(f"Elapsed time : {(end_time - config.start_time)}")

    print(f"End : {end_time.strftime('%y%m%d_%H:%M:%S')}")


if __name__ == '__main__':
    main()
