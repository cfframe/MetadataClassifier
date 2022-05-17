import csv
import datetime
import numpy as np
import os
from pathlib import Path
import torch

from src.file_tools import FileTools


class Config:
    def __init__(self, args):

        # Many attributes set via self.initial_prep()

        # Common stuff
        self.src_data_dir = os.path.normcase(args.src_data_dir)
        self.data_file_name = args.data_file_name
        self.src_data_file_path = os.path.join(self.src_data_dir, self.data_file_name)
        self.labels_file_name = args.labels_file_name
        self.labels_path = None
        self.target_dir = args.target_dir
        self.bert_model = None
        self.save_prefix = args.save_prefix
        self.output_dir = None

        self.eval_only = args.eval_only
        self.for_training = not args.eval_only

        self.device = None
        self.use_cuda = (args.device >= 0) and torch.cuda.is_available()
        self.set_device(args.device)

        self.to_archive = args.to_archive
        self.start_time = datetime.datetime.now()

        # For training
        self.test_file_name = None
        self.src_test_file_path = None
        self.dropout = None
        self.learning_rate = None
        self.batch_size = None
        self.num_epochs = None
        self.to_archive_model = None

        self.images_dir = None
        self.trained_dir = None
        self.label_digits = None
        self.labels_dict = None
        self.old_model_dir = None
        self.new_model_dir = None

        # Evaluation only
        self.state_dict_path = None
        self.model_path = None

        # Set values
        self.initial_prep(args)

    def initial_prep(self, args):
        # Set up configuration
        # Ensure required directories exist

        self.labels_path = os.path.join(self.src_data_dir, self.labels_file_name)
        self.labels_dict = self.load_labels_dict(self.labels_path)
        self.output_dir = os.path.join(args.target_dir, f'outputs_{self.save_prefix}')
        FileTools.ensure_empty_directory(self.output_dir)
        self.bert_model = args.bert_model

        if self.for_training:
            self.test_file_name = args.test_file_name
            self.src_test_file_path = os.path.join(self.src_data_dir, self.test_file_name) \
                if len(self.test_file_name) > 0 else ''
            self.to_archive_model = args.to_archive_model
            self.dropout = args.dropout
            self.learning_rate = args.learning_rate
            self.batch_size = args.batch_size
            self.num_epochs = args.num_epochs

            self.to_archive_model = args.to_archive_model

            self.images_dir = os.path.join(self.output_dir, 'images')
            self.trained_dir = os.path.join(self.output_dir, 'trained')
            self.label_digits = int(np.log10(self.num_epochs)) + 1
            self.old_model_dir = os.path.join(self.output_dir, 'trained')
            self.new_model_dir = os.path.join(f'{Path(self.output_dir).parent}',
                                              f'{Path(self.output_dir).name}_model')

            FileTools.ensure_empty_directory(self.new_model_dir)
            FileTools.ensure_empty_directory(self.images_dir)
            FileTools.ensure_empty_directory(self.trained_dir)

        # eval_only
        else:
            if len(args.state_dict_path) > 0:
                self.state_dict_path = args.state_dict_path
            else:
                self.model_path = os.path.normcase(args.model_path)

        FileTools.save_command_args_to_file(args=vars(args), save_path=os.path.join(self.output_dir, 'command.txt'))

    def set_device(self, device: int):
        if (device >= 0) and torch.cuda.is_available():
            self.device = 'cuda:' + str(device)
            print(f'# Using CUDA device: {self.device}')
        else:
            self.device = 'cpu'
            print(f'# Using CPU')

        torch.device(self.device)

    @staticmethod
    def load_labels_dict(labels_dict_path: str) -> dict:
        with open(labels_dict_path, mode='r') as infile:
            reader = csv.DictReader(infile)
            # Ensure label index is type integer for BERT
            labels_dict = {rows['category']: int(rows['index']) for rows in reader}

        return labels_dict
