# misc_tools.py

import csv
import os
from pathlib import Path
import torch

from src.config import Config
import torch.nn as nn


class MiscTools:
    """Miscellaneous utilities for this repo"""

    # @staticmethod
    # def get_activation(arg: str = 'tanh'):
    #     activation = nn.Tanh
    #     arg = arg.lower()
    #
    #     if arg == 'tanh':
    #         activation = nn.Tanh
    #     elif arg == 'leakrelu':
    #         activation = nn.LeakyReLU
    #     elif arg == 'relu':
    #         activation = nn.ReLU
    #     elif arg == 'sigmoid':
    #         activation = nn.Sigmoid
    #
    #     return activation
    #
    # @staticmethod
    # def get_loss_function_name(arg: str = 'bce'):
    #     loss_function = 'binary_cross_entropy'
    #     final_sigmoid = True
    #     loss_title = 'BCE Loss'
    #
    #     if arg == 'mse':
    #         loss_function = 'mse_loss'
    #         final_sigmoid = False
    #         loss_title = 'Mean Square Error'
    #     elif arg == 'bce_with_logits':
    #         loss_function = 'binary_cross_entropy_with_logits'
    #         final_sigmoid = False
    #         loss_title = 'BCE Loss'
    #
    #     return loss_function, final_sigmoid, loss_title
    #
    @staticmethod
    def save_label(args):
        value_args = {'batch_size': 'bs',
                      'num_epochs': 'ep'}
        label = args.save_prefix + '_'
        args = vars(args)
        for arg in args:
            if arg in value_args:
                label += value_args[arg] + str(args[arg])

        return label

    @staticmethod
    def append_data(output_dir: Path, data: list, file_name: str, header: list = []):
        # Expects data in form of a list of lists
        save_path = os.path.join(output_dir, file_name)

        if not Path(save_path).exists() and len(header) > 0:
            with open(save_path, 'w', encoding='utf-8', newline='') as data_file:
                writer = csv.writer(data_file)
                writer.writerow(header)

        with open(save_path, 'a', encoding='utf-8', newline='') as data_file:
            writer = csv.writer(data_file)
            writer.writerows(data)

    @staticmethod
    def save_trained_model(epoch: int, model: nn.Module, config: Config) -> [str, str]:
        # path_prefix, epoch, digits, trained_dir, p_net, q_net, use_cuda
        path_prefix = config.save_prefix
        digits = config.label_digits
        trained_dir = config.trained_dir

        model_path = ''
        state_dict_path = ''

        if path_prefix is not None:
            epoch_str = str(epoch + 1).zfill(digits)

            # Save entire model - for simpler coding later, but less flexible usage
            model_path = os.path.join(trained_dir, path_prefix + '_model_epoch{}.pt'.format(epoch_str))
            model.eval().cpu()
            torch.save(model, model_path)

            # Save state dictionary - more coding to use, but more flexible
            state_dict_path = os.path.join(trained_dir, path_prefix + '_state_dict_epoch{}.pt'.format(epoch_str))
            model.eval().cpu()
            torch.save(model.state_dict(), state_dict_path)

            # Revert to cuda
            if config.use_cuda:
                model.cuda()

        return [model_path, state_dict_path]

    @staticmethod
    def save_model_specs_to_file(outputs_dir, models):
        path = os.path.join(outputs_dir, 'models.txt')

        with open(path, 'w') as file:
            for model in models:
                print(model, file=file)

    @staticmethod
    def save_dictionary_with_headers_to_file(target_path: str, dictionary: dict, header: list):
        with open(target_path, 'w', newline='') as file:
            writer = csv.writer(file, delimiter=',', quoting=csv.QUOTE_NONE, escapechar='\\')
            writer.writerow(header)
            for key in dictionary.keys():
                writer.writerow([dictionary[key], key])
