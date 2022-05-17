# nb_args.py
import os


class NaiveBayesArgs:
    """NaiveBayesArgs object"""
    def __init__(self, args: {}):
        """Arguments and direct dependants.

        Keyword arguments:
        :param args: Arguments dictionary
        """
        self.src_path = args.src_path
        self.output_dir = args.output_dir
        self.train_data_file_name = args.train_data
        self.val_data_file_name = args.val_data
        self.classes_file_name = args.classes
        self.train_data_path = os.path.join(args.src_path, args.train_data)
        self.val_data_path = os.path.join(args.src_path, args.val_data)
        self.classes_path = os.path.join(args.src_path, args.classes)
        self.save_label = args.save_label
        self.to_save_model = args.to_save_model
        self.verbose = args.verbose


