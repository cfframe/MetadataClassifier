import argparse
import datetime
import os.path
from pathlib import Path
import subprocess


"""
Description: Repeatedly call training script with permutations of given arguments.

Assumptions: 
* All data matches the same labels content. Note that some have fewer labels than others.

"""
DEFAULT_SRC_DIR = '.data'
DEFAULT_DATA_FILE_NAMES = ['serums_fcrb_Tokenized.csv', 'serums_fcrb_NotTokenized.csv']
DEFAULT_LABELS_FILE_NAME = 'serums_fcrb_labels.txt'


def parse_args():
    parser = argparse.ArgumentParser(description='Loop through script with variations on arguments.')
    parser.add_argument('-sdd', '--src_data_dir', type=str, default=DEFAULT_SRC_DIR,
                        help='Path to source data directory.')
    parser.add_argument('-dfn', '--data_file_names', type=str, nargs='+', default=DEFAULT_DATA_FILE_NAMES,
                        help='Source data file names.')
    parser.add_argument('-tfn', '--test_file_names', type=str, nargs='+', default=[],
                        help='Optional source test file names.')
    parser.add_argument('-l', '--labels_file_name', type=str, default=DEFAULT_LABELS_FILE_NAME,
                        help='Source labels file name.')
    parser.add_argument('-td', '--target_dir', type=str, default=Path(__file__).parent,
                        help='Working directory for saving files etc. Default: parent directory of this script.')
    parser.add_argument('-bm', '--bert_models', type=str, nargs='+', default=['bert-base-uncased'],
                        help='List of pre-trained BERT models (default: [\'bert-base-uncased\']).')
    parser.add_argument('-do', '--dropouts', type=float, nargs='+', default=[0.5],
                        help='List of dropouts (default: []).')
    parser.add_argument('-lr', '--learning_rates', type=float, nargs='+', default=[1e-6],
                        help='List of learning rates (default: [0.000001])')
    parser.add_argument('-bs', '--batch_sizes', type=int, nargs='+', default=[5],
                        help='List of batch sizes (default: [5])')
    parser.add_argument('-ep', '--num_epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('-d', '--device', type=int, default=-1, help='Compute device to use (default: -1, for any cpu)')
    parser.add_argument('-r', '--to_run_sub_scripts', action='store_true',
                        help='Whether to run each sub-script; otherwise just print the command.')
    parser.add_argument('-am', '--to_archive_model', action='store_true',
                        help='Flag to create an archive file of the model; assume to_archive is true (default: false)')

    args = parser.parse_args()

    return args


def main():

    args = parse_args()

    start_time = datetime.datetime.now()
    print(f"Loop Start : {start_time.strftime('%y%m%d_%H:%M:%S')}")

    template_script = "python bert_train.py -sdd <src_data_dir> <data_and_test_arguments> " \
                      "-td <target_dir> " \
                      "-l <labels_file_name> " \
                      "-bm <bert_model> -do <dropout> " \
                      "-lr <learning_rate> -bs <batch_size> -ep <epochs> -d <device> " \
                      "--save_prefix <save_prefix> --to_archive"
    if args.to_archive_model:
        template_script = f'{template_script} --to_archive_model'

    src_data_dir = args.src_data_dir
    data_file_names = args.data_file_names
    test_file_names = args.test_file_names
    labels_file_name = args.labels_file_name
    target_dir = args.target_dir
    bert_models = args.bert_models
    dropouts = args.dropouts
    learning_rates = args.learning_rates
    batch_sizes = args.batch_sizes
    epochs = args.num_epochs
    device = args.device

    base_script_to_run = template_script.replace('<src_data_dir>', src_data_dir)\
        .replace('<labels_file_name>', labels_file_name)\
        .replace('<target_dir>', target_dir)\
        .replace('<device>', str(device))

    for i in range(len(data_file_names)):
        data_file_name = str(data_file_names[i])
        data_and_test_arguments = ' '.join(['-dfn', data_file_name])
        if len(test_file_names) > 0:
            test_file_name = str(test_file_names[i])
            data_and_test_arguments = ' '.join([data_and_test_arguments, '-tfn', test_file_name])
        for bert_model in bert_models:
            for dropout in dropouts:
                dropout = float(dropout)
                for learning_rate in learning_rates:
                    learning_rate = float(learning_rate)
                    for batch_size in batch_sizes:
                        batch_size = int(batch_size)
                        script_to_run = base_script_to_run

                        t = 'NT' if 'NotTokenized' in data_file_name else 'T'
                        bm = 'X'
                        if bert_model == 'bert-base-uncased':
                            bm = 'BUC'
                        elif bert_model == 'bert-base-cased':
                            bm = 'BC'
                        if bert_model == 'bert-large-uncased':
                            bm = 'LUC'
                        elif bert_model == 'bert-large-cased':
                            bm = 'LC'

                        save_prefix = f'f{t}_bm{bm}_do{dropout}_lr{learning_rate}_bs{batch_size}_ep{epochs}'
                        script_to_run = script_to_run.replace('<data_and_test_arguments>', data_and_test_arguments)\
                            .replace('<bert_model>', bert_model)\
                            .replace('<dropout>', str(dropout))\
                            .replace('<learning_rate>', str(learning_rate))\
                            .replace('<batch_size>', str(batch_size))\
                            .replace('<epochs>', str(epochs))\
                            .replace('<save_prefix>', save_prefix)
                        print(script_to_run)

                        if args.to_run_sub_scripts:
                            subprocess.run(script_to_run, shell=True)

    end_time = datetime.datetime.now()

    # File test_results.csv is appended in bert_train.py
    results_path = Path(os.path.join(target_dir, 'test_results.csv'))
    if results_path.exists():
        new_path = Path(os.path.join(
            results_path.parent,
            f'{end_time.strftime("%y%m%d_%H%M%S")}_{results_path.name}'))
        results_path.rename(new_path)
        print(f'Renamed file "{results_path}" to "{new_path.name}"')
    else:
        print(f'Path does not exist, so not renamed: {results_path}')

    print(f"Loop Elapsed time : {(end_time - start_time)}")
    print(f"Loop End : {end_time.strftime('%y%m%d_%H:%M:%S')}")


if __name__ == '__main__':
    main()
