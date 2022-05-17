# file_tools.py

import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import os
import random
import re
import shutil
from skimage.transform import resize
import sys


class FileTools:
    """Utilities for managing data from and to files"""

    @staticmethod
    def chunks_generator(input_list: list, chunk_size: int) -> list:
        """Yield chunks of supplied data by given size

        :param input_list: the list from which chunks are to be yielded
        :param chunk_size: number of items in each chunk
        :return: list of length chunk_size
        """
        remainder = len(input_list) - (int(len(input_list)/chunk_size) * chunk_size)

        for i in range(0, int(len(input_list) / chunk_size)):
            yield input_list[i * chunk_size: (i + 1) * chunk_size]
        if remainder > 0:
            i = int(len(input_list)/chunk_size)
            yield input_list[i * chunk_size: i * chunk_size + remainder]

    @staticmethod
    def create_dirs_from_file_header(file_path: str, separator: str, target_root: str) -> list():
        """Generate folder names from the first line of a file

        Assumes the first column is a list of files/items and the remaining column headers require associated folders

        Keyword arguments:
        :param file_path: full path to file
        :param separator: string separator for folder names
        :param target_root: root where new folders to be created
        :returns list of folder names
        """
        with open(file_path, 'r') as infile:
            header_line = (infile.readline()).strip()

        headers = header_line.split(separator)[1:]

        for header in headers:
            Path(os.path.join(target_root, header)).mkdir(parents=True, exist_ok=True)

        return headers

    @staticmethod
    def copy_files_to_class_dirs(info_file_path: str, separator: str, src_root: str, target_root: str,
                                 extension: str = ''):
        """Copy files from source dir to class dirs

        Keyword arguments:
        :param info_file_path: full path to file with class data of source files; assume this structure:
            line 1: headers
            column 1: file names
        :param separator: string separator for class names
        :param src_root: root for source files
        :param target_root: root for class dirs
        :param extension: if extension given, then suffix to file names
        :return (dataframe) list of folder names
        """

        df = pd.read_csv(info_file_path, index_col=0)

        FileTools.create_dirs_from_file_header(info_file_path, separator, target_root)

        for col in df.columns:
            target_dir = os.path.join(target_root, col)
            count = 0
            for filename in df[df[col] == 1].index:
                src_file = os.path.join(src_root, '.'.join([filename, extension]))
                target_file = os.path.join(target_dir, '.'.join([filename, extension]))

                shutil.copyfile(src_file, target_file)
                count += 1
            print(f'{count} files copied to {target_dir}')

        return df

    # @staticmethod
    # def copy_file_splits_to_class_dirs(info_file_path: str, separator: str, src_root: str,
    #                                    target_major_split_root: str, target_minor_split_root: str, main_split: float,
    #                                    extension: str = ''):
    #     """Copy files from source dir to class dirs, with main_split % going to the major split directory
    #
    #     Keyword arguments:
    #     :param info_file_path: full path to file with class data of source files; assume this structure:
    #         line 1: headers
    #         column 1: file names
    #     :param separator: string separator for class names
    #     :param src_root: root for source files
    #     :param target_major_split_root: main root for class dirs
    #     :param target_minor_split_root: split root for class dirs
    #     :param main_split: size of major split as decimal fraction of whole
    #     :param extension: if extension given, then suffix to file names
    #     :return (dataframe) list of folder names
    #     """
    #
    #     df = pd.read_csv(info_file_path, index_col=0)
    #
    #     FileTools.create_dirs_from_file_header(info_file_path, separator, target_major_split_root)
    #     FileTools.create_dirs_from_file_header(info_file_path, separator, target_minor_split_root)
    #
    #     for col in df.columns:
    #         paths = []
    #         for filename in df[df[col] == 1].index:
    #             paths.append(os.path.join(src_root, '.'.join([filename, extension])))
    #
    #         random.shuffle(paths)
    #         split_count = int(len(paths) * main_split)
    #
    #         output_dir = os.path.join(target_major_split_root, col)
    #         count = 0
    #         for src_file in paths[:split_count]:
    #             target_file = src_file.replace(src_root, output_dir)
    #             shutil.copyfile(src_file, target_file)
    #             count += 1
    #         print(f'{count} files copied to {output_dir}')
    #
    #         output_dir = os.path.join(target_minor_split_root, col)
    #         count = 0
    #         for src_file in paths[split_count:]:
    #             target_file = src_file.replace(src_root, output_dir)
    #             shutil.copyfile(src_file, target_file)
    #             count += 1
    #         print(f'{count} files copied to {output_dir}')
    #
    #     return df

    @staticmethod
    def copy_file_splits_to_class_dirs(info_file_path: str, separator: str, src_root: str,
                                       split_roots: list,
                                       # target_major_split_root: str, target_minor_split_root: str,
                                       splits: list,
                                       # main_split: float,
                                       extension: str = ''):
        """Copy files from source dir to class dirs, with main_split % going to the major split directory

        Keyword arguments:
        :param info_file_path: full path to file with class data of source files; assume this structure:
            line 1: headers
            column 1: file names
        :param separator: string separator for class names
        :param src_root: root for source files
        :param split_roots: list of root directories for classes, starting with main root split
        :param splits: list of relative sizes of splits as ints
        :param extension: if extension given, then suffix to file names
        :return (dataframe) list of folder names
        """

        df = pd.read_csv(info_file_path, index_col=0)

        splits_total = np.sum(splits)

        for split_root in split_roots:
            FileTools.create_dirs_from_file_header(info_file_path, separator, split_root)

        # Handle by class, for all splits
        for col in df.columns:
            # Get file paths
            paths = []
            for filename in df[df[col] == 1].index:
                paths.append(os.path.join(src_root, '.'.join([filename, extension])))

            random.shuffle(paths)

            # Calculate the number of files per split; last one is special case
            split_counts = []
            for split in splits[:len(splits) - 1]:
                split_count = int(len(paths) * split / splits_total)
                split_counts.append(split_count)

            split_count = len(paths) - np.sum(split_counts)
            split_counts.append(split_count)

            # Handle the directory splits.
            temp_split_roots = split_roots.copy()
            while len(temp_split_roots) > 0:
                target_dir = os.path.join(temp_split_roots[0], col)
                count = 0
                for src_file in paths[:split_counts[0]]:
                    target_file = src_file.replace(src_root, target_dir)
                    shutil.copyfile(src_file, target_file)
                    count += 1
                    print(f'{count} files copied to {target_dir}')

                # Remove items just done
                if split_counts[0] > 0:
                    paths = (paths[split_counts[0]:])
                temp_split_roots.remove(temp_split_roots[0])
                split_counts.remove(split_counts[0])

        return df

    @staticmethod
    def ensure_empty_directory(dir_path: str) -> str:
        """If path does not exist, create it. If it does exist, empty it.

        Keyword arguments:
        :param dir_path: root directory path
        :return: descriptor of result
        """
        result = 'Invalid'

        try:
            if not dir_path:
                raise ValueError('No value supplied for directory path.')

            if Path(dir_path).exists():
                if len(os.listdir(dir_path)) > 0:
                    result = 'Directory exists, not empty, deleting content'
                    # Clear sub-dirs first then tackle files
                    for root, dirs, files in os.walk(dir_path, topdown=False):
                        for directory in dirs:
                            to_remove = os.path.join(root, directory)
                            shutil.rmtree(to_remove)
                    for root, dirs, files in os.walk(dir_path, topdown=False):
                        for file in files:
                            to_remove = os.path.join(dir_path, file)
                            os.remove(to_remove)
                else:
                    result = 'Directory exists'
            else:
                result = 'Creating directory'
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        except ValueError as err:
            raise err

        except Exception as err:
            error_message = \
                "Unexpected error in FileTools.ensure_empty_directory\n"\
                + str(err.args)
            raise Exception(error_message)

        print('{}: {}'.format(result, dir_path))
        return result

    @staticmethod
    def lines_list_from_file(file_path: str) -> list:
        """Retrieve lines of text from file, return list

        Keyword arguments:
        :param file_path: full path to file
        :return: list of text lines
        """

        with open(file_path, 'r') as infile:
            line_list = infile.readlines()

        # Strip newline chars
        line_list = [line.strip() for line in line_list]

        return line_list

    @staticmethod
    def make_datetime_named_archive(src_path_to_archive: str, base_target_path: str,
                                    format: str = 'zip',
                                    datestamp: datetime = datetime.datetime.now()) -> str:
        """Make archive, name prefixed with current datetime (yyyymmdd_HHMM_).
        For more detail of each parameter, see definition of shutil.make_archive.

        Example usage:

        shutil.make_archive('/home/code/target_file_name', 'zip', '/home/code/', 'base_directory')

        Keyword arguments:

        :param src_path_to_archive: str, the path to the directory that is to be archived
        :param base_target_path: str, the full path of the file to create, including the base name, minus any format-specific
        extension; datetime will be prefixed to the base name
        :param format: str, the archive format (default: 'zip')
        :param datestamp: datestamp to add to archive name (default: now())
        :return: path of resultant archive file
        """
        print(f'Archiving {src_path_to_archive} to {format} file ...')
        if datestamp is None:
            datestamp = datetime.datetime.now()

        datestamp = datestamp.strftime('%y%m%d_%H%M')

        file_name = f'{datestamp}_{Path(base_target_path).name}'
        dir_path = Path(base_target_path).parent
        base_target_path = os.path.join(dir_path, file_name)

        root_dir = Path(src_path_to_archive).parent
        base_dir = Path(src_path_to_archive).name
        # print('\nmake_archive params etc')
        # print('base_target_path: {}'.format(base_target_path))
        # print('root_dir: {}'.format(root_dir))
        # print('base_dir: {}'.format(base_dir))

        result = shutil.make_archive(base_target_path, format, root_dir, base_dir)

        end_file_name = base_target_path + '.' + format

        print('Output saved at {}'.format(end_file_name))

        return result

    @staticmethod
    def save_command_args_to_file(args: dict, save_path: str, to_print: bool = False):
        """Save arguments and their values to file. Expects args of type dict, so use vars(args) as input.

        Keyword arguments:

        :param args: dict, full arguments list
        :param save_path: str, path to file
        :param to_print: bool, flag for printing args
        """
        datestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        header_line = f'Saved at {datestamp}'

        parts = ['python']
        lines = []
        parts.append(os.path.basename(sys.argv[0]))
        for item in sys.argv[1:]:
            parts.append(item)

        command_line = ' '.join(parts) + '\n'

        for k, v in args.items():
            lines.append('{}={}'.format(k, v or ''))

        lines.insert(0, command_line)
        lines.insert(0, header_line)

        content = '\n'.join(lines)

        with open(save_path, 'w', encoding='utf-8') as outfile:
            outfile.write(content)
            print('Command arguments saved to {}.'.format(save_path))

        if to_print:
            print(content)

    @staticmethod
    def create_numpy_archive_from_images_dir(src_dir: str, target_path: str,
                                             new_shape: tuple = 0,
                                             suffix: str = '.jpg'):
        """Create a numpy array archive of images sourced from a single directory.

        If new_shape is not provided, and images are of different dimensions, then this will generate
        an exception.

        Keyword arguments:

        :param src_dir: path to source directory
        :param target_path: path to final final, excluding extension
        :param new_shape: optional, end shape of resized image arrays
        :param suffix: suffix of images to be processed, including preceding full-stop (default '.jpg')
        """
        # Catch items where None passed in
        if new_shape is None:
            new_shape = 0
        if suffix is None:
            suffix = '.jpg'

        if src_dir == '':
            result = f'No source directory supplied for images, so no npy file created.'
        elif not Path(src_dir).is_dir():
            result = f'"{src_dir}" is not a directory, so no npy file created.'
        else:
            image_files = [os.path.join(src_dir, f) for f in os.listdir(src_dir)
                           if os.path.isfile(os.path.join(src_dir, f))
                           and Path(os.path.join(src_dir, f)).suffix == suffix]

            if len(image_files) == 0:
                result = f'No {suffix} files at {src_dir} so no npy file created.'
            else:
                processed_images = []

                try:
                    for img in [np.array(Image.open(image_path)) for image_path in image_files]:
                        processed_images.append(
                            np.asarray(
                                np.asarray(img, dtype='int') if new_shape == 0 else
                                resize(img, new_shape, preserve_range=True, anti_aliasing=False),
                                dtype='int'
                            )
                        )
                except Exception as err:
                    error_message = \
                        "Unexpected error in FileTools.create_numpy_archive_from_images_dir\n"\
                        + str(err.args)
                    raise Exception(error_message)

                final_path = target_path + '.npy'
                np.save(final_path, processed_images)

                result = f'Npy file saved at {final_path}'

        return result

    @staticmethod
    def path_of_first_file_of_type(directory: str, extension: str = '.jpg'):
        found_path = ''
        for root, dirs, files in os.walk(directory, topdown=False):
            for file in files:
                if Path(file).suffix == extension:
                    found_path = os.path.join(root, file)
                    break

        return found_path

    @staticmethod
    def dataset_type_from_name(name: str) -> str:
        """Return dataset type (train|validation|test) based on name.

        :param name: base name
        :return: dataset type
        """
        dataset_type = 'invalid'
        types = ['train', 'validation', 'test']
        for t in types:
            if name.startswith(t):
                dataset_type = t
                break

        return dataset_type

    @staticmethod
    def copy_dir_as_unclassed(source_dir: str, target_dir: str, replace_content: bool = False) -> str:
        dataset_type = FileTools.dataset_type_from_name(Path(source_dir).name)
        can_copy_files = replace_content

        if dataset_type == 'invalid':
            print(f'Invalid dataset type. Data not copied')
            can_copy_files = False
            return 'Invalid'

        # Need extra directory levels to allow PyTorch Dataset to work on unclassified data
        leaf_target_dir = os.path.join(target_dir, dataset_type, 'unknown')
        if replace_content or not Path(leaf_target_dir).exists():
            FileTools.ensure_empty_directory(leaf_target_dir)
            can_copy_files = True
        elif len(os.listdir(leaf_target_dir)) > 0:
            # There's content not getting replaced, get me out of here
            print(f'Not replacing content of {leaf_target_dir}')
            return 'Not replaced'
        else:
            Path(leaf_target_dir).mkdir(parents=True, exist_ok=True)
            can_copy_files = True

        if can_copy_files:
            for root, dirs, files in os.walk(source_dir, topdown=False):
                for file in files:
                    shutil.copy(os.path.join(source_dir, file), leaf_target_dir)

        return leaf_target_dir

    @staticmethod
    def collate_files_by_low_level_dir_name(source_dir: str, low_level_dir_name: str, path_parts_re: list)\
            -> np.ndarray:
        """
        Collate files within a regular structure but deep structure into an alternative one.
        Assume source files of interest are in commonly named sub-directories.

        E.g. Copy only those within 'Start' folders (this is the low_level_dir_name)
        from SourceDir/chXX/XX_NN/Start/filenameXX_NN.ext
        to TargetDir/XX_NN/filenameXX_NN.ext

        :param source_dir: top level source directory
        :param low_level_dir_name: lowest level commonly named directory, or common suffix
        :param path_parts_re: list of common parts of file paths to rename or remove, defined as regular expressions
        :return: data list of lists [file path, file name, copy path]
        """
        path = Path(source_dir)

        file_paths = []
        file_names = []

        for p in [p for p in path.rglob("*") if p.is_file() and p.parent.name.endswith(low_level_dir_name)]:
            file_paths.append(str(p))
            file_names.append(p.name)

        data = np.zeros(len(file_paths), dtype={'names': ('FilePath', 'FileName', 'CopyPath'),
                                                'formats': ('U256', 'U64', 'U256')})

        copy_paths = [str(fp) for fp in file_paths]

        for part in path_parts_re:
            copy_paths = [re.sub(part[0], part[1], fp) for fp in copy_paths]

        data['FilePath'] = file_paths
        data['FileName'] = file_names
        data['CopyPath'] = copy_paths

        for item in data:
            Path(Path(item['CopyPath']).parent).mkdir(parents=True, exist_ok=True)
            shutil.copy(item['FilePath'], item['CopyPath'])
            print(f'Copy {item["FileName"]} from {Path(item["FilePath"]).parent} to {Path(item["CopyPath"]).parent}')

        return data
