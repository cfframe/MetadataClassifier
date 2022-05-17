# plot_helper.py
# Targeted at training/validation vs epoch data

from enum import IntEnum
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import os
import pandas as pd


class ResultColumns(IntEnum):
    Epoch = 0
    TrainLoss = 1
    TrainAccuracy = 2
    ValLoss = 3
    ValAccuracy = 4


class PlotHelper:
    """Utilities for plotting graphs"""

    @staticmethod
    def legend_location_from_data(dataset: np.ndarray) -> str:
        """Attempt to calculate a good position for the legend location based on the data

        Assumes dataset is a 1 dimensional ndarray, and that its graph follows a general trend

        :param dataset: a dataset
        """

        # Base on the best fit slopes of the first and second halves of the dataset, without the first 2 data points
        # as the earliest epochs can be far from the general trend

        # Remove first two data points, and get length of sub sets
        dataset = dataset[2:]
        half_len = math.ceil(len(dataset) / 2.0)

        # Dataset is exceptionally small, so just set defaults
        if half_len <= 1:
            vertical = 'upper'
            horizontal = 'center'

        else:

            # Work out the average slopes of each half
            y1 = dataset[:half_len]
            y2 = dataset[-half_len:]
            x = np.asarray(range(0, half_len))

            m1 = (len(x) * np.sum(x * y1) - np.sum(x) * np.sum(y1)) / (len(x) * np.sum(x * x) - np.sum(x) ** 2)
            m2 = (len(x) * np.sum(x * y2) - np.sum(x) * np.sum(y2)) / (len(x) * np.sum(x * x) - np.sum(x) ** 2)

            vertical = 'upper' if m1 < m2 else 'lower'
            horizontal = 'right' if abs(m1) > abs(m2) else 'left'

        legend_location = f'{vertical} {horizontal}'

        return legend_location

    @staticmethod
    def basic_train_val_plot_and_save(title, y_label, train_data, validation_data, output_dir):
        """Plot pairs of datasets

        :param title: str -- figure title, used as base for saved file name
        :param y_label: str -- y-axis label
        :param train_data: dataset -- training results
        :param validation_data: dataset -- validation results
        :param output_dir: str -- target directory for saving plot
        """

        legend_location = PlotHelper.legend_location_from_data(train_data)
        target_path = os.path.join(output_dir, title.replace(' ', '_')+'.svg')

        # Force integer values for x axis.
        ax = plt.figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.plot(train_data, color='b', label='Training')
        plt.plot(validation_data, color='g', label='Validation')
        plt.title = title
        plt.ylabel(y_label)
        plt.xlabel('Epoch')
        plt.legend(['Training', 'Validation'], loc=legend_location)
        plt.grid()

        # If use plt.show() before saving, then saved figure is blank. Works ok other way round.
        plt.savefig(target_path)

        # plt.show()

    @staticmethod
    def basic_run_plot(df: pd.DataFrame, output_dir: str):
        PlotHelper.basic_train_val_plot_and_save(
            # style='seaborn',
            title='Accuracy',
            y_label='Accuracy',
            train_data=df[df.columns[ResultColumns.TrainAccuracy]],
            validation_data=df[df.columns[ResultColumns.ValAccuracy]],
            output_dir=output_dir)

        PlotHelper.basic_train_val_plot_and_save(
            # style='seaborn',
            title='Loss',
            y_label='Loss',
            train_data=df[df.columns[ResultColumns.TrainLoss]],
            validation_data=df[df.columns[ResultColumns.ValLoss]],
            output_dir=output_dir)
