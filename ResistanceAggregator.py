import logging
import os
import tkinter as tk
from tkinter import filedialog
import pandas as pd
from tqdm import tqdm
import logging
import os
import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator)

import pandas as pd
from tqdm import tqdm
import re

def open_file(fpath):
    """ Function to open a file and return its content
    :param fpath:   Path to the file to be opened
    :return:        String with file content
    """
    with open(fpath, 'r') as myfile:
        content = myfile.read()
    return content


def save_df_to_file(df, datapath, suffix):
    """
    Saves the data frame to the specified path
    :param df:          Dataframe to be saved
    :param datapath:    relative path to save location
    :param suffix:      File suffix = name after datapath
    :return:            true for successful save
    """
    # generate filename to save to
    file_dir = os.path.join(os.path.dirname(datapath), 'R_csv')
    filepath = os.path.join(file_dir, os.path.splitext(os.path.basename(datapath))[0])
    savepath = filepath + suffix + '.csv'

    df.to_csv(savepath, sep='\t')


def string_to_numpy(string, seperator='\t'):
    """"Turns a csv string of numbers into a 2D numpy array
    :param string:      input string that has csv format and only numbers
    :param seperator:   csv seperator within a row
    :return:            numpy array with values from csv string
    """

    # create list of rows
    rows = string.split('\n')
    row_list = []

    # turn each row into a np vector and save them in a list
    for row in rows:
        np_row = np.fromstring(row, dtype=float, sep=seperator)
        if not row == "":
            row_list.append(np_row)

    # stack both row vectors to one array
    np_array = np.vstack(row_list)

    return np_array


def plot_resistance_diff(df, datapath, suffix, title=None):
    """"
    :param df:          dataframe to plot.
    :param datapath:    Path to plot to.
    :param semilogy:    Bool that decides whether y axis is plotted in log scale.
    :param take_abs:    Bool that decides whether absolute value is plotted.
    :param title:       Displayed Title of the Plot
    :param suffix:      Suffix to datapath for plots.
    """
    # make plt-close non-blocking
    plt.ion()

    # generate filename to save to
    file_dir = os.path.join(os.path.dirname(datapath), 'R_plots')
    filepath = os.path.join(file_dir, os.path.splitext(os.path.basename(datapath))[0])
    filename_all = filepath + "_" + suffix + ".png"
    ax = df.plot()
    ax.set_ylabel('Resistance Ratio', fontsize=14)

    if title is None:
        plt.title(os.path.splitext(os.path.basename(datapath))[0])
    else:
        plt.title(title + ": " + os.path.splitext(os.path.basename(datapath))[0])


    minorLocator = AutoMinorLocator()
    ax.xaxis.set_minor_locator(minorLocator)
    yminorLocator = AutoMinorLocator()
    ax.yaxis.set_minor_locator(yminorLocator)


    ax.tick_params(axis='both', direction='in', top=True, right=True, which='both', width=1, length=4)
    ax.tick_params(axis='both', which='major', length=8)
    ax.xaxis.label.set_size(14)

    for side in ax.spines.keys():  # 'top', 'bottom', 'left', 'right'
        ax.spines[side].set_linewidth(2)




    plt.savefig(filename_all)

    plt.show()
    plt.close('all')



if __name__ == "__main__":
    start_row = 2 #todo: make this an external parameter
    root = tk.Tk()
    root.withdraw()
    dirname = filedialog.askdirectory()
    if not os.path.exists(os.path.join(dirname, 'R_csv')):
        os.makedirs(os.path.join(dirname, 'R_csv'))
        os.makedirs(os.path.join(dirname, 'R_plots'))

    for file in tqdm(os.listdir(dirname)):
        if file.endswith("Resistance.txt"):
            # open file and get data
            filename = os.path.join(dirname, file)
            # print(file)
            logging.info(f'Opening file: {filename}')
            resistances = open_file(filename)
            df = pd.DataFrame([x.split('\t') for x in resistances.split('\n')][start_row:-1])[[1,2]]
            df.columns = ["Resistance[\u03A9]", "Residue[\u03A9]"]
            df["Resistance[\u03A9]"] = df["Resistance[\u03A9]"].astype(float)
            df["Residue[\u03A9]"] = df["Residue[\u03A9]"].astype(float)
            print(df)
            diff_df = df / df.shift(1)
            df1 = diff_df[diff_df.index % 2 != 0]
            df2 = diff_df[diff_df.index % 2 == 0]


            save_df_to_file(df1, filename, '_diff1')
            save_df_to_file(df2, filename, '_diff2')
            plot_resistance_diff(df1, filename, '_diff1')
            plot_resistance_diff(df2, filename, '_diff2')





