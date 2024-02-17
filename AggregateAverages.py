import logging
import os
import re
import tkinter as tk
from tkinter import filedialog

import pandas as pd
from tqdm import tqdm


def join_dfs(ldf,rdf):
    return ldf.join(rdf, how='outer')


def aggregate_averages(mode):
    """
    Aggregates all mean values of the measurements in a selected folder.
    :param mode:    selector to determine if current or current density values should be aggregated
    :return: The DataFrames containing the values of all even (even_mean) and odd (odd_mean) sweeps,
    as well as the selected folder dirname
    """
    root = tk.Tk()
    root.withdraw()
    dirname = filedialog.askdirectory()

    df_dict = {}
    df_columns = []
    df_odd = []
    df_even = []
    if mode == 'current':
        ending = "_stats-Current_[A].csv"
    elif mode == 'density':
        ending = "_stats-Current_[A].csv"

    for file in tqdm(os.listdir(dirname)):
        if file.endswith(ending):
            #############################
            # Make Dataframes from file #
            #############################

            # open file and get data
            filename = os.path.join(dirname, file)
            #print(file)
            logging.info(f'Opening file: {filename}')
            df = pd.read_csv(filename, sep='\t').set_index('Voltage [V]')
            regex_size = r'(?<=PS)(.*?)(?=u)'
            regex_device = r'(?<=u)(.*?)(?=I)'
            match_size = re.search(regex_size, file)
            match_device = re.search(regex_device, file)
            junction_size = match_size.group(0)+'um'
            device_number = match_device.group(0)[:-1]
            print(device_number)


            df.mean_odd.columns = f'{junction_size} ({device_number})'
            df_columns.append(df.mean_odd.columns)
            df_odd.append(df.mean_odd)

            df.mean_even.columns = f'{junction_size} ({device_number})'
            df_even.append(df.mean_even)

            df_dict[f'{file}'] = df


    odd_mean = pd.concat(df_odd, axis=1)
    even_mean = pd.concat(df_even, axis=1)
    print(df_columns)
    odd_mean.columns = df_columns
    print(odd_mean.head())
    print(even_mean.head())
    return even_mean, odd_mean, dirname


def save_df_to_file(df, datapath, suffix):
    """
    Saves the data frame to the specified path
    :param df:          Dataframe to be saved
    :param datapath:    relative path to save location
    :param suffix:      File suffix = name after datapath
    :return:            true for successful save
    """
    # generate filename to save to
    file_dir = os.path.join(os.path.dirname(datapath), 'csv')
    filepath = os.path.join(file_dir, os.path.splitext(os.path.basename(datapath))[0])
    savepath = filepath + suffix + '.csv'

    df.to_csv(savepath, sep='\t')


if __name__ == "__main__":
    mode = 'current'
    even_df, odd_df, dirname = aggregate_averages(mode=mode)
    save_df_to_file(even_df, dirname, f'{mode}-averages-even')
    save_df_to_file(odd_df, dirname, f'{mode}-averages-odd')