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

if __name__ == "__main__":
    start_row = 2 #todo: make this an external parameter
    root = tk.Tk()
    root.withdraw()
    dirname = filedialog.askdirectory()

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
            df2 = diff_df[diff_df.index % 2 != 0]
            df3 = diff_df[diff_df.index % 2 == 0]
            print(df2, df3)





