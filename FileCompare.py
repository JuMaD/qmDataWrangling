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

import pandas as pd
from tqdm import tqdm


def join_dfs(ldf,rdf):
    return ldf.join(rdf, how='outer')


def aggregate_averages():
    root = tk.Tk()
    root.withdraw()
    dirname = filedialog.askdirectory()

    df_dict = {}
    df_odd = []
    df_even = []
    for file in tqdm(os.listdir(dirname)):
        if file.endswith("_stats-Current_[A].csv"):
            #############################
            # Make Dataframes from file #
            #############################

            # open file and get data
            filename = os.path.join(dirname, file)
            #print(file)
            logging.info(f'Opening file: {filename}')
            df = pd.read_csv(filename, sep='\t').set_index('Voltage [V]')

            df_odd.append(df.mean_odd)

            df_dict[f'{file}'] = df


    odd_mean = pd.concat(df_odd, axis=1)

    print(odd_mean.head())




if __name__ == "__main__":
    aggregate_averages()