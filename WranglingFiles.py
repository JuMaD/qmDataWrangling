import configparser
import datetime
import logging
import os
import tkinter as tk
from itertools import chain
from tkinter import filedialog, messagebox

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model

pd.options.compute.use_bottleneck = True

# todo: make a higher level object / df / dict that acummulates all junctions in one directory
# todo: implement tool that shows plots from several "higher level plots" in on figure
# todo: implement calc_tools: calculate current density, low bias resistance, ...

#################
# File Handling #
#################

def open_file(fpath):
    """ Function to open a file and return its content
    :param fpath:   Path to the file to be opened 
    :return:        String with file content
    """
    with open(fpath, 'r') as myfile:
        content = myfile.read()
    return content


def string_to_numpy(string, seperator='\t'):
    """Turns a csv string of numbers into a 2D numpy array
    :param string:      input string that has csv format and only numbers
    :param seperator:   csv seperator within a row
    :return:            numpy array with values from csv string
    """

    # create list of rows
    rows = string.split("\n")
    row_list = []

    # turn each row into a np vector and save them in a list
    for row in rows:
        np_row = np.fromstring(row, dtype=float, sep=seperator)
        if not row == "":
            row_list.append(np_row)

    # stack both row vectors to one array
    np_array = np.vstack(row_list)

    return np_array


def split_string(string, measurement_procedure='sweep', keyword='#Sweep:'):
    """Function that splits larger files into smaller chunks
    (e.g., when several sweeps are stored in same file) and returns the labels
    :param string:     The string to be split
    :param measurement_procedure:   The measurement procedure --> indicates filetype
    :param keyword:     The keyword to use in string.split()
    :return:            List of strings split by the given method
    """
    labels = None
    all_string_lists = None

    if measurement_procedure == 'sweep':
        # split at keyword
        list_strings = string.split(keyword)

        # preserve first row that are the labels
        labels = list_strings[0].strip().split("\t")

        # break down string into list that can be converted to numpy
        all_string_lists = []
        for i in range(0, len(list_strings) - 1):
            # remove first row of each split
            list_strings[i] = list_strings[i].split("\n", 1)[1]
            all_string_lists.append(list_strings[i])

    return labels, all_string_lists


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
    savepath = filepath + suffix

    df.to_csv(savepath, sep='\t')


########################
# Data Frame Functions #
########################
def make_pandas_df(np_arrays, labels, index_column=1):
    """Creates a Pandas dataframe joining both np_arrays at index_column
    :param  np_arrays:      Array containing both np_arrays to be joined
    :param  labels:         List of Strings containing the Labels - needs to have np_array.shape(0) elements
    :param  index_column:   index of the column to be used as index
    :return:                pandas dataframes for both, odd and even number in list - labeled array
    """

    final_df = None
    e_final_df = None
    o_final_df = None

    # turn numpy arrays into data frames and join them at the indicated index
    for i in range(0, len(np_arrays)):
        logging.debug(f'column {i}')

        np_array = np_arrays[i]
        num_labels = labels[:]

        data_frame = pd.DataFrame(data=np_array,
                                  columns=num_labels)
        data_frame.drop_duplicates('Voltage [V]', inplace=True)
        data_frame.set_index(num_labels[index_column], inplace=True)

        data_frame.columns = map(lambda col: '{}_{}'.format(str(col), i), data_frame.columns)

        if i == 0:
            final_df = data_frame.copy()
            e_final_df = data_frame.copy()
        else:
            if i == 1:
                o_final_df = data_frame.copy()
                final_df = join_dfs(final_df, data_frame)

            else:

                final_df = join_dfs(final_df, data_frame)

                # data_frames.append(data_frame)
                if i % 2 == 0:
                    # e_data_frames.append(data_frame)
                    e_final_df = join_dfs(e_final_df, data_frame)
                else:
                    # o_data_frames.append(data_frame)
                    o_final_df = join_dfs(o_final_df, data_frame)

        logging.debug('Data frames created')

        #todo: make sure odd and even have the same shape



    return final_df, o_final_df, e_final_df


def filter_df(df, filter_string='Current [A]'):
    """
    Returns dataframe with only the columns that have a label similar to filter
    :param df:      Data Frame to be filtered
    :param filter_string:  String that specified that column labels are filtered by
    :return:        Filtered data frame
    """
    filtered = df.filter(like=filter_string)

    return filtered


def join_dfs(ldf, rdf):
    """
    Joins two pandas dataframes with method "outer"
    :param ldf:     first data frame
    :param rdf:     second data frame
    :return:        joint data frame
    """
    return ldf.join(rdf, how='outer')


################
# Calculations #
################
# todo: implement LBR (Low bias resistance) estimator and save data

def calculate(functions):
    """
    Encapsulates all calculations to be made
    :param functions: List of functions to execute on the geiven data
    :return:
    """




def calc_stats(dfs):
    """Calls pandas describe on both dataframes in list dfs
    :param  dfs:    List of pandas dfs
    :return         pd df with stats"""

    # calculate stats for HRS & LRS
    oe_stats = []
    stats_df = []
    if len(dfs) == 1:
        stat = dfs[0].apply(pd.DataFrame.describe, axis = 1)
        stats_df.append(stat)
    else:
        for df in dfs:
            if df.name in ['odd', 'even']:
                stat = df.apply(pd.DataFrame.describe, axis=1)
                oe_stats.append(stat)

        stats_df = oe_stats[0].join(oe_stats[1], how='outer', lsuffix='_odd', rsuffix='_even')

    return stats_df


def calc_fowler_nordheim(dfs, alpha=2):
    """Calculates ln(I/V^alpha) and 1/V and returns a df to plot a fowler nordheim plot
       :param  dfs:    List of pandas dfs: both, odd, even
       :param alpha:   value for alpha in the ln calculation
       :return         pd df with 1/V"""

    fn_dfs = []
    for df in dfs:

        if df.name in ['odd', 'even']:

            # get x and y from df
            x = np.asarray(df.index.values.tolist())
            y = np.absolute(np.asarray(df))
            # Calculate ln(I/V^alpha) and 1/V
            reciprocal = np.reciprocal(x)

            power = np.power(reciprocal, alpha)
            power = np.expand_dims(power, axis=0).transpose()
            j_v = y * power
            log = np.log(j_v)

            # make it a df again
            data_frame = pd.DataFrame(data=log)
            data_frame.set_index(reciprocal, inplace=True)

            columns = []
            for column in range(0, len(data_frame.columns)):
                name = df.columns[column].split('_')[1]
                columns.append(f'ln(I/V^{alpha})_{name}')


            data_frame.columns = columns
            data_frame.index.name = '1/X'
            fn_dfs.append(data_frame)
    fn_df = fn_dfs[0].join(fn_dfs[1], how='outer', lsuffix='_odd', rsuffix='_even')
    fn_dfs.append(fn_df)

    fn_list = []

    for element in reversed(fn_dfs):
        fn_list.append(element)

    dfs_names = ['both', 'odd', 'even']
    for i in range(0, len(fn_list)):
        fn_list[i].name = dfs_names[i]

    return fn_list


def calc_memory_window(dfs, method="divide"):
    """
    Calculates the "memory window" i.e. the difference between the absolute values of odd and even sweeps.
    :param dfs:     List of data frames: both, odd, even
    :param method:  Method to calculate the window: divide or subtract
    :return:        Data frame containing the memory window for both sweeps
    """

    x = np.asarray(dfs[0].index.values.tolist())
    y_odd = None
    y_even = None
    delta1 = None
    delta2 = None

    for df in dfs:
        if df.name == 'odd':
            y_odd = np.asarray(df)
        if df.name == 'even':
            y_even = np.asarray(df)



    if method == "divide":
        delta1 = np.divide(np.abs(y_odd), np.abs(y_even))
        delta2 = np.divide(np.abs(y_even), np.abs(y_odd))

    elif method == "subtract":
        delta1 = np.subtract(np.maximum(np.abs(y_odd), np.abs(y_even)), np.minimum(np.abs(y_odd), np.abs(y_even)))
        delta2 = np.subtract(np.maximum(np.abs(y_odd), np.abs(y_even)), np.minimum(np.abs(y_odd), np.abs(y_even)))

    delta = np.maximum(delta1, delta2)

    # make it a df again
    data_frame = pd.DataFrame(data=delta)
    data_frame.set_index(x, inplace=True)

    columns = []
    for column in range(0, len(data_frame.columns)):
        name = dfs[0].columns[column].split('_')[1]
        columns.append(f'deltaI_{name}')

    data_frame.columns = columns
    data_frame.index.name = 'Voltage (V)'

    # change into list to make compatible with plot methods
    delta_list = [data_frame]

    return delta_list


def linear_fit(df, method='ransac', column=1):
    """
    Fits given Data. X values are assumed to be index of the df
    :param df:          DataFrame to be fit
    :param method:      Method used to fit the data e.g. ransac or linreg
    :param column:      Index of column with y data
    :return:            sklearn linear model
                        - use returned.predict(line_x) with line_x = np.arange(X.min(), X.max())[:, np.newaxis]
    """

    # get x and y from df
    x = np.asarray(df.index.values.tolist())
    y = np.asarray(df.iloc[:, column].tolist())

    # reshape to 2d so sklearn can work with it
    print(len(y))
    print(len(x))
    x = x.reshape((x.shape[0], 1))
    y = y.reshape((y.shape[0], 1))

    # Fit data accordingly

    if method == 'ransac':
        ransac = linear_model.RANSACRegressor()
        ransac.fit(x, y)

        return ransac

    if method == 'linreg':
        lr = linear_model.LinearRegression()
        lr.fit(x, y)
        return lr


###################
#  Visualizations #
###################

def plot_sweeps(df, datapath, suffix, semilogy=True, takeabs=True):
    """"
    :param df:          List of dataframes to plot in one plot.
    :param datapath:    Path to plot to.
    :param semilogy:    Bool that decides whether y axis is plotted in log scale.
    :param takeabs:     Bool that decides whether absolute value is plotted.
    :param suffix       Suffix to datapath for plots.
    """
    # make plt-close non-blocking
    plt.ion()

    # make colormap
    cmap_oddeven = make_colormap(len(df[0].columns))

    # generate filename to save to
    file_dir = os.path.join(os.path.dirname(datapath), 'plots')
    filepath = os.path.join(file_dir, os.path.splitext(os.path.basename(datapath))[0])
    filename_all = filepath + "_" + suffix + ".png"

    if takeabs:
        ax2 = df[0].abs().plot(colormap=cmap_oddeven)
    else:
        ax2 = df[0].plot(colormap=cmap_oddeven)

    ax2.set_ylabel(df[0].columns.values[0].split('_')[0])
    if semilogy:
        plt.semilogy()
    plt.title(os.path.splitext(os.path.basename(datapath))[0])
    plt.savefig(filename_all)

    plt.show()
    plt.close('all')


# noinspection PyShadowingNames,PyShadowingNames,PyShadowingNames
def plot_stats(stats_df, datapath, suffix, stats=None, ylabel='Current [A]', semilogy=True):
    """

    :param stats_df:    Dataframe containing the stats.
    :param datapath:    Path to save the Plot to.
    :param suffix:      Suffix for filename after datapath.
    :param stats:       List of stats to plot options are (mean, max, min, 25%,50%,75%).
    :param ylabel       Label displayed at y-axis.
    :param semilogy:    Set the graph to semilog.
    """

    if stats is None:
        stats = ['mean', 'min', 'max']
    # make 'plt.close' non-blocking
    plt.ion()

    # make colormap
    cmap_stats = mcolors.LinearSegmentedColormap.from_list('my_colormap',
                                                           ['#009933', '#cc3300', '#99ffbb', '#ffc6b3', '#33ff77',
                                                            '#ff8c66'])

    # generate filename to save to
    file_dir = os.path.join(os.path.dirname(datapath), 'plots')
    filepath = os.path.join(file_dir, os.path.splitext(os.path.basename(datapath))[0])
    filename_stats = filepath + "_" + suffix + ".png"


    if len(stats_df) == 1:

        ax = stats_df[0][stats].abs().plot(colormap=cmap_stats)

    else:
        # add suffix to labels so both joint columns can be shown
        stats_arr = []
        for string in stats:
            odd = string + '_odd'
            even = string + '_even'
            stats_arr.append(odd)
            stats_arr.append(even)

        # plot stats
        ax = stats_df[stats_arr].abs().plot(colormap=cmap_stats)

    ax.set_ylabel(ylabel)
    if semilogy:
        plt.semilogy()
    plt.title(os.path.splitext(os.path.basename(datapath))[0])
    plt.savefig(filename_stats)

    plt.show()
    plt.close('all')


#########################
# Convenience Functions #
#########################

def make_colormap(values=1024):
    """creates an intermixed colormap of two color maps to distingush between odd and even sweeps
      :param values   Number of total colors required, i.e. number of sweeps
    """

    # sample the colormaps - use less than full range to avoid white and black and twice the same color
    colors1 = plt.cm.Reds(np.linspace(0.3, 0.9, round(values / 2)))
    colors2 = plt.cm.Greens(np.linspace(0.3, 0.9, round(values / 2)))

    # combine them, alternating between both lists, and build a new colormap
    colors = list(chain.from_iterable(zip(colors1, colors2)))

    mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

    return mymap


"""""
  Pandas data frame methods that are good to know:
  
  e_currents.abs().plot()
  plt.title('e_Sweep')
  plt.semilogy()
  o_currents.abs().plot()
  plt.title('o_Sweep')
  plt.semilogy()

  # rowstats
  stats = currents.apply(pd.DataFrame.describe, axis=1)
  e_stats = e_currents.apply(pd.DataFrame.describe, axis=1)
  o_stats = o_currents.apply(pd.DataFrame.describe, axis=1)

  e_stats[['mean', 'max', 'min', '50%']].abs().plot()
  plt.semilogy()
  plt.title('e_Stats')
  o_stats[['mean', 'max', 'min']].abs().plot()
  plt.semilogy()
  plt.title('o_Stats')
  plt.grid(True)
  plt.show()

  final_df.reset_index(inplace=True)
  # print(final_df.head())"""

if __name__ == "__main__":
    logging.basicConfig(filename='RuntimeLog.log', level=logging.INFO)
    logging.info('LOGGING STARTED:' + str(datetime.datetime.now()))
    config = configparser.ConfigParser()
    ####################
    # Read Config File #
    ####################

    config.read('config.ini')


    try:
        initial_dir = config['Directory']['home_directory']
    except:
        initial_dir = '.'
        logging.error('No directory in config file!')

    logging.info('Config file read and parameters set.')

    ###############
    # Main Script #
    ###############

    while True:
        ####################
        # Select Directory #
        ####################

        root = tk.Tk()
        root.withdraw()
        dirname = filedialog.askdirectory(initialdir=initial_dir)
        config.set('Directory', 'home_directory', os.path.dirname(dirname))
        csv_path = os.path.join(dirname, 'csv')
        if not os.path.exists(csv_path):
            os.makedirs(csv_path)
            os.makedirs(os.path.join(dirname, 'plots'))

        ################
        # Wrangle Data #
        ################
        for file in os.listdir(dirname):
            if file.endswith(".txt") and not file.endswith("Resistance.txt") and not file.endswith("SMU-Puls.txt"):
                        #############################
                        # Make Dataframes from file #
                        #############################

                        # todo: encapsulate make_dfs, perform_calculations & save_data

                        # open file and get data
                        filename = os.path.join(dirname, file)
                        logging.info(f'Opening file: {filename}')
                        # split file into several sweeps and get labels
                        labels, string_list = split_string(open_file(filename))
                        np_arrays = []
                        # make numpy array for every sweep and add data to list
                        for string in string_list:
                            if string == "":
                                continue
                            np_arrays.append(string_to_numpy(string))
                        # turn both data into Pandas data frame, using voltage to join
                        both, odd, even = make_pandas_df(np_arrays, labels, 1)

                        if odd is not None and even is not None:
                            dfs = [both, odd, even]
                            dfs_names = ['both', 'odd', 'even']

                            # get only currents
                            currents = []

                            for d in range(0, len(dfs)):
                                current = filter_df(dfs[d], 'Current [A]')
                                current.name = dfs_names[d]
                                currents.append(current)

                            ########################
                            # Perform Calculations #
                            ########################

                            # get stats on currents
                            stats_df = calc_stats(currents)

                            # calculate fn data
                            fn_df = calc_fowler_nordheim(currents)
                            fn_stats = calc_stats(fn_df)

                            # Memory window

                            window_df = calc_memory_window(currents)
                            window_stats = calc_stats(window_df)


                            #################
                            # Save To Files #
                            #################

                            # todo:make tosave a selectable in GUI

                            tosave = {  # "both": currents[0],
                                "all_abs": currents[0].abs(),
                                "stats": stats_df,
                                "stats_abs": stats_df.abs(),
                                "fn": fn_df[0],
                                "fn_stats": fn_stats,
                                "mwindow": window_df[0],
                            }

                            for key, value in tosave.items():
                                save_df_to_file(value, filename, '_' + key)

                            #############
                            # Visualize #
                            #############

                            # todo:make tosave a selectable in GUI
                            # todo: clean up plot fn_stats
                            # todo: (just rewrite "toplot" in bool dict and then for for key ... + use plot_stats)
                            toplot = {  # "both": currents[0],
                                "both": True,
                                "stats": True,
                                "fn": True,
                                "fn_stats": True,
                                "mwindow": True,
                            }

                            if toplot["both"]:
                                plot_sweeps(currents, filename, suffix='both')
                            if toplot["stats"]:
                                plot_stats(stats_df, filename, suffix='stats')
                            if toplot["fn"]:
                                plot_sweeps(fn_df, filename, suffix='fn', semilogy=False, takeabs=False)
                            if toplot["fn_stats"]:
                                plot_stats(fn_stats, filename, suffix='fn_stats')
                            if toplot["mwindow"]:
                                plot_sweeps(window_df, filename, suffix='mwindow', semilogy=True, takeabs=False)
                                plot_stats(window_stats, filename, suffix='window_stats')

                            # for key, value in toplot.items():
                            #   visualizeSweeps(currents, stats_df, filename)

        again = messagebox.askyesno("Finished!", f"Finished wrangling files in {dirname}!\n Select another directory?")

        if again:
            continue
        else:
            break

    #####################
    # write config file #
    #####################

    cfgfile = open('config.ini', 'w')
    config.write(cfgfile)
    cfgfile.close()
    logging.info('Saved config file')
