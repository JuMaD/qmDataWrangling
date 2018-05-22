import functools
import os
import gc
import tkinter as tk
import datetime

from tkinter import filedialog, messagebox

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from itertools import chain
from sklearn import linear_model

import logging

import configparser

pd.options.compute.use_bottleneck = True


# todo: make a higher level object / df / dict that acummulates all junctions in one directory
# todo: implement tool that shows plots from several "higher level plots" in on figure
# todo: implement calc_tools: calculate current density, low bias resistance, ...

#################
# File Handling #
#################

def openFile(fpath):
    """ Function to open a file and return its content
    :param fpath:   Path to the file to be opened 
    :return:        String with file content
    """
    with open(fpath, 'r') as myfile:
        content = myfile.read()
    return content


def stringToNumpy(string, seperator='\t'):
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

    # stack all row vectors to one array
    np_array = np.vstack(row_list)

    return np_array


def splitString(string, measurement_procedure='sweep', keyword='#Sweep:'):
    """Function that splits larger files into smaller chunks (e.g., when several sweeps are stored in same file) and returns the labels
    :param string:     The string to be split
    :param measurement_procedure:   The measurement procedure --> indicates filetype
    :param keyword:     The keyword to use in string.split()
    :return:            List of strings split by the given method
    """

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


def saveDfToFile(df, datapath, suffix):
    """
    Saves the data frame to the specified path
    :param df:      Dataframe to be saved
    :param path:    relative path to save location
    :return:        true for successful save
    """
    # generate filename to save to
    dir = os.path.join(os.path.dirname(datapath), 'csv')
    filepath = os.path.join(dir, os.path.splitext(os.path.basename(datapath))[0])
    savepath = filepath + suffix

    df.to_csv(savepath, sep='\t')


########################
# Data Frame Functions #
########################
def makePandasDf(np_arrays, labels, index_column=1):
    """Creates a Pandas Dataframe joining all np_arrays at index_column
    :param  np_arrays:      Array containing all np_arrays to be joined
    :param  labels:         List of Strings containing the Labels - needs to have np_array.shape(0) elements
    :param  index_column:   index of the column to be used as index
    :return:                pandas dataframes for all, odd and even number in list - labeled array
    """

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

            else:

                final_df = joinDfs(final_df, data_frame)

                # data_frames.append(data_frame)
                if i % 2 == 0:
                    # e_data_frames.append(data_frame)
                    e_final_df = joinDfs(e_final_df, data_frame)
                else:
                    # o_data_frames.append(data_frame)
                    o_final_df = joinDfs(o_final_df, data_frame)

    # final_df = functools.reduce(joinDfs, data_frames)
    # e_final_df = functools.reduce(joinDfs, e_data_frames)
    # o_final_df = functools.reduce(joinDfs, o_data_frames)
    logging.debug('Data frames created')
    return final_df, o_final_df, e_final_df


def filterDf(df, filter='Current [A]'):
    """
    Returns dataframe with only the columns that have a label similar to filter
    :param df:      Data Frame to be filtered
    :param filter:  String that specified that column labels are filtered by
    :return:        Filtered data frame
    """
    filtered = df.filter(like=filter)

    return filtered


def joinDfs(ldf, rdf):
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
# todo: implement memory window calculation

def CalcStats(dfs):
    """Calls pandas describe on all dataframes in list dfs
    :param  dfs:    List of pandas dfs
    :return         pd df with stats"""

    # calculate stats for HRS & LRS
    oe_stats = []
    for df in dfs:
        if df.name in ['odd', 'even']:
            stat = df.apply(pd.DataFrame.describe, axis=1)
            oe_stats.append(stat)

    stats_df = oe_stats[0].join(oe_stats[1], how='outer', lsuffix='_odd', rsuffix='_even')

    return stats_df

def CalcFN(dfs, alpha=2, column=1):
    """Calculates ln(I/V^alpha) and 1/V and returns a df to plot a fowler nordheim plot
       :param  dfs:    List of pandas dfs
       :return         pd df with 1/V"""

    oe_fn = []
    for df in dfs:
        if df.name in ['odd', 'even']:

            # get x and y from df
            x = np.asarray(df.index.values.tolist())
            y = np.absolute(np.asarray(df.iloc[:, column].tolist()))

            # Calculate ln(I/V^alpha) and 1/V
            reciprocal = np.reciprocal(x)
            power = np.power(reciprocal, alpha)
            j_v = np.multiply(power, y)
            log = np.log(j_v)

            #make it a df again
            data_frame = pd.DataFrame(data=log)
            


    fn_df = oe_fn[0].join(oe_fn[1], how='outer', lsuffix='_odd', rsuffix='_even')

    return fn_df

def linearFit(df, method='ransac', column=1):
    """
    Fits given Data. X values are assumed to be index of the df
    :param df:          DataFrame to be fit
    :param method:      Method used to fit the data e.g. ransac or linreg
    :param column:      Index of column with y data
    :return:            sklearn linear model - use returned.predict(line_x) with line_x = np.arange(X.min(), X.max())[:, np.newaxis]
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
        lr.fit(X, y)
        return lr

###################
#  Visualizations #
###################

def plotSweeps(df,datapath, suffix, semilogy=True):
    """"
    :param df:          List of dataframes to plot in one plot.
    :param datapath:    Path to plot to.
    :param suffix       Suffix to datapathh for plots.
    """
    #make plt-close non-blocking
    plt.ion()

    #make colormap
    cmap_oddeven = makeColormap(len(df[0].columns))

    # generate filename to save to
    dir = os.path.join(os.path.dirname(datapath), 'plots')
    filepath = os.path.join(dir, os.path.splitext(os.path.basename(datapath))[0])
    filename_all = filepath + "_"+suffix + ".png"

    ax2 = df[0].abs().plot(colormap=cmap_oddeven)
    ax2.set_ylabel(df[0].columns.values[0].split('_')[0])
    if semilogy:
        plt.semilogy()
    plt.title(os.path.splitext(os.path.basename(datapath))[0])
    plt.savefig(filename_all)

    plt.show()
    plt.close('all')

def plotStats(stats_df, datapath, stats=['mean', 'min', 'max'], ylabel='Current [A]', semilogy=True):
    """

    :param stats_df:    Dataframe containing the stats.
    :param datapath:    Path to save the Plot to.
    :param stats:       List of stats to plot options are (mean, max, min, 25%,50%,75%)
    :param ylabel       Label displayed at y-axis
    :param semilogy:    Set the graph to semilog
    """
    # make 'plt.close' non-blocking
    plt.ion()

    # make colormap
    cmap_stats = mcolors.LinearSegmentedColormap.from_list('my_colormap',
                                                           ['#009933', '#cc3300', '#99ffbb', '#ffc6b3', '#33ff77',
                                                            '#ff8c66'])

    # generate filename to save to
    dir = os.path.join(os.path.dirname(datapath), 'plots')
    filepath = os.path.join(dir, os.path.splitext(os.path.basename(datapath))[0])
    filename_stats = filepath + "_stats.png"


    # add suffix to labels so all joint columns can be shown
    stats_arr = []
    for string in stats:
        odd = string + '_odd'
        even = string + '_even'
        stats_arr.append(odd)
        stats_arr.append(even)

    # plot stats
    ax = stats_df[stats_arr].abs().plot(colormap=cmap_stats)
    ax.set_ylabel(ylabel)
    if semilogy == True:
        plt.semilogy()
    plt.title(os.path.splitext(os.path.basename(datapath))[0])
    plt.savefig(filename_stats)

    plt.show()
    plt.close('all')


#########################
# Convenience Functions #
#########################

def makeColormap(values=1024):
    """creates an intermixed colormap of two color maps to distingush between odd and even sweeps
      :param values   Number of total colors required, i.e. number of sweeps
    """

    # sample the colormaps - use less than full range to avoid white and black and twice the same color
    colors1 = plt.cm.Reds(np.linspace(0.3, 0.9, values / 2))
    colors2 = plt.cm.Greens(np.linspace(0.3, 0.9, values / 2))

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
    logging.info('LOGGING STARTED:'+str(datetime.datetime.now()))
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
            if file.endswith(".txt"):
                # todo: Change this into a switch-case structure to fetch all three types of txt
                if not file.endswith("Resistance.txt"):
                    if not file.endswith("SMU-Puls.txt"):

                        #############################
                        # Make Dataframes from file #
                        #############################


                        # todo: encapsulate makeDfs, performCalculations & save data

                        # open file and get data
                        filename = os.path.join(dirname, file)
                        logging.info(f'Opening file: {filename}')
                        # split file into several sweeps and get labels
                        labels, string_list = splitString(openFile(filename))
                        np_arrays = []
                        # make numpy array for every sweep and add data to list
                        for string in string_list:
                            if string == "":
                                continue
                            np_arrays.append(stringToNumpy(string))
                        # turn all data into Pandas data frame, using voltage to join
                        all, odd, even = makePandasDf(np_arrays, labels, 1)

                        dfs = [all, odd, even]
                        dfs_names = ['all', 'odd', 'even']

                        # get only currents
                        currents = []

                        for d in range(0, len(dfs)):
                            current = filterDf(dfs[d], 'Current [A]')
                            current.name = dfs_names[d]
                            currents.append(current)
                        ########################
                        # Perform Calculations #
                        ########################

                        # get stats on currents
                        stats_df = CalcStats(currents)

                        #calculate fn plots
                        #fn_df = CalcFN(currents)





                        #################
                        # Save To Files #
                        #################

                        #todo:make tosave a selectable in GUI

                        tosave = {  # "all": currents[0],
                            "all_abs": currents[0].abs(),
                            # "stats": stats_df ,
                            "stats_abs": stats_df.abs(),
                            # "fn":fn_df
                        }

                        for key, value in tosave.items():
                            saveDfToFile(value, filename,'_'+key)



                        #############
                        # Visualize #
                        #############

                        # todo:make tosave a selectable in GUI
                        toplot = {  # "all": currents[0],
                            "all_abs": currents[0].abs(),
                            # "stats": stats_df ,
                            "stats_abs": stats_df.abs(),
                            # "fn":fn_df
                        }

                        plotSweeps(currents, filename, suffix='all')
                        plotStats(stats_df.abs(), filename)
                        #for key, value in toplot.items():
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
