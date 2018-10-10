import configparser
import datetime
import logging
import os
import re
import tkinter as tk
from decimal import Decimal
from itertools import chain
from tkinter import filedialog, messagebox

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import r2_score

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
        # split header and main string
        parts = string.split("\n\n")

        # todo: use header values in functions! (dictonary- name:value is available already)

        header_values = {}
        if len(parts) > 1:
            header_lines = parts[0].split('\n')
            for line in header_lines:
                header_name = re.search('(?<=#)(.*?)(?=[:,=])', line).group(0)
                regex_value = '(?<=' + re.escape(header_name) + ')[^#]+'
                header_value = re.search(regex_value, line).group(0)[2:]
                header_values[header_name.split(" ")[0].replace(" ", "")] = header_value.replace(" ", "_")

        # split at keyword
        list_strings = parts[len(parts)-1].split(keyword)

        # preserve first row that are the labels
        labels = list_strings[0].strip().split("\t")

        # break down string into list that can be converted to numpy
        all_string_lists = []
        for i in range(1, len(list_strings) - 1):
            # remove first row of each split
            list_strings[i] = list_strings[i].split("\n", 1)[1]
            all_string_lists.append(list_strings[i])

    return labels, all_string_lists, header_values


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
def make_pandas_df(np_arrays, labels, index_column=1, start_index=2):
    """Creates a Pandas dataframe joining both np_arrays at index_column
    :param  np_arrays:      Array containing both np_arrays to be joined
    :param  labels:         List of Strings containing the Labels - needs to have np_array.shape(0) elements
    :param  index_column:   index of the column to be used as index
    :param start_index      Column index at which the returned pandas dataframes
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

    final_df = final_df[final_df.columns[2*start_index:final_df.shape[0]]]

    o_final_df = o_final_df[o_final_df.columns[start_index:o_final_df.shape[0]]]
    e_final_df = e_final_df[e_final_df.columns[start_index:e_final_df.shape[0]]]

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
# todo: encapsulate all calculations inside calculate
functions = {"resistance": {"all": True, "stats": True}, "fn": {"all": True, "stats": True}}


def calculate(functions):
    """
    Encapsulates all calculations to be made
    :param functions:   Dictionary of functions to execute on the given data
    :return:
    """
    for function in functions.keys():
        if functions[function]['all']:
            print(function + " all " + str(functions[function]['all']))
        if functions[function]['stats']:
            print(function + " stats " + str(functions[function]['stats']))


def calc_stats(dfs):
    """Calls pandas describe on both dataframes in list dfs
    :param  dfs:    List of pandas dfs
    :return         pd df with stats"""

    # calculate stats for HRS & LRS
    oe_stats = []
    stats_df = []

    if len(dfs) == 1:
        stat = dfs[0].apply(pd.DataFrame.describe, axis=1)
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
            y_odd = np.copy(np.asarray(df))
        if df.name == 'even':
            y_even = np.copy(np.asarray(df))

    if y_even.shape != y_odd.shape:

        y_even.resize(y_odd.shape)
        x.resize(y_odd.shape[0])

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
        columns.append(f'$\Delta I$_{name}')

    data_frame.columns = columns
    data_frame.index.name = 'Voltage (V)'

    # change into list to make compatible with plot methods
    delta_list = [data_frame]

    return delta_list


def calc_diff_resistance(df, window_range=0.1, fit_method='ransac'):
    """Calculates the differential resistance
    :param df:  Data frame with IV data
    :param window_range: range defining the subset of df that is used for the fit (see calc_linear_fit)
    :param fit_method: fit method used to fit the data (see calc_linear_fit)
    :return  list containing dfs ('all', 'even', 'odd') with the result of the calculations
    """
    # fill nan values with 0s
    df = df.fillna(0)

    voltages = df.index.values.tolist()

    max_voltage = np.max(voltages)
    last = voltages[np.abs(voltages - (max_voltage - window_range / 2)).argmin() - 1]

    min_voltage = np.min(voltages)
    first = voltages[np.abs(voltages - (min_voltage + window_range / 2)).argmin() + 1]

    voltage_list = [num for num in voltages if first <= num <= last]

    for c in range(len(df.columns)):
        df_column = df.iloc[:, [c]]
        resistance_list = []
        r2_list = []

        for voltage in voltage_list:
            try:
                fit_results = calc_linear_fit(df_column, column=0, start=voltage, method=fit_method,
                                              fit_range=window_range)
                resistance_list.append(fit_results["resistance"])
                r2_list.append(fit_results["r2"])
            except:
                resistance_list.append(np.nan)
        print(f'{c+1}/{len(df.columns)}')

        if c == 0:
            resistance_df = pd.DataFrame({'Voltage [V]': voltage_list,
                                          f'Resistance [$\Omega$]_{c}': resistance_list})
            resistance_df.set_index('Voltage [V]', inplace=True)
        else:
            resistance_df[f'Resistance [$\Omega$]_{c}'] = pd.Series(resistance_list, index=resistance_df.index)

    e_resistance_df = resistance_df.iloc[0:, 0::2].copy()
    o_resistance_df = resistance_df.iloc[0:, 1::2].copy()

    resistance_df.name = 'all'
    e_resistance_df.name = 'even'
    o_resistance_df.name = 'odd'

    resistance_df_list = [resistance_df, e_resistance_df, o_resistance_df]

    return resistance_df_list


def calc_linear_fit(df, fit_range=1, start=0, method='ransac', column=1, debug=False, datapath=None, title=None):
    """
    Fits the slice [ start - 1 * fit_range / 2, start + fit_range / 2] of given Data.
    X values are assumed to be index of the df.

    :param df:          DataFrame to be fit
    :param start        middle point of the range to be analyzed
    :param fit_range    Range of x values to be used for fitting
    :param method:      Method used to fit the data e.g. ransac or linreg
    :param column:      Index of column with y data
    :param debug:       Turns off/on the debug function: plotting the fit and displaying the return
    :paran title:       Displayed Title
    :param datapath     Datapath being displayed in the debug plot
    :return:            Dictionarry with {coefficent, resistance, R2}

    """
    # todo: make it a rolling regression over the whole dataset
    # get x and y from df
    # get only a part of the data frame and extract numpy array
    mask = (df.index > -1 * fit_range / 2 + start) & (df.index <= fit_range / 2 + start)
    masked_df = df.loc[mask]
    x = np.asarray(masked_df.index.values.tolist())
    y = np.asarray(masked_df.iloc[:, column].tolist())

    # reshape to 2d so sklearn can work with it
    x = x.reshape((x.shape[0], 1))
    y = y.reshape((y.shape[0], 1))

    # Fit data accordingly

    if method == 'ransac':

        # fit and get parameters

        ransac = linear_model.RANSACRegressor()
        ransac.fit(x, y)

        coef = ransac.estimator_.coef_[0][0]
        intercept = ransac.estimator_.intercept_[0]
        # resistance = "{:.2e}".format(Decimal(1 / coef))
        resistance = 1 / coef
        line_x = np.linspace(x.min(), x.max(), x.shape[0])[:, np.newaxis]
        line_y_ransac = ransac.predict(line_x)
        r2 = r2_score(y, line_y_ransac)

        if debug:
            print(f'Fit Function: y = {coef}x + {intercept}')
            print(f'R^2: {r2}')
            print(f'Resistance: {resistance} $\Omega$')

            plt.scatter(x, y, color='yellowgreen', marker='.',
                        label='datapoints')
            plt.plot(line_x, line_y_ransac, color='cornflowerblue', linewidth=2,
                     label='RANSAC regressor')

            if title is None:
                plt.title(os.path.splitext(os.path.basename(datapath))[0])
            else:
                plt.title(title + ": " + os.path.splitext(os.path.basename(datapath))[0])

            plt.show()

        return {'coefficient': coef, 'resistance': resistance, 'intercept': intercept, 'R2': r2}

    if method == 'linreg':
        lr = linear_model.LinearRegression()
        lr.fit(x, y)

        coef = lr.coef_[0][0]
        intercept = lr.intercept_[0]
        resistance = "{:.2e}".format(Decimal(1 / coef))

        line_x = np.linspace(x.min(), x.max(), x.shape[0])[:, np.newaxis]
        line_y = lr.predict(line_x)
        r2 = r2_score(y, line_y)

        if debug:
            print(f'Fit Function: y = {coef}x + {intercept}')
            print(f'R^2: {r2}')
            print(f'Resistance: {resistance} $\Omega$')

            plt.scatter(x, y, color='yellowgreen', marker='.',
                        label='datapoints')
            plt.plot(line_x, line_y, color='cornflowerblue', linewidth=2,
                     label='LINREG regressor')
            if title is None:
                plt.title(os.path.splitext(os.path.basename(datapath))[0])
            else:
                plt.title(title + ": " + os.path.splitext(os.path.basename(datapath))[0])

            plt.show()

        return {'coefficient': coef, 'resistance': resistance, 'intercept': intercept, 'R2': r2}


def get_slice(resistance_df, at=0):
    """
    #todo:refine get_slice docstring
    :param resistance_df:   datafrane to slice
    :param at:  value that is compared to index value to slice
    :return: slice at position
    """
    voltages = resistance_df.index.values.tolist()
    zero = voltages[np.abs(np.subtract(voltages, at)).argmin()]
    sliced = resistance_df.loc[zero]

    return sliced


###################
#  Visualizations #
###################

def plot_sweeps(df, datapath, suffix, semilogy=True, take_abs=True, title=None):
    """"
    :param df:          List of dataframes to plot in one plot.
    :param datapath:    Path to plot to.
    :param semilogy:    Bool that decides whether y axis is plotted in log scale.
    :param take_abs:    Bool that decides whether absolute value is plotted.
    :param title:       Displayed Title of the Plot
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

    if take_abs:
        ax2 = df[0].abs().plot(colormap=cmap_oddeven)
    else:
        ax2 = df[0].plot(colormap=cmap_oddeven)

    ax2.set_ylabel(df[0].columns.values[0].split('_')[0])
    if semilogy:
        plt.semilogy()
    if title is None:
        plt.title(os.path.splitext(os.path.basename(datapath))[0])
    else:
        plt.title(title + ": " + os.path.splitext(os.path.basename(datapath))[0])
    plt.savefig(filename_all)

    plt.show()
    plt.close('all')


# noinspection PyShadowingNames,PyShadowingNames,PyShadowingNames
def plot_stats(stats_df, datapath, suffix, stats=None, y_label='Current [A]', semilogy=True, take_abs=True, title=None):
    """

    :param stats_df:    Dataframe containing the stats.
    :param datapath:    Path to save the Plot to.
    :param suffix:      Suffix for filename after datapath.
    :param stats:       List of stats to plot options are (mean, max, min, 25%,50%,75%).
    :param y_label      Label displayed at y-axis.
    :param semilogy:    Set the graph to semilog.
    :param take_abs:    Take abs before plotting if true.
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
        if take_abs:
            ax = stats_df[0][stats].abs().plot(colormap=cmap_stats)
        else:
            ax = stats_df[0][stats].plot(colormap=cmap_stats)

    else:
        # add suffix to labels so both joint columns can be shown
        stats_arr = []
        for string in stats:
            odd = string + '_odd'
            even = string + '_even'
            stats_arr.append(odd)
            stats_arr.append(even)

        # plot stats

        if take_abs:
            ax = stats_df[stats_arr].abs().plot(colormap=cmap_stats)
        else:
            ax = stats_df[stats_arr].plot(colormap=cmap_stats)

    ax.set_ylabel(y_label)
    if semilogy:
        plt.semilogy()

    if title is None:
        plt.title(os.path.splitext(os.path.basename(datapath))[0])
    else:
        plt.title(title + ": " + os.path.splitext(os.path.basename(datapath))[0])

    plt.savefig(filename_stats)

    plt.show()
    plt.close('all')


def plot_slice(df, datapath, suffix, title):
    # make plt-close non-blocking
    plt.ion()

    # make colormap
    # cmap_oddeven = make_colormap(len(df[0].columns))

    # generate filename to save to
    file_dir = os.path.join(os.path.dirname(datapath), 'plots')
    filepath = os.path.join(file_dir, os.path.splitext(os.path.basename(datapath))[0])
    filename_all = filepath + "_" + suffix + ".png"

    plt.scatter(df.index, df)
    # ax2.set_ylabel(df.values[0].split('_')[0])

    if title is None:
        plt.title(os.path.splitext(os.path.basename(datapath))[0])
    else:
        plt.title(title + ": " + os.path.splitext(os.path.basename(datapath))[0])
    plt.savefig(filename_all)

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
    if values == 1:
        return plt.cm.Greens
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
                labels, string_list, header = split_string(open_file(filename))
                print(header)
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
                        # todo: implement something to choose Current or Current Density
                        filter = 'Current [A]'
                        current = filter_df(dfs[d], filter)
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

                    # linear fit --- TEST AREA
                    # todo: encapsulate (howto: pass function arguments??)
                    # todo: change so that multiple columns can be processed!

                    resistance_dfs = calc_diff_resistance(currents[0])

                    resistance_slice = get_slice(resistance_dfs[0])

                    resistance_stats = calc_stats(resistance_dfs)

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
                        "mwindow_stats": window_stats[0],
                        "resistance": resistance_stats,
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
                        "resistance": True,
                        "resistance_slice": True
                    }

                    if 'Material-ID' in header:
                        title = header['Material-ID']
                    else:
                        title = None

                    if toplot["mwindow"]:
                        plot_sweeps(window_df, filename, suffix='mwindow', semilogy=True, take_abs=False, title=title)
                        plot_stats(window_stats, filename, y_label=r'$\Delta I$', suffix='mwindow_stats', title=title)
                    if toplot["resistance"]:
                        plot_sweeps(resistance_dfs, filename, suffix='resistance',
                                    semilogy=False, take_abs=False, title=title)
                        plot_stats(resistance_stats, filename, y_label=r'Resistance [$\Omega$]',
                                   suffix='resistance_stats', title=title)
                    if toplot["both"]:
                        plot_sweeps(currents, filename, suffix='all', title=title)
                    if toplot["stats"]:
                        plot_stats(stats_df, filename, suffix='all_stats', title=title)
                    if toplot["fn"]:
                        plot_sweeps(fn_df, filename, suffix='fn', semilogy=False, take_abs=False, title=title)
                    if toplot["fn_stats"]:
                        plot_stats(fn_stats, filename, y_label=r'$\ln{I/V^2}$', suffix='fn_stats', semilogy=False,
                                   take_abs=False, title=title)
                    if toplot["resistance_slice"]:
                        plot_slice(resistance_slice, filename, "LBR", title=title)

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
