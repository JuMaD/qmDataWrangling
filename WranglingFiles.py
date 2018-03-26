import functools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# todo: sortby sweep up and sweep down
# todo: add file dialogue to select directory
# todo: make a higher level object / df / dict that acummulates all junctions in one directory
# todo: implement tool that shows plots from several "higher level plots" in on figure
# todo: implement calc_tools: calculate current density, low bias resistance, ...
# todo: implement origin export function
#
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

    data_frames = []
    e_data_frames = []
    o_data_frames = []

    # turn numpy arrays into data frames and join them at the indicated index
    for i in range(0, len(np_arrays)):

        np_array = np_arrays[i]
        num_labels = labels[:]

        data_frame = pd.DataFrame(data=np_array,
                                  columns=num_labels)
        data_frame.set_index(num_labels[index_column], inplace=True)

        data_frame.columns = map(lambda col: '{}_{}'.format(str(col), i), data_frame.columns)

        data_frames.append(data_frame)
        if i % 2 == 0:
            e_data_frames.append(data_frame)
        else:
            o_data_frames.append(data_frame)

    final_df = functools.reduce(joinDfs, data_frames)
    e_final_df = functools.reduce(joinDfs, e_data_frames)
    o_final_df = functools.reduce(joinDfs, o_data_frames)

    # todo: generalize and encapsulate filter!
    currents = final_df.filter(like='Current [A]')
    e_currents = e_final_df.filter(like='Current [A]')
    o_currents = o_final_df.filter(like='Current [A]')

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


def visualizeSweeps(dfs, datapath, stats=['mean', 'min', 'max'], plotall=True):
    """
    Sets up the plots for sweeps: all sweeps & stats min max
    :param dfs:         List of data frames to plot
    :param stats:       List of stats to plot (mean, max, min, 25%,50%,75%)
    :param plotall:     Bool that decides whether all curves (not only stats) are plot in a seperate plot
    """
    #todo: break into make stats array and plot/save
    #make plt.close non-blocking
    plt.ion()

    #generate filename to save to
    dir = os.path.join(os.path.dirname(datapath),'plots')
    filepath = os.path.join(dir,os.path.splitext(os.path.basename(datapath))[0])
    filename_stats = filepath + "_stats.png"
    filename_all = filepath + "_all.png"



    #calculate stats for HRS & LRS
    oe_stats = []
    for df in dfs:
        if df.name in ['odd', 'even']:
            stat = df.apply(pd.DataFrame.describe, axis=1)
            oe_stats.append(stat)

    stats_df = oe_stats[0].join(oe_stats[1], how='outer', lsuffix='_odd', rsuffix='_even')

    #add suffix to labels so all joint columns can be shown
    stats_arr = []
    for string in stats:
        odd = string + '_odd'
        even = string + '_even'
        stats_arr.append(odd)
        stats_arr.append(even)
    #plot all joint columns
    stats_df[stats_arr].abs().plot()
    plt.semilogy()
    plt.savefig(filename_stats)


    if plotall:
        dfs[0].abs().plot()
        plt.semilogy()
        plt.savefig(filename_all)

    plt.show()
    plt.close('all')
    return stats_arr,



    """ for d in range(0, len(dfs)):
       df = dfs[d]
       if df.name in plots:
           df.abs().plot()
           plt.semilogy()
           plt.title(df.name)
           if stats[d]:
               df.apply(pd.DataFrame.describe, axis=1)[['mean', 'max', 'min']].abs().plot()
               plt.semilogy()
               plt.title(df.name)"""

    return True


"""
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


    dirname = "testdata"

    # do for all .txt files in directory
    for file in os.listdir(dirname):
        if file.endswith(".txt"):
            if not file.endswith("Resistance.txt"):
                if not file.endswith("SMU-Puls.txt"):
                    # open file and get data
                    filename = os.path.join(dirname, file)

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
                    # get dfs with current
                    currents = []
                    for d in range(0, len(dfs)):
                        current = filterDf(dfs[d], 'Current [A]')
                        current.name = dfs_names[d]
                        currents.append(current)

                    saveDfToFile(currents[0],filename,'_all')
                    saveDfToFile(currents[0].abs(), filename, '_all_abs')

                    visualizeSweeps(currents, filename)



