import functools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# todo: bring in form x - y1 -y2 -y3 ... -yn for origin output and so on
# todo:sortby sweep up and sweep down
# todo: add file dialogue to select several files
def openFile(filename):
    with open(filename, 'r') as myfile:
        raw = myfile.read()
    return raw


def stringToNumpy(string):
    # create list of rows
    rows = string.split("\n")
    row_array = []
    for row in rows:
        np_row = np.fromstring(row, dtype=float, sep='\t')
        if not row == "":
            row_array.append(np_row)

    np_array = np.vstack(row_array)

    return np_array


def splitToStringArray(string, procedure='sweep', keyword='#Sweep:'):
    """Function that splits larger files into smaller chunks (e.g., when several sweeps are stored in same file) and returns the labels
    :param  string:     The string to be split
    :param procedure:   The measurement procedure --> indicates filetype
    :param keyword:     The keyword to use in string.split()

    """
    if procedure == 'sweep':
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


def MakePandasDf(np_arrays, labels, index_column=1):
    """Creates a Pandas Dataframe joining all np_arrays
    :param  np_arrays:      Array containing all np_arrays to be joined
    :param  labels:         List of Strings containing the Labels - needs to have np_array.shape(0) elements
    :param  index_column:   index of the column to be used as index """

    data_frames = []
    e_data_frames = []
    o_data_frames = []
    for i in range(0, len(np_arrays)):
        np_array = np_arrays[i]
        num_labels = labels[:]
        # for s in range(0,len(labels)):
        #     num_labels[s] += str(i)


        data_frame = pd.DataFrame(data=np_array,
                                  columns=num_labels)
        data_frame.set_index(num_labels[index_column], inplace=True)

        data_frame.columns = map(lambda col: '{}_{}'.format(str(col), i), data_frame.columns)
        # todo: append to either data_frame even or odd
        data_frames.append(data_frame)
        if i % 2 == 0:
            e_data_frames.append(data_frame)
        else:
            o_data_frames.append(data_frame)

    final_df = functools.reduce(joinDataFrames, data_frames)
    e_final_df = functools.reduce(joinDataFrames, e_data_frames)
    o_final_df = functools.reduce(joinDataFrames, o_data_frames)

    # todo: generalize filter!
    currents = final_df.filter(like='Current [A]')
    e_currents = e_final_df.filter(like='Current [A]')
    o_currents = o_final_df.filter(like='Current [A]')

    e_currents.abs().plot()
    plt.title('e_Sweep')
    plt.semilogy()
    o_currents.abs().plot()
    plt.title('o_Sweep')
    plt.semilogy()

    # plt.subplot(221)
    # currents.boxplot()
    # plt.semilogy()
    # plt.title('all')
    # plt.grid(True)

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
    # print(final_df.head())


def joinDataFrames(ldf, rdf):
    return ldf.join(rdf, how='outer')


def dataToNumpy(string, procedure='sweep'):
    """obsolete function --> now split into splitToStringArray & stringToNumpy"""
    if procedure == 'sweep':

        # split at keyword
        list_strings = string.split('#Sweep:')

        # preserve first row that are the labels
        labels = list_strings[0]

        np_arrays = []
        # break down string into list that can be converted to numpy

        for i in range(1, len(list_strings) - 1):

            # remove first row of each split
            list_strings[i] = list_strings[i].split("\n", 1)[1]

            # create list of rows
            rows = list_strings[i].split("\n")
            row_array = []
            for row in rows:
                np_row = np.fromstring(row, dtype=float, sep='\t')
                if not row == "":
                    row_array.append(np_row)

            if not row_array == []:
                np_array = np.vstack(row_array)

            np_arrays.append(np_array)

        return labels, np_arrays


if __name__ == "__main__":
    filename = "M0002-1d-20u-1x4-3x_00.txt"

    labels, string_list = splitToStringArray(openFile(filename))

    np_arrays = []
    for string in string_list:
        if string == "":
            continue
        np_arrays.append(stringToNumpy(string))

    MakePandasDf(np_arrays, labels, 1)
