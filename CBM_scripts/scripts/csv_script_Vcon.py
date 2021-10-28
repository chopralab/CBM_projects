# TODO: Some of these imports are not being used, they should be removed
import os
from os import name, set_inheritable
from numpy.lib.function_base import percentile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import csv
# TODO: Were these imported by VS code or did you manually import them
from pandas.core.base import DataError
from pandas.core.frame import DataFrame

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.signal import peak_widths
from scipy.signal.ltisys import dfreqresp
from sklearn import preprocessing
from scipy import interpolate

import statistics
import re


if os.path.exists('../plotData') is True:
    print('\nYou already have plotData Folder\n')
else:
    print('\nMaking a plotData folder\n')
    os.mkdir('../plotData')

if os.path.exists('../resData') is True:
    print('\nYou already have resData Folder\n')
else:
    print('\nMaking a resData folder\n')
    os.mkdir('../resData')

if os.path.exists('../data') is True:
    print('\nYou already have data Folder\n')
else:
    print('\nMaking a data folder\n')
    os.mkdir('../data')

def get_data(file):

    df = pd.read_csv('../csvFiles/mop_data.csv')
    df.columns=['Time','Intensity']
    df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
    df['Intensity'] = pd.to_numeric(df['Intensity'], errors='coerce')
    df = df.dropna(subset=['Time'])
    df = df.dropna(subset=['Intensity'])
    
    save_data = df.to_csv('../data_test/{}.csv'.format('mop_data'))

    #assign dfx (time) as X values, and convert to numpy
    dfx = df['Time'].to_numpy()
    #assign dfy (intensity) as Y values, and convert to numpy
    dfy = df['Intensity'].to_numpy()
    print(max(dfy))
    peak_list = -np.sort(-dfy)

    peaks, _ = find_peaks(dfy, height = peak_list[300]*0.2, distance = 3.5)
    #peak width at 1/2 peak height
    results_half = peak_widths(dfy, peaks, rel_height=0.5)
    results_half[0]
    #plot data
    plt.plot(dfy)
    plt.plot(peaks, dfy[peaks], "o")
    plt.hlines(*results_half[1:], color="C2")
    #plt.hlines(*results_full[1:], color="C3")
    #plt.xticks(np.arange(0,20,5))
    #plt.title('Peak Data', fontsize = 20)
    plt.xlabel('Data Points')
    plt.ylabel('Intensity')
    #save data
    plt.savefig('../plotData/{}.png'.format('mop_data'))
    plt.show()

    # make dictionary from data
    data = {'peaks':peaks, 'result_width': results_half[0] }
    #Turn dictionary into pandas dataframe
    df = pd.DataFrame.from_dict(data)
    peak_res = []
    for ind in df.index[:-1]:
        resolution = (df['peaks'][ind+1]-df['peaks'][ind])/(df['result_width'][ind]+df['result_width'][ind+1])
        resolution = '{:.2f}'.format(resolution)
        peak_res.append(resolution)
        peak_res_panda = pd.DataFrame(peak_res)
        peak_res_panda.to_csv('../resData/resolution_{}_{}.csv'.format('mop_data',2))
    #print(peak_res)
    #return peak_res

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, default='mop_data.csv')
args = parser.parse_args()

test_data = get_data('mop_data.csv')