import os
from os import name, set_inheritable
from numpy.lib.function_base import percentile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import csv
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
from sklearn import preprocessing
from scipy import interpolate

import statistics
import re

# create dictionary d = {sheets: data} as {keys:values}
d = {}
isExist = os.path.exists('plotData')
isExist2 = os.path.exists('resData')
isExist3 = os.path.exists('data')

#make folders to store data - Plots | resolution | data
if isExist is True:
    print('\nYou already have plotData Folder\n')
    pass
else:
    print('\nMaking a plotData folder\n')
    os.mkdir('plotData')

if isExist2 is True:
    print('\nYou already have resData Folder\n')
    pass
else:
    print('\nMaking a resData folder\n')
    os.mkdir('resData')
if isExist3 is True:
    print('\nYou already have data Folder\n')
    pass
else:
    print('\nMaking a data folder\n')
    os.mkdir('data')

#function to import data and assign {sheets:data} in dictionary format. 
#saves data as a csv file
def get_data(name):
    #import data to a dictionary
    sheets_dict = pd.read_excel(name, sheet_name=None,usecols= 'A,B,E,F,I,J',skiprows=3)

    # iterate through sheets_dict
    # drop NaN values
    for name, sheet in sheets_dict.items():
        sheet = sheet.dropna()
        sheets_dict[name] = name
        d[name] = sheet
        d[name].to_csv('data/{}.csv'.format(name))
        for l in d.keys():
            d[l].to_csv('data/{}.csv'.format(name))
    return d[name]


#function to plot data
#save plots in plotData folder
def plot_data(d):

    #iterate through {sheet:data} in the excel file
    for l in d.keys():
        #assign dfx (time) as X values, and convert to numpy
        dfx = d[l]['Time.1'].to_numpy()
        #assign dfy (intensity) as Y values, and convert to numpy
        dfy = d[l]['Intensity.1'].to_numpy()
        #peak finder via Y values. Threshold = height (can alter based on data)
        peaks, _ = find_peaks(dfy, height = 5000)
        #peak width at 1/2 peak height
        results_half = peak_widths(dfy, peaks, rel_height=0.5)
        results_half[0]
      
        # make dictionary from data
        data = {'peaks':peaks, 'result_width': results_half[0] }
        #Turn dictionary into pandas dataframe
        df = pd.DataFrame.from_dict(data)
        peak_res = []
        for ind in df.index[:-1]:
            resolution = (df['peaks'][ind+1]-df['peaks'][ind])/(df['result_width'][ind]+df['result_width'][ind+1])
            peak_res.append(resolution)
        peak_res_panda = pd.DataFrame(peak_res)
        peak_res_panda.to_csv('resData/resolution_{:.2f}_{}.csv'.format(l,2))
            
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
        plt.savefig('plotData/{}.png'.format(l))
        plt.show()

        
#enter your file name in getMyData variable
getMyData = get_data('PT120.xlsx')
getMyPlots = plot_data(d)
