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
from sklearn import preprocessing
from scipy import interpolate

import statistics
import re

# TODO: Good job on defining the dictionary with appropriate commenting
# create dictionary d = {sheets: data} as {keys:values}
d = {}
# TODO: Do not do this, while it can help in organization it makes the code more complex
isExist = os.path.exists('plotData')
isExist2 = os.path.exists('resData')
isExist3 = os.path.exists('data')

#make folders to store data - Plots | resolution | data
# TODO: Format as follows
# TODO: redo pathing
if os.path.exists('plotData') is True:
    print('\nYou already have plotData Folder\n')
    # TODO: remove this, you should not need the pass statement here, pass is used for a different reason
    pass
else:
    print('\nMaking a plotData folder\n')
    # TODO: redo pathing
    os.mkdir('plotData')

# TODO: Format these as the statement above
if isExist2 is True:
    print('\nYou already have resData Folder\n')
    # TODO: remove pass
    pass
else:
    print('\nMaking a resData folder\n')
    # TODO: redo pathing
    os.mkdir('resData')

if isExist3 is True:
    print('\nYou already have data Folder\n')
    # TODO: remove pass
    pass
else:
    print('\nMaking a data folder\n')
    #TODO: redo pathing
    os.mkdir('data')

#function to import data and assign {sheets:data} in dictionary format. 
#saves data as a csv file

# TODO: for documenting functions, you should define parameters and return value either above or right inside (generally prefered)
# TODO: use a block comment
'''
TODO:
function to import data and assign {sheets:data} in dictionary format

name:(describe name parameter)...
returns: saves data as a csv file
'''
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
    '''
    TODO: 
    function to plot data

    d: ...
    returns: ... 
    '''

    # TODO: does anything here need to be put in as command line arguments
    #iterate through {sheet:data} in the excel file
    for l in d.keys():
        #assign dfx (time) as X values, and convert to numpy
        dfx = d[l]['Time.1'].to_numpy()
        #assign dfy (intensity) as Y values, and convert to numpy
        dfy = d[l]['Intensity.1'].to_numpy()
        print(max(dfy))
        peak_list = -np.sort(-dfy)
        #highs[4]
        print(peak_list[3])
        #peak finder via Y values. Threshold = height (can alter based on data)
        #Can use peak_list variable to select which peak you want to base the height off of
        #peaks, _ = find_peaks(dfy, height = max(dfy)*0.4)
        peaks, _ = find_peaks(dfy, height = peak_list[4]*0.2, distance = 3.5)
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
            resolution = '{:.2f}'.format(resolution)
            peak_res.append(resolution)
        peak_res_panda = pd.DataFrame(peak_res)
        peak_res_panda.to_csv('resData/resolution_{}_{}.csv'.format(l,2))
            
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

'''
TODO:
You should use command line arugments here instead of having users define in the code

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, default='PT120.xlsx')
args = parser.parse_args()

now run the file as 
python CBM_parse.py --filename example_file_name.xlsx
'''
getMyData = get_data('PT120.xlsx')
getMyPlots = plot_data(d)