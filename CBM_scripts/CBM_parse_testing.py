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


if isExist is True:
    print('\nYou already have plotData Folder\n')
    pass
else:
    print('\nMaking a plotData folder\n')
    os.mkdir('plotData')
#function to import data and assign {sheets:data} in dictionary format. 
#saves data as a csv file
def get_data(name):
    #import data to a dictionary
    sheets_dict = pd.read_excel(name, sheet_name=None,usecols= 'A,B,E,F,I,J',skiprows=3)

    # iterate through sheets_dict
    # drop NaN values
    for name, sheet in sheets_dict.items():
        #print(name)
        #print(type(sheet))
        sheet = sheet.dropna()
        sheets_dict[name] = name
        d[name] = sheet
        ##print time and intensity
        #print(d[name][['Time.1','Intensity.1']])
        d[name].to_csv('{}.csv'.format(name))
        #for l in d.keys():
        #print(d[l])
            #print(d[l][['Time.1','Intensity.1']])
            #d[l].to_csv('{}.csv'.format(name))
    return d[name]


#function to plot data
#save plots in plotData folder
def plot_data(d):

    #iterate through {sheet:data} in the excel file
    for l in d.keys():
        #print(d[l][['Time.1','Intensity.1']])
        #assign dfx (time) as X values, and convert to numpy
        dfx = d[l]['Time.1'].to_numpy()
        #assign dfy (intensity) as Y values, and convert to numpy
        dfy = d[l]['Intensity.1'].to_numpy()
    #print(type(dfy))
    #print(dfy)
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
        print(peak_res_panda)
        # for i in range(len(peak_res)):
        #for i in range(len(peak_res_panda[:-1])):
        peak_res_panda.to_csv('resolution_{}_{}.csv'.format(l,2))
        # with open('resolution_{}.csv'.format(len(peak_res)), 'w+', newline ='') as f:
        #     writer = csv.writer(f)
        #     for i in range(len(peak_res)):
        #         writer.writerow('peak_{}'.format(i))
        #         writer.writerow(peak_res])
        #     f.close
        #print(d[l][['Time.1','Intensity.1']])
        # for i in range(len(peak_res)):
        #     print("Peak {}: {:.2f}".format(i+2, peak_res[i]))
            # file = open('resolution_{}.csv'.format(d[name]), 'w+', newline ='')
            # with file:
            #     write = csv.writer(file)
            #     writer.writerows(peak_res)
            #     file.close


            #d[name].to_csv('{}.csv'.format(name))
            #df['RESOLUTION'] = (df['peaks'][ind+1]-df['peaks'][ind])/(df['result_width'][ind]+df['result_width'][ind+1])
            # peak_res_panda = pd.DataFrame(peak_res)
            # peak_res_panda.to_csv('resolution_{}_{}.csv'.format(i,d))

            


        # #plot data
        # plt.plot(dfy)
        # plt.plot(peaks, dfy[peaks], "o")
        # plt.hlines(*results_half[1:], color="C2")
        # #plt.hlines(*results_full[1:], color="C3")
        # #plt.xticks(np.arange(0,20,5))
        # #plt.title('Peak Data', fontsize = 20)
        # plt.xlabel('Data Points')
        # plt.ylabel('Intensity')
        # #save data
        # plt.savefig('plotData/{}.png'.format(l))
        # plt.show()

# def get_res(d):
  
#     # make dictionary from data
#     data = {'peaks':peaks, 'result_width': results_half[0] }
#     #Turn dictionary into pandas dataframe
#     df = pd.DataFrame.from_dict(data)

#     peak_res = []
#     for ind in df.index[:-1]:
#         resolution = (df['peaks'][ind+1]-df['peaks'][ind])/(df['result_width'][ind]+df['result_width'][ind+1])
#         peak_res.append(resolution)

#     for i in range(len(peak_res)):
#         print("Peak {}: {:.2f}".format(i+2, peak_res[i]))
#         df['RESOLUTION'] = (df['peaks'][ind+1]-df['peaks'][ind])/(df['result_width'][ind]+df['result_width'][ind+1])
#         df.to_csv('resolution_{}_{}.csv'.format(i,name))

        
#enter your file name in getMyData variable
getMyData = get_data('PT120.xlsx')
getMyPlots = plot_data(d)
# getMyRes = get_res(d)



#print(type(sheets_dict))
#print(type(d))
#full_table.to_csv('test.csv')
