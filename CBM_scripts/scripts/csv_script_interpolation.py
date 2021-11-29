import os
import pandas as pd
import matplotlib.pyplot as plt
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

    ############################
    #interpolate
    y_pts = dfy
    x_pts = np.linspace(0,40,len(y_pts))

    #determine length of x and y values
    #The values must be the same length for interpolate to work
    print(len(x_pts))
    print(len(y_pts))
    splines = interpolate.splrep(x_pts, y_pts)

    x_vals = np.linspace(0,40,400)
    y_vals = interpolate.splev(x_vals, splines)
    plt.plot(x_pts,y_pts, 's',color='red') #plot known data
    plt.plot(x_vals,y_vals, '-x',color='black') #plot interpolated points
    #plt.xticks(np.arange(0,20,5))
    plt.xlabel('Data Points')
    plt.ylabel('Intensity')
    plt.title('Peak Data \nInterpolated', fontsize = 20)
    plt.show()

    
    print('int worked')
    norm = normalize(y_vals)
    norm = np.array(norm)

    peaks1, _ = find_peaks(norm, width = 1, height = 0.2)

    results_half2 = peak_widths(norm, peaks1, rel_height=0.5)
    results_half2[0]  # widths

    #results_full calcutates width of entire peak
    #results_full = peak_widths(dfy, peaks, rel_height=1)
    #results_full[0]  # widths

    #TODO: same comment on plotting as before
    #plot data
    plt.plot(norm)
    plt.plot(peaks1, norm[peaks1], "o")
    plt.hlines(*results_half2[1:], color="C2")
    #plt.hlines(*results_full[1:], color="C3")
    # plt.xlim(120,180)
    #plt.xticks(np.arange(0,20,5))
    #plt.title('Interpolated and Normalized', fontsize = 20)
    plt.xlabel('Data Points')
    plt.ylabel('Intensity')
    plt.show()

############################################

#resolution 


    #RESOLUTION before Interpolation

    # make dictionary from data
    data = {'peaks':peaks, 'result_width': results_half[0] }
    #Turn dictionary into pandas dataframe
    df = pd.DataFrame.from_dict(data)

    peak_res = []
    for ind in df.index[:-1]:
        resolution = (df['peaks'][ind+1]-df['peaks'][ind])/(df['result_width'][ind]+df['result_width'][ind+1])
        peak_res.append(resolution)
        
    for i in range(len(peak_res)):
        print("Peak {}: {:.2f}".format(i+2, peak_res[i]))
        
    #RESOLUTION after Interpolation
    data2 = {'peaks':peaks1, 'result_width': results_half2[0] }
    df2 = pd.DataFrame.from_dict(data2)

    peak_res2 = []
    for ind in df2.index[:-1]:
        resolution = (df2['peaks'][ind+1]-df2['peaks'][ind])/(df2['result_width'][ind]+df2['result_width'][ind+1])
        peak_res2.append(resolution)
        
    for i in range(len(peak_res2)):
        print("Peak {}: {:.2f}".format(i+2, peak_res2[i]))



    #Mean and STDev of Resolution before interporlation
    mean = statistics.mean(peak_res)
    standard_deviation = statistics.stdev(peak_res)
    print('\nMean of Peaks = {:.2f}'.format(mean))
    print('Standard Deviation of Peaks = {:.2f}'.format(standard_deviation))
    print('Max Resolution Value = {:.2f} \nMin Resolution Value = {:.2f}\n'.format(max(peak_res),min(peak_res)))
    #Mean and STDev of Resolution after interporlation
    mean2 = statistics.mean(peak_res2)
    standard_deviation2 = statistics.stdev(peak_res2)
    print('\nMean of Peaks after Interpolation = {:.2f}'.format(mean2))
    print('Standard Deviation of Peaks after Interpolation = {:.2f}'.format(standard_deviation2))
    print('Max Resolution Value After Interpolation = {:.2f} \nMin Resolution Value After Interpolation = {:.2f}\n'.format(max(peak_res2),min(peak_res2)))

    print('Smoothing lowers the mean by: {:.2f}, and reduces the STDev by: {:.2f}'.format(abs(mean2-mean),abs(standard_deviation2-standard_deviation)))
######################################

def normalize(normz):
    list1 = []
    for i in normz:
        list1.append((i - min(normz))/(max(normz)-min(normz)))
    return list1
    


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, default='mop_data.csv')
args = parser.parse_args()

test_data = get_data('mop_data.csv')