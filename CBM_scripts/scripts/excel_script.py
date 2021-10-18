# TODO: Some unused imports here, please remove them
from os import name, set_inheritable
from numpy.lib.function_base import percentile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import csv
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

#import data from excel workbook
#import all sheets, skip first 3 rows, and use only 6 columns 

# TODO: consider changing this to a command line argument using argparse
df_master = pd.read_excel("PT110.xlsx", sheet_name =None, skiprows=3, usecols= 'A,B,E,F,I,J')
#df_csv = pd.read_csv('paddy3.txt', sep='\s+',skiprows=3)

#print(df_csv)
#sheet names in excel workbook
df_master_sheets = df_master.keys()
#number of sheets in excel workbook 
sheet_length = range(len(df_master_sheets))

print(df_master.keys())
# print(sheet_length)

#print out master dataframe, prints every sheet
# print(df_master)

#print out specific sheet and specific column from the master dataframe 
# print(df_master['80us_1p1s'][['Time.1','Intensity.1']])


# TODO: You should be able to use dropna() on an entire dataframe but it should not make much of a difference of how its done
for key, item in df_master.items():
    item = item.dropna()

    df_master[key] = item

#print(type(df_master))
#print(type(df_master['120us_1.0s']))

# print(df_master['80us_1p1s'][['Time.1','Intensity.1']])

# #append if cant figure out how to iterate 
# for key, item in df_master.items():
#     peaks, _ = find_peaks(item['Time.1','Intensity.1'], threshold = 4000)

# TODO: Should any of these be command line arguments (i.e. non hardcoded values)
dfx = df_master['110us_1p9s']['Time.1'].to_numpy()
dfy = df_master['110us_1p9s']['Intensity.1'].to_numpy()
#print(type(dfy))
#print(dfy)

# TODO: Same comment on CL args on height and rel_height
peaks, _ = find_peaks(dfy, height = 8000)
results_half = peak_widths(dfy, peaks, rel_height=0.5)
results_half[0]


# TODO: if you have to make similar plots multiple times, I would use a function to reduce the amount of code
#plot data
plt.plot(dfy)
plt.plot(peaks, dfy[peaks], "o")
plt.hlines(*results_half[1:], color="C2")
#plt.hlines(*results_full[1:], color="C3")
#plt.xticks(np.arange(0,20,5))
#plt.title('Peak Data', fontsize = 20)
plt.xlabel('Data Points')
plt.ylabel('Intensity')
plt.show()



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


#Normalize the interpolation
def normalize(normz):
    list1 = []
    for i in normz:
        list1.append((i - min(normz))/(max(normz)-min(normz)))
    return list1
    
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

# df_1p1 = df_master['80us_1p1s'][['Time.1','Intensity.1']]
# df_1p2 = df_master['80us_1p2s'][['Time.1','Intensity.1']]
# df_1p3 = df_master['80us_1p3s'][['Time.1','Intensity.1']]

# peaks, _ = find_peaks(df_1p1['Intensity.1'], threshold = 4000)
# results_half = peak_widths(df_1p1['Intensity.1'], peaks, rel_height=0.5)
# results_half[0]  # widths

ydata = peak_res

#plot peak resolution before and after interpolation
plt.scatter(range(len(peak_res)),peak_res)
plt.plot(peak_res)
plt.xlabel('Peaks')
plt.ylabel('Resolution')
locs, labels = plt.xticks()
#plt.xticks(np.arange(0,20,5))
plt.title('Resolution Plot of Normal Data', fontsize = 20)
#plt.ylim([1.4,2.2])
#plt.yticks([1.4,2.1])
#plt.text(14.5,1.7,'Mean = 1.99\nStd Dev = 0.20')
plt.show()


#List for for loop
for_loop_list = []
for_loop_min = min(peak_res)
for_loop_max = max(peak_res)
#for loop test, removing min and max for possible outliers
for i in peak_res:
    if i > for_loop_min and i < for_loop_max:
      for_loop_list.append(i)
      #print(for_loop_list)
      #print(type(for_loop_list))
      #print(type(peak_res))
#for_loop_list = list(for_loop_list)
#print(type(for_loop_list))
plt.scatter(range(len(for_loop_list)),for_loop_list)
plt.plot(for_loop_list)
plt.xlabel('Peaks')
plt.ylabel('Resolution')
locs, labels = plt.xticks()
plt.title('testing for loop', fontsize = 20)
plt.show()

plt.scatter(range(len(peak_res2)),peak_res2)
plt.plot(peak_res2)
plt.xlabel('Peaks')
plt.ylabel('Resolution')
locs, labels = plt.xticks()
#plt.xticks(np.arange(0,20,5))
plt.title('Resolution Plot of Interpolated Data', fontsize = 20)
#plt.text(0,2.01,'Mean = 1.81\nStd Dev = 0.13')
plt.show()

plt.scatter(range(len(peak_res)),peak_res)
plt.scatter(range(len(peak_res2)),peak_res2)
plt.plot(peak_res, label = 'Original Data')
plt.plot(peak_res2, label = 'Interpolated Data')
plt.title('Resolution Plot', fontsize = 20)
plt.xlabel('Peaks')
locs, labels = plt.xticks()
#plt.xticks(np.arange(0,20,5))
plt.ylabel('Resolution')

plt.legend(loc = 'lower right')
#plt.text(0,1.99,'O Mean = 1.99\nO Std Dev = 0.20\nI Mean = 1.81\nI Std Dev = 0.13')
# plt.text(0,1.98,'N Std Dev = 0.20')
# plt.text(0,1.95,'I Mean = 1.81')
# plt.text(0,1.92,'I Std Dev = 0.13')
plt.show()
#TODO: where are you saving any of these plots