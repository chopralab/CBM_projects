"""
Backend dev script, uses ploting to display thresholds to alert user if the spectra
was processed correctly
"""
# Note that this code will break if the number of peaks are bellow paddy params
# Note that the indexing for iterations past 0 may break

import paddy
from optparse import OptionParser
import os
from os import name, set_inheritable
from numpy.lib.function_base import percentile
import pandas as pd
import csv

from pandas.core.base import DataError
from pandas.core.frame import DataFrame

import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.signal import peak_widths
from scipy.signal.ltisys import dfreqresp
from sklearn import preprocessing
from scipy import interpolate

import statistics
import re

from optparse import OptionParser
import matplotlib.pyplot as plt

parser = OptionParser()
parser.add_option("-x", dest="path_var")
parser.add_option("-y", dest="crom_path")
parser.add_option("-z", dest="paddy_itt")
parser.add_option("-v", dest="valve")
opts,args = parser.parse_args()

path_var = opts.path_var
crom_path = opts.crom_path
paddy_itt = opts.paddy_itt


class space(object):
    def __init__(self):
        self.pump = pumping_time
        self.pulse = pulsing_time

def dummy_eval_function(input):
    print(input[0])
    print(input[1])
    return(-100)

def pumping_times(paddy_itt):
    """Returns pumping times from a first iteration given pickle."""
    replicates = 4 #may want to make replicates a user input in the future (the number of times paddy values are replicated)
    runner = paddy.utils.paddy_recover('C:/Users/Chopr/Desktop/CBM/CBM_projects/GUI/'+'paddytest_{0}'.format(str(paddy_itt)))
    pump_list = []
    print(paddy_itt)
    if int(paddy_itt) == 0:
        print('itt is 0')
        for i in range(10):
            pump_list.append(runner.seed_params[i][0][0])
    else:
        for i in np.arange(runner.generation_data[str(paddy_itt)][0],runner.generation_data[str(paddy_itt)][1]):
            print(np.arange(runner.generation_data[str(paddy_itt)][0],runner.generation_data[str(paddy_itt)][1]))
            pump_list.append(runner.seed_params[i][0][0])
    # we will probably want to write the tuning pulses
    times = [1.0,2.0,3.0,4.0,5.0]
    time = 7#time when the initial trial startsw
    print(len(pump_list))
    for i in pump_list:
        for j in range(replicates):
            times.append(round(i+time,1))
            time = time + i
        time = time + 3.0
    return(times,pump_list)

'''
def pumping_times():
    """Returns pumping times from a first iteration given pickle."""
    replicates = 10 #may want to make replicates a user input in the future (the number of times paddy values are replicated)
    pump_list = [4.0,1.4,3.2,3.0,2.7,2.0,3.2,2.4,1.5,4.0]
    # we will probably want to write the tuning pulses
    times = [1.0,2.0,3.0,4.0,5.0]
    time = 7#time when the initial trial startsw 
    for i in pump_list:
        for j in range(replicates):
            times.append(round(i+time,1))
            time = time + i
        time = time + 3.0
    return(times,pump_list)
'''

runner = paddy.utils.paddy_recover('C:/Users/Chopr/Desktop/CBM/CBM_projects/GUI/'+'paddytest_{0}'.format(str(paddy_itt)))
dft = pd.read_csv(crom_path,skiprows=4)
dft.columns=['Time','Intensity']
dft['Time'] = pd.to_numeric(dft['Time'], errors='coerce')
dft['Intensity'] = pd.to_numeric(dft['Intensity'], errors='coerce')
dft = dft.dropna(subset=['Time'])
dft = dft.dropna(subset=['Intensity'])
#assign dfx (time) as X values, and convert to numpy
dfx = dft['Time'].to_numpy()
#assign dfy (intensity) as Y values, and convert to numpy
dfy = dft['Intensity'].to_numpy()
print(max(dfy))
peak_list = -np.sort(-dfy)
peaks, _ = find_peaks(dfy, height = peak_list[300]*0.2, distance = 3.5)
#peak width at 1/2 peak height
results_half = peak_widths(dfy, peaks, rel_height=0.5)

lower = []
higher = []
for i , j in zip(results_half[2], results_half[3]):
    remainders = [i % 1 , j % 1]
    x0_times = [dfx[int(i-remainders[0])] , dfx[int(j-remainders[1])]]
    x1_times = [dfx[int(i-remainders[0]+1)] , dfx[int(j-remainders[1]+1)]]
    lower.append(x0_times[0]+remainders[0]*(x1_times[0]-x0_times[0]))
    higher.append(x0_times[1]+remainders[1]*(x1_times[1]-x0_times[1]))

width_list = np.array([results_half[1],lower,higher])
#plot data
#looks fine if tuning peaks are first
plt.plot(dfx,dfy)
plt.plot(dfx[peaks], dfy[peaks], "o")
times,pump_list = pumping_times(paddy_itt)
print(len(times))
for i in times:
    plt.plot((dfx[peaks[0]]+(i-1)/60,dfx[peaks[0]]+(i-1)/60),(0,max(dfy)),c='k')
plt.hlines(*width_list, color="C2")
plt.xlabel('Time (Minutes)')
plt.ylabel('Intensity')
plt.show()

data = {'peaks':peaks, 'result_width': results_half[0] }
#Turn dictionary into pandas dataframe
dft2 = pd.DataFrame.from_dict(data)
peak_res = []
for ind in dft2.index[:-1]:
    resolution = (dft2['peaks'][ind+1]-dft2['peaks'][ind])/(
                    dft2['result_width'][ind]+dft2['result_width'][ind+1])
    resolution = '{:.2f}'.format(resolution)
    peak_res.append(resolution)
    peak_res_panda = pd.DataFrame(peak_res)
print(peak_res_panda)


time_widths = []
for i, j in zip(lower,higher):
    time_widths.append(j-i) 

data = {'peaks':peaks, 'result_width': time_widths }
#Turn dictionary into pandas dataframe
dft2 = pd.DataFrame.from_dict(data)
peak_res = []
for ind in dft2.index[:-1]:
    resolution = (dfx[dft2['peaks'][ind+1]]-dfx[dft2['peaks'][ind]])/(
                    dft2['result_width'][ind]+dft2['result_width'][ind+1])
    #resolution = '{:.2f}'.format(resolution)
    peak_res.append(resolution)
    peak_res_panda = pd.DataFrame(peak_res)

orient = None
orienting = True
while orienting:#getting the tuning peaks
    for i in range(len(peaks)):
    #this is not fool proof, but should work for most cases thrown
        if (dfx[peaks[i+4]] - dfx[peaks[i]])*60 > 3.5:
            if (dfx[peaks[i+4]] - dfx[peaks[i]])*60 < 4.5:
                print((dfx[peaks[i+5]] - dfx[peaks[i+4]])*60)
                print(2+pump_list[0] + .5)
                if (dfx[peaks[i+5]] - dfx[peaks[i+4]])*60 < (2+pump_list[0] + .5):#bug
                    print('True')
                    print(i)
                    print((dfx[peaks[i+5]] - dfx[peaks[i+4]])*60)
                    if (dfx[peaks[i+5]] - dfx[peaks[i+4]])*60 > (2+pump_list[0] - .5):
                        print('Done')
                        orienting = False
                        orient = i + 4 #this is the last tuning peak
                        break
    print('hi')
print(orient)

thresh_vals = [dfx[peaks][orient]*60+(2+pump_list[0])/2]
thresh_vals.append(thresh_vals[0]+(pump_list[0]*9.5+2.5)+pump_list[1]*.5)
c = 2
for i in pump_list[1:]:
    if c != len(pump_list):
        temp = i * 9.5 + thresh_vals[-1] + 3 + pump_list[c] * .5
    else:
        temp = i * 9.5 + thresh_vals[-1] + 3 + 2
    thresh_vals.append(temp)
    c += 1
    
print(thresh_vals)

fitness_list = []
for i in range(len(thresh_vals)-1):
    fitness_list.append([])
print(len(peak_res))
print(len(fitness_list))
t = 0
for i in peaks[orient+1:]:
    if t < len(thresh_vals)-1:
        if dfx[i]*60 > thresh_vals[1+t]:
            t += 1
    fitness_list[t].append(i)
c = 0
for z in fitness_list:
    if len(z) != 4:
        fitness_list[c] = -9999999
    else:
        results_half = peak_widths(dfy, z, rel_height=0.5)
        lower = []
        higher = []
        for i , j in zip(results_half[2], results_half[3]):
            remainders = [i % 1 , j % 1]
            x0_times = [dfx[int(i-remainders[0])] , dfx[int(j-remainders[1])]]
            x1_times = [dfx[int(i-remainders[0]+1)] , dfx[int(j-remainders[1]+1)]]
            lower.append(x0_times[0]+remainders[0]*(x1_times[0]-x0_times[0]))
            higher.append(x0_times[1]+remainders[1]*(x1_times[1]-x0_times[1]))

        width_list = np.array([results_half[1],lower,higher])
        #plot data
        data = {'peaks':z, 'result_width': results_half[0] }
        #Turn dictionary into pandas dataframe
        dft2 = pd.DataFrame.from_dict(data)
        peak_res = []
        for ind in dft2.index[:-1]:
            resolution = (dft2['peaks'][ind+1]-dft2['peaks'][ind])/(
                        dft2['result_width'][ind]+dft2['result_width'][ind+1])
            #resolution = '{:.2f}'.format(resolution)
            print(resolution)
            peak_res.append(resolution)
        fitness_list[c] = (sum(peak_res)/3)
    c += 1

    print(fitness_list)

solution_index = [] 
c = 0 
for i in fitness_list:# get index values for solutions 
    if ( ( ( i - 1.5 ) ** 2 ) ** .5 ) <= 0.05:
        solution_index.append(c)
    c += 1

#This block writes over seed fitness
#curently just maximizes fitness
#sense it iterates off of paddy seed indexi, it can be used with chromatograms containing more peaks than needed (demo use)
print(fitness_list)
print(solution_index)
if len(solution_index) == 0:

    if int(paddy_itt) == 0:
        for i in range(10):
            runner.seed_fitness[i] = - ( ( ( fitness_list[i] - 1.5 ) ** 2 ) ** .5 )
        runner.file_name = path_var + "iteration_1"
        runner.recover_run()#should terminate at iteration number '1'
        runner_index = runner.generation_data['1'][0]
        runner_index_clone = runner.generation_data['1'][0]

    if int(paddy_itt) != 0:
        c = 0
        for i in np.arange(runner.generation_data[str(paddy_itt)][0],runner.generation_data[str(paddy_itt)][1]):
            runner.seed_fitness[i] = - ( ( ( fitness_list[c] - 1.1 ) ** 2 ) ** .5 )
            c += 1
        runner.file_name = path_var + "iteration_{}".format(str(int(paddy_itt)+1))
        runner.extend_paddy(1)
        runner_index = runner.generation_data[str(int(paddy_itt)+1)][0]
        runner_index_clone = runner.generation_data[str(int(paddy_itt)+1)][0]

    '''
    '''

    replicates = 4 #may want to make replicates a user input in the future (the number of times paddy values are replicated)

    param_list = []
    for i in runner.seed_params[runner_index:]:
        pump , pulse = round(i[0][0],1) , round(i[1][0],6)
        param_list.append([pump,pulse])
        runner.seed_params[runner_index_clone][0][0] = round(runner.seed_params[runner_index_clone][0][0],1)#rounds at the resolution of 0.1s
        runner.seed_params[runner_index_clone][1][0] = round(runner.seed_params[runner_index_clone][1][0],6)#rounds at the resolution of 1µs
        #updating the seed parameters is done prior to propogation 
        runner_index_clone += 1 

    reagent = opts.valve
    new_recipe_file = open(path_var+'iteration_{}'.format(str(int(paddy_itt)+1)),'w+')
    new_recipe_file.write('{}  Segments\n'.format(len(param_list)*replicates+5))
    new_recipe_file.write('Reagent,Delay,HV Width, LV Width\n')
    # we will probably want to write the tuning pulses
    new_recipe_file.write('{0},1.00000000E+0,1.00000000E-4,0.00000000E+0,\n'.format(reagent))
    new_recipe_file.write('{0},2.00000000E+0,1.20000000E-4,0.00000000E+0,\n'.format(reagent))
    new_recipe_file.write('{0},3.00000000E+0,2.00000000E-4,0.00000000E+0,\n'.format(reagent))
    new_recipe_file.write('{0},4.00000000E+0,1.50000000E-4,0.00000000E+0,\n'.format(reagent))
    new_recipe_file.write('{0},5.00000000E+0,1.30000000E-4,0.00000000E+0,\n'.format(reagent))
    # because pumping/delay times increase in a summative manner, we add the previous time for each line writen
    # while iterating over the actuall parameter pair N times
    time = 7#time when the initial trial starts 
    for i in param_list:
        for j in range(replicates):
            pump = "{:.8E}".format(i[0]+time)
            pulse = "{:.8E}".format(i[1])
            #process pump into scientific notation
            # || *pulse
            new_recipe_file.write('{0},{1},{2},0.00000000E+0,\n'.format(reagent,pump,pulse))#writes a recipie line and
            #might want to iteration to get resolution
            time = time + i[0]
        time = time + 3
    
    new_recipe_file.close()

else:
    #note that this section of code may be buggy sense it hasn't been tested
    solution_file = open(path_var+'solution_file.txt','w+') #file that gets writen to working dir that contains solutions
    if int(paddy_itt) == 0:
        for i in solution_index:
            temp_sol = runner.seed_params[i]
            pump = str(round(temp_sol[0][0],1))
            pulse = str(round(temp_sol[1][0]*1000000))
            print(i)
            solution_file.write("Pumping-Out Time: {0} seconds, Pulsing Time: {1} micro-seconds, Resolution: {2}\n".format(pump,pulse,str(fitness_list[i])))
    else:
        for i in solution_index:
            temp_sol = np.arange(runner.generation_data[str(paddy_itt)][0],runner.generation_data[str(paddy_itt)][1])[i]
            pump = str(round(temp_sol[0][0],1))
            pulse = str(round(temp_sol[1][0]*1000000))
            print(i)
            solution_file.write("Pumping-Out Time: {0} seconds, Pulsing Time: {1} micro-seconds, Resolution: {2}\n".format(pump,pulse,str(fitness_list[i])))
    solution_file.close()
    complete_dumby = open(path_var+'complete_var','w+')#file just for seeing if the optimizagtion is over
    complete_dumby.write('done')
    complete_dumby.close()