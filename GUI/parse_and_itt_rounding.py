# Note that this code will break if the number of peaks are bellow paddy params

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


def get_res(file_path):
    '''
    Works with general imports from Xcalibur

    needs to get interpolation integrated?
    '''
    df = pd.read_csv(file_path)
    df.columns=['Time','Intensity']
    df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
    df['Intensity'] = pd.to_numeric(df['Intensity'], errors='coerce')
    df = df.dropna(subset=['Time'])
    df = df.dropna(subset=['Intensity'])
    dfx = df['Time'].to_numpy()
    #assign dfy (intensity) as Y values, and convert to numpy
    dfy = df['Intensity'].to_numpy()
    peak_list = -np.sort(-dfy)
    peaks, _ = find_peaks(dfy, height = peak_list[300]*0.2, distance = 3.5)
    #peak width at 1/2 peak height
    results_half = peak_widths(dfy, peaks, rel_height=0.5)
    #results_half[0] - is the actual widths
    # make dictionary from data
    data = {'peaks':peaks, 'result_width': results_half[0] }
    #Turn dictionary into pandas dataframe
    df = pd.DataFrame.from_dict(data)
    peak_res = []
    for ind in df.index[:-1]:
        resolution = (df['peaks'][ind+1]-df['peaks'][ind])/(df['result_width'][ind]+df['result_width'][ind+1])
        resolution = '{:.2f}'.format(resolution)
        peak_res.append(float(resolution))
        #peak_res_panda = pd.DataFrame(peak_res)
    #print(peak_res)
    return peak_res

peak_res_list = get_res(crom_path)#list of peak resolutions, will probably need to have some fail safe for the case of some error (missing peak)


runner = paddy.utils.paddy_recover(path_var+'itteration_{0}'.format(str(paddy_itt)))

#gets resolution
fitness_list = []
rep_var = 0
temp_list = []
for i in peak_res_list[4:]:#skip the first 5 peak pairs
    if rep_var < 4:
        temp_list.append(i)
        rep_var += 1
    elif rep_var == 4:#this actually skips the pair between two peaks of differing parameters (convieniantly)
        fitness_list.append(sum(temp_list)/4)#there are 4 peak pairs to average
        temp_list = [] 
        rep_var = 0

if len(temp_list) == 4:#because im tired, im pretty sure the last index would just append and not hit the elif
    fitness_list.append(sum(temp_list)/4)


solution_index = [] 
c = 0 
for i in fitness_list:# get index values for solutions 
    if ( ( ( i - 1.1 ) ** 2 ) ** .5 ) < 0.05:
        solution_index.append(c)
    c += 1



#This block writes over seed fitness
#curently just maximizes fitness
#sense it itterates off of paddy seed indexi, it can be used with chromatograms containing more peaks than needed (demo use)

if len(solution_index) == 0:

    if int(paddy_itt) == 0:
        for i in range(10):
            runner.seed_fitness[i] = - ( ( ( fitness_list[i] - 1.1 ) ** 2 ) ** .5 )
        runner.file_name = path_var + "itteration_1"
        runner.recover_run()#should terminate at itteration number '1'
        runner_index = runner.generation_data['1'][0]
        runner_index_clone = runner.generation_data['1'][0]

    if int(paddy_itt) != 0:
        c = 0
        for i in np.arange(runner.generation_data[str(paddy_itt)][0],runner.generation_data[str(paddy_itt)][1]):
            runner.seed_fitness[i] = - ( ( ( fitness_list[c] - 1.1 ) ** 2 ) ** .5 )
            c += 1
        runner.file_name = path_var + "itteration_{}".format(str(int(paddy_itt)+1))
        runner.extend_paddy(1)
        runner_index = runner.generation_data[str(int(paddy_itt)+1)][0]
        runner_index_clone = runner.generation_data[str(int(paddy_itt)+1)][0]

    '''
    '''

    replicates = 5 #may want to make replicates a user input in the future (the number of times paddy values are replicated)

    param_list = []
    for i in runner.seed_params[runner_index:]:
        pump , pulse = round(i[0][0],1) , round(i[1][0],6)
        param_list.append([pump,pulse])
        runner.seed_params[runner_index_clone][0][0] = round(runner.seed_params[runner_index_clone][0][0],1)#rounds at the resolution of 0.1s
        runner.seed_params[runner_index_clone][1][0] = round(runner.seed_params[runner_index_clone][1][0],6)#rounds at the resolution of 1µs
        #updating the seed parameters is done prior to propogation 
        runner_index_clone += 1 

    reagent = opts.valve
    new_recipe_file = open(path_var+'itteration_{}'.format(str(int(paddy_itt)+1)),'w+')
    new_recipe_file.write('{}  Segments\n'.format(len(param_list)*5+5))
    new_recipe_file.write('Reagent,Delay,HV Width, LV Width\n')
    # we will probably want to write the tuning pulses
    new_recipe_file.write('{0},1.00000000E+0,1.00000000E-4,0.00000000E+0,\n'.format(reagent))
    new_recipe_file.write('{0},2.00000000E+0,1.20000000E-4,0.00000000E+0,\n'.format(reagent))
    new_recipe_file.write('{0},3.00000000E+0,2.00000000E-4,0.00000000E+0,\n'.format(reagent))
    new_recipe_file.write('{0},4.00000000E+0,1.50000000E-4,0.00000000E+0,\n'.format(reagent))
    new_recipe_file.write('{0},5.00000000E+0,1.30000000E-4,0.00000000E+0,\n'.format(reagent))
    # because pumping/delay times increase in a summative manner, we add the previous time for each line writen
    # while itterating over the actuall parameter pair N times
    time = 7#time when the initial trial starts 
    for i in param_list:
        for j in range(replicates):
            pump = "{:.8E}".format(i[0]+time)
            pulse = "{:.8E}".format(i[1])
            #process pump into scientific notation
            # || *pulse
            new_recipe_file.write('{0},{1},{2},0.00000000E+0,\n'.format(reagent,pump,pulse))#writes a recipie line and
            #might want to itter to get resolution
            time = time + i[0]
        time = time + 5
    
    new_recipe_file.close()

else:
    #note that this section of code may be buggy sense it hasn't been tested
    solution_file = open(path_var+'solution_file','w+') #file that gets writen to working dir that contains solutions
    if int(paddy_itt) == 0:
        for i in solution_index:
            temp_sol = runner.seed_params[i]
            pump = str(temp_sol[0][0])
            pulse = str(temp_sol[1][0])
            solution_file.write("Pumping-Out Time: {0} , Pulsing Time: {1} , Resolution: {2}".format(pump,pulse,str(fitness_list[c])))
    else:
        for i in solution_index:
            temp_sol = np.arange(runner.generation_data[str(paddy_itt)][0],runner.generation_data[str(paddy_itt)][1])[i]
            pump = str(temp_sol[0][0])
            pulse = str(temp_sol[1][0])
            solution_file.write("Pumping-Out Time: {0} , Pulsing Time: {1} , Resolution: {2}".format(pump,pulse,str(fitness_list[c])))
    solution_file.close()
    complete_dumby = open(path_var+'complete_var','w+')#file just for seeing if the 
    complete_dumby.write('done')
    complete_dumby.close()