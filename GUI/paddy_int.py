import paddy
from optparse import OptionParser
import numpy as np
import os

print(os.getcwd())
print(type(paddy))


#takes in the path_var from the main GUI script
parser = OptionParser()
parser.add_option("-x", dest="path_var")
parser.add_option("-v", dest="valve")
opts,args = parser.parse_args()

path_var = opts.path_var

pumping_time = paddy.PaddyParameter(param_range=[1,4,.1],param_type='continuous',limits=[1,4], gaussian='default',normalization = True)
pulsing_time = paddy.PaddyParameter(param_range=[0.000070,0.000180,.000010],param_type='continuous',limits=[0.000070,0.000180], gaussian='default',normalization = True)

class space(object):
    def __init__(self):
        self.pump = pumping_time
        self.pulse = pulsing_time

def dummy_eval_function(input):
    print(input[0])
    print(input[1])
    return(-100)


pumping_time = paddy.PaddyParameter(param_range=[1,4,.1],param_type='continuous',limits=[1,4], gaussian='default',normalization = True)
pulsing_time = paddy.PaddyParameter(param_range=[0.000070,0.000180,.000010],param_type='continuous',limits=[0.000070,0.000180], gaussian='default',normalization = True)
Space = space()
runner = paddy.PFARunner(space=Space, eval_func=dummy_eval_function,
                        paddy_type='population', rand_seed_number=10,
                        yt=5,Qmax=5,r=.02,iterations=1)

runner.run_paddy()
runner.save_paddy(path_var+"iteration_0")#might want to get file handle 

####
### this code block should serve as the module that 
# pumping & pulsing

replicates = 4 #may want to make replicates a user input in the future (the number of times paddy values are replicated)

param_list = []
for i in runner.seed_params:
    pump , pulse = i[0][0] , i[1][0]
    param_list.append([pump,pulse])

reagent = opts.valve
new_recipe_file = open(path_var+'iteration_{}'.format('0'),'w+')
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
        #might want to iterate to get resolution
        time = time + i[0]
    time = time + 3

new_recipe_file.close()

complete_dumby = open(path_var+'complete_var','w+')#file just for seeing if the 
complete_dumby.write('not done')
complete_dumby.close()
