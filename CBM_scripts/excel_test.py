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

# def square(a):
#     return a*a

f = lambda a: a*a
g = lambda a,b=11: a+b
alist = [1,2,3,4,5,6,7,8,9]

newList = list(map(lambda x: x+5, alist))

print(newList)



result = f(5)
result2 = g(5,22)

print(result)
print(result2)

