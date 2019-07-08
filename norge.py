import numpy as np
import pandas as pd
from pandas import DataFrame as df
import statsmodels
import scipy
from scipy.stats import sem, t
import matplotlib.pyplot as plt

# SETTINGS USED: BW 1.6 SF 6 80 Packets
# around 10-11 measurements per distance
#abovesnowcave = [12.8,13.8,14.7,13.8,14.7,14.0,13.8,14.7,13.8] var de alle nul? se norgemål6 mappe på pc
d7m = [5.8,6.8,6.8,6.8,6.8,6.8,6.8,6.8,6.8,5.8]
d15m = [11.8,13.8,14.7,14.7,14.7,15.7,14.7,16.6,15.7,15.7,15.2]
d25m = [23.9,25.1,24.1,25.1,25.7,24.2,25.7,27,25,25.1,25,25.7]
d50m = [39.3,42.5,38.7,39.9,38.7,40.1,38.7,40.3,38.2,39.4,42.9,38.7]
all = pd.DataFrame(np.transpose(np.array([[5.8,6.8,6.8,6.8,6.8,6.8,6.8,6.8,6.8,5.8,],
                                          [11.8,13.8,14.7,14.7,14.7,15.7,14.7,16.6,15.7,15.7,15.2],
                                          [23.9,25.1,24.1,25.1,25.7,24.2,25.7,27,25,25.1,25,25.7],
                                          [39.3,42.5,38.7,39.9,38.7,40.1,38.7,40.3,38.2,39.4,42.9,38.7]])),
                   columns=['7m','15m','40m','50m'])
print(all)
mean = df.mean(d7m)
std = df.std(d7m)

print(mean)
print(std)