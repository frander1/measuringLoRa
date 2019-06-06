import numpy as np
import pandas as pd
from pandas import DataFrame as df
import statsmodels
import scipy
import matplotlib.pyplot as plt
import seaborn as sns

#distance was 100 meters
Distance = 100
NumberOfTestsPerDistance = 10

listline = [20,40,60,80,100,125,150]
#######################################################
BW_16_SF_10 = pd.DataFrame(np.transpose(np.array([[121.5,121,122,122,125.5,122.5,123.5,122.5,121,122.6],[123,122.5,123.5,122,123,123.5,123,122,122,123],[124,124,123,125,121,123,122,121,121,121],
                    [121,122,121,123,121.5,123,124,123,124,123],[123,123.5,123,124,123,121.8,123,123,121.5,123],[123,122,123,122,123,122,123,122,123,123],[122,123,122,122,122,122,122,122,121,123]]))
                        ,columns=['20','40','60','80','100','125','150'])
BW_16_SF_10_means = df.mean(BW_16_SF_10)
BW_16_SF_10_means.columns = ['packets','meters']
BW_16_SF_10_std = df.std(BW_16_SF_10)
BW_16_SF_10_std.columns = ['packets','std']
#######################################################
BW_16_SF_5 = pd.DataFrame(np.transpose(np.array([[96.5,96.5,100.5,96.2,93.3,95.9,96.1,100.5,95.2,94.7],[95.2,96.1,96.1,100.0,100.3,96.5,100.0,96.1,100.0,96.1],[100,100,96.1,96.1,95.2,96.1,96.1,96.1,96.1,96.5],
                    [100, 100, 96.5, 96.1, 100, 100, 100, 100, 96.1, 100],[96.1,100,100.4,101,101,102,101,103,101,103],[100,100,100,96.5,101,100,100,100,96.1,100],[100,101,101,101,96.1,95.2,95.2,96.5,100.5,101]]))
                        ,columns=['20','40','60','80','100','125','150'])
BW_16_SF_5_means = df.mean(BW_16_SF_5)
BW_16_SF_5_means.columns = ['packets','meters']
BW_16_SF_5_std = df.std(BW_16_SF_5)
BW_16_SF_5_std.columns = ['packets','std']

#######################################################
# Too many errors and bad transmissions - not good enough
BW_400_SF_10 = pd.DataFrame(np.transpose(np.array([[185,182.5,184.5,181.5,183,186,184,184,186,183],[0,0,0,186,186,0,0,184.5,184.5,0],[184,0,0,0,0,0,0,0,0,0]])),columns=['20','40','60'])
BW_400_SF_10_means = df.mean(BW_400_SF_10)
BW_400_SF_10_means.columns=['packets','meters']
BW_400_SF_10_std = df.mean(BW_400_SF_10)
BW_400_SF_10_std.columns = ['packets','std']
#######################################################
BW_400_SF_5 = pd.DataFrame(np.transpose(np.array([[103.5,102,102.5,101.7,103.4,1,102.5,101.6,105.5,101.9],[105,105,105,107,104,106,107,103,104,106.5],[104,106,105,103.5,105,102,104,103,104,104],
                    [105,104.5,104,105,106,105,105,104,105,103.5],[104,106,104.6,104,106,104,104.5,104,104,103.5],[106.3,105,106,104,106,104,107,103,105,103],[105.5,105.5,105,106,106,105,104,104,103,103]]))
                        ,columns=['20','40','60','80','100','125','150'])
BW_400_SF_5_means = df.mean(BW_400_SF_5)
BW_400_SF_5_means.columns = ['packets','meters']
BW_400_SF_5_std = df.std(BW_400_SF_5)
BW_400_SF_5_std.columns = ['packets','std']
#######################################################
BW_16_SF_6 = pd.DataFrame(np.transpose(np.array([[101,100,100,100,99.2,99,100,100,99.2,98.3],[98.0,100,101,101,102,98.9,98.9,100,98.3,100],[98.9,100,100,100,98.9,100,98.9,100,100,101],
                    [101,98.3,101,100,101,100,101,100,100,98.9]]))
                    ,columns=['60','80','100','125'])
BW_16_SF_6_means = df.mean(BW_16_SF_6)
BW_16_SF_6_means.columns = ['packets','meters']
BW_16_SF_6_std = df.std(BW_16_SF_6)
BW_16_SF_6_std.columns = ['packets','std']


#######################################################
# MEAN
plt.figure(num=1,figsize=(10,7),dpi=150)
#BW_16_SF_10_means_plot = plt.plot(BW_16_SF_10_means,'-ro',label='BW: 1.6 MHz, SF: 10')
BW_16_SF_5_means_plot = plt.plot(BW_16_SF_5_means,'-bo',label='BW: 1.6 MHz, SF: 5')
#BW_400_SF_10_means_plot = plt.plot(BW_400_SF_10_means,'-yo',label='BW: 400 Khz, SF: 10')
BW_400_SF_5_means_plot = plt.plot(BW_400_SF_5_means,'-go',label='BW: 400 Khz, SF: 5')
BW_16_SF_6_means_plot = plt.plot(BW_16_SF_6_means,'-mo',label='BW: 1.6 MHz, SF: 6')
groundtruth=plt.plot([100,100,100,100,100,100,100],'--ko',label='Ground Trouth')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel('Number of Packets')
plt.ylabel('Meters')
plt.show()

#######################################################
# STANDARD DEVIATION
plt.figure(num=2,figsize=(10,7),dpi=150)
BW_16_SF_10_std_plot = plt.plot(BW_16_SF_10_std, '-ro',label='BW: 1.6 MHz, SF: 10')
BW_16_SF_5_std_plot = plt.plot(BW_16_SF_5_std,'-bo',label='BW: 1.6 MHz, SF: 5')
#BW_400_SF_10_std_plot = plt.plot(BW_400_SF_10_std,'-yo',label='BW: 400 Khz, SF: 10')
#BW_400_SF_5_std_plot = plt.plot(BW_400_SF_5_std,'-go',label='BW: 400 Khz, SF: 5')
BW_16_SF_6_std_plot = plt.plot(BW_16_SF_6_std,'-mo',label='BW: 1.6 MHz, SF: 6')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()