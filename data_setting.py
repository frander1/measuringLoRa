import numpy as np
import pandas as pd
from pandas import DataFrame as df
import statsmodels
import scipy
import matplotlib.pyplot as plt


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
BW_400_SF_10_std = df.std(BW_400_SF_10)
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

print(BW_16_SF_5_means)
print(BW_16_SF_10_std)
print(BW_16_SF_6_std)
print(BW_400_SF_5_std)
print(BW_400_SF_10_std)
#######################################################
# MEAN
skod = plt.figure(num=1,dpi=200)
means_plot = plt.plot(listline,BW_16_SF_10_means,'-ro',label='BW: 1.6 MHz, SF: 10')
means_plot = plt.plot(listline,BW_16_SF_5_means,'-bo',label='BW: 1.6 MHz, SF: 5')
#means_plot = plt.plot([20,40,60],BW_400_SF_10_means,'-yo',label='BW: 400 Khz, SF: 10')
means_plot = plt.plot(listline,BW_400_SF_5_means,'-go',label='BW: 400 Khz, SF: 5')
means_plot = plt.plot([60,80,100,125],BW_16_SF_6_means,'-mo',label='BW: 1.6 MHz, SF: 6')
means_plot =plt.plot(listline,[100,100,100,100,100,100,100],'--ko',label='100m Ground Trouth')
plt.legend()
plt.title('Mean of measurements at different settings')
plt.xlabel('Number of requests per exchange')
plt.ylabel('Meters')
plt.show()
skod.savefig('correctedmean.pdf')

#######################################################
# STANDARD DEVIATION
skod2 = plt.figure(num=2,figsize=(10,7),dpi=150)
BW_16_SF_10_std_plot = plt.plot(listline,BW_16_SF_10_std, '-ro',label='BW: 1.6 MHz, SF: 10')
BW_16_SF_5_std_plot = plt.plot(listline,BW_16_SF_5_std,'-bo',label='BW: 1.6 MHz, SF: 5')
#BW_400_SF_10_std_plot = plt.plot([20,40,60],BW_400_SF_10_std,'-yo',label='BW: 400 Khz, SF: 10')
#BW_400_SF_5_std_plot = plt.plot(listline,BW_400_SF_5_std,'-go',label='BW: 400 Khz, SF: 5')
BW_16_SF_6_std_plot = plt.plot([60,80,100,125],BW_16_SF_6_std,'-mo',label='BW: 1.6 MHz, SF: 6')
plt.legend()
plt.title('Standard deviation of measurements at different settings')
plt.xlabel('Number of requests per exchange')
plt.ylabel('Meters')
plt.show()
skod2.savefig('correctedstd.pdf')



# Hvilke BW og SF er bedst til hvilke distancer?
# 0-500 , 500 - 1000, 1000-1500, 1500-2000, 2000-2500, 2500+
# sudden burst where it would get connection / not get connection? why?
# section om auto tuning hvor den justerer parametre selv i forhold til hvad der er bedst
