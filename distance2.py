import numpy as np
import pandas as pd
from pandas import DataFrame as df
from scipy.stats import sem, t
import matplotlib.pyplot as plt

# 25 measurements per distance BW 1.6 MHz and SF 6 as found yesterday to be best 100 packets mange pakker bliver tabt / noter hvor mange måske 10/100 ca.
# 10 m
n = 25
confidence = 0.95
groundtruth = [10,25,50,75,100,150,200,250,300,400,510,642,841,1122]
#######################################################
BW_16_SF_6_Measurements = pd.DataFrame(np.transpose(np.array([
                         [9.9,11.8,10.8,9.9,11.8,9.9,10.8,10.8,10.8,10.8,9.9,10.8,10.8,10.8,9.9,9.1,9.9,9.9,10.8,10.4,11.8,10.8,10.8,10.8,10.8],
                         [26.5,26.5,28,25.7,26.5,26.5,26.1,25,26.9,25.7,25.7,25,25,25.7,26.7,25.4,26.1,25.7,26.5,26.5,25,25.7,24.2,24.2,24.2],
                         [42.5,43,42.5,42.5,42.5,42.5,42.5,40.9,41.7,42.5,41.7,42.5,42.5,42.5,40.9,42.9,42.9,42.5,42.9,41.7,42.5,44,44,42.5,41.7],
                         [69.7,69.7,69.7,71,69.7,69.7,68.3,68.3,67.7,68.3,68.3,70.4,68.3,69.7,69.7,69.7,69.7,69.6,71,71,69.7,69.7,69.7,69.7,69.7],
                         [99.2, 98.9, 98.3, 98.9, 100.5, 97.5, 100, 98.9, 98.9, 98.3, 100, 98.9, 97.5, 97.5, 100, 97.5, 98.3, 100, 98.9, 100, 97.5, 100, 98.7, 101, 103],
                         [151,149,150.5,152,151.5,151.5,151.5,151.5,151.5,151,151,151,150,150,151,150,148.5,152,149.8,152,149,151,150.5,150,151.1],
                         [199,201,199,199,199,203,201.5,199.6,203.7,193.5,193.5,201,202.2,204,202,192.9,197.4,200.3,200.5,200.3,202.5,202,203,203,199],
                         [248,249,249,247.7,247.7,249,248.7,249,248.5,249,250,248,250,251,250,248.5,249,250,246,246,248,247,248,247,243.5],
                         [296,296,291.4,302.5,302.5,302.5,301,299,296.2,290.8,303,301.9,294,295,303,299,299.5,301,301,304,298,305.1,300,300.5,300],
                         [400,405.4,403.8,398.7,402.6,399.7,408.8,402.1,403.2,415.7,400,401.6,400,400,399,399.5,399,398.4,397,395.5,399,397,400.5,400,395.9]])),
                         columns=['10m','25m','50m','75m','100m','150m','200m','250m','300m','400m'])
BW_16_SF_6_mean = df.mean(BW_16_SF_6_Measurements)
BW_16_SF_6_mean.columns=['Meters','Measurements']
BW_16_SF_6_mean_list = list(BW_16_SF_6_mean)
BW_16_SF_6_std = df.std(BW_16_SF_6_Measurements)
BW_16_SF_6_std.columns=['Meters','std']
print(BW_16_SF_6_mean)
#print(BW_16_SF_6_std )
print((BW_16_SF_6_std ))
print(df.mean(BW_16_SF_6_std))

#######################################################
All_Measurements = pd.DataFrame(np.transpose(np.array([
                 [9.9,11.8,10.8,9.9,11.8,9.9,10.8,10.8,10.8,10.8,9.9,10.8,10.8,10.8,9.9,9.1,9.9,9.9,10.8,10.4,11.8,10.8,10.8,10.8,10.8],
                 [26.5,26.5,28,25.7,26.5,26.5,26.1,25,26.9,25.7,25.7,25,25,25.7,26.7,25.4,26.1,25.7,26.5,26.5,25,25.7,24.2,24.2,24.2],
                 [42.5,43,42.5,42.5,42.5,42.5,42.5,40.9,41.7,42.5,41.7,42.5,42.5,42.5,40.9,42.9,42.9,42.5,42.9,41.7,42.5,44,44,42.5,41.7],
                 [69.7,69.7,69.7,71,69.7,69.7,68.3,68.3,67.7,68.3,68.3,70.4,68.3,69.7,69.7,69.7,69.7,69.6,71,71,69.7,69.7,69.7,69.7,69.7],
                 [99.2, 98.9, 98.3, 98.9, 100.5, 97.5, 100, 98.9, 98.9, 98.3, 100, 98.9, 97.5, 97.5, 100, 97.5, 98.3, 100, 98.9, 100, 97.5, 100, 98.7, 101, 103],
                 [151,149,150.5,152,151.5,151.5,151.5,151.5,151.5,151,151,151,150,150,151,150,148.5,152,149.8,152,149,151,150.5,150,151.1],
                 [199,201,199,199,199,203,201.5,199.6,203.7,193.5,193.5,201,202.2,204,202,192.9,197.4,200.3,200.5,200.3,202.5,202,203,203,199],
                 [248,249,249,247.7,247.7,249,248.7,249,248.5,249,250,248,250,251,250,248.5,249,250,246,246,248,247,248,247,243.5],
                 [296,296,291.4,302.5,302.5,302.5,301,299,296.2,290.8,303,301.9,294,295,303,299,299.5,301,301,304,298,305.1,300,300.5,300],
                 [400,405.4,403.8,398.7,402.6,399.7,408.8,402.1,403.2,415.7,400,401.6,400,400,399,399.5,399,398.4,397,395.5,399,397,400.5,400,395.9],
                 [525.5, 524.9, 5515, 524, 522, 525.5, 523, 523, 525, 522, 526, 536, 524, 523, 41342, 523.5, 525, 526, 524, 535, 523, 522, 525, 526, 524],# remove 41342 from 500 m
                 [664,664,665,664,664,708,665,665,664,664,665,665,665,664,664,664,665,665,664,664,665,663,664,664,664],
                 [892.5,1129,892,891,892,889,890,889,888,890,892,889,891,891,891,892,890.5,892,0,893,893,893,890,889,889],
                 [1181,1181,1182,1181,1183,1182,1180,1182,1180,1180,1180,1181,1181,1182,1183,1182,1183,1183,1184,1184,1184,1183,1183,1184,1183]])),
                 columns=['10m','25m','50m','75m','100m','150m','200m','250m','300m','400m','510m','642m','841m','1122m'])

All_median = df.median(All_Measurements)
All_median.columns=['Meters','Measurements']
All_mean = df.mean(All_Measurements)
All_mean_list = list(All_mean)
All_mean.columns=['Meters','Measurements']
All_std = df.std(All_Measurements)
All_std.columns=['Meters','std']

### confidence intervals per distance

std_error_list = []
h_list = []
start_list = []
end_list = []

for i in range(0,14):
    std_error_list.append(sem(All_Measurements[(str(groundtruth[i])+'m')]))
    h_list.append(std_error_list[i]*t.ppf((1+confidence)/2,n-1))
    start_list.append(All_mean_list[i] - h_list[i])
    end_list.append(All_mean_list[i] + h_list[i])
    print(round(start_list[i],2), 'm  to  ', round(end_list[i],2), 'm')


#######################################################
# PLOTS
skod = plt.figure(num=1,dpi=200)
BW_16_SF_6_mean_plot = plt.plot(groundtruth[0:10],BW_16_SF_6_mean,'-ro',label = 'BW 1.6 Mhz SF 6')
ground_truth_plot = plt.plot(groundtruth[0:10],groundtruth[0:10], '--k',label = 'ground truth')
plt.title('Mean of measurements at different distances')
plt.xlabel('Meters')
plt.ylabel('Meters')
plt.legend()
plt.grid()
skod.savefig('400measurement.pdf')
plt.show()

skod2 = plt.figure(num=4,dpi=200)
plt.plot(groundtruth[0:10],BW_16_SF_6_std ,'-ro',label = 'BW 1.6 Mhz SF 6')
plt.title('Standard devation of measurements at different distances')
plt.xlabel('Meters')
plt.ylabel('Standard deviation')
plt.legend()
plt.grid()
skod2.savefig('400std.pdf')
plt.show()

plt.figure(num=2,figsize=(10,7),dpi=150)
All_mean_plot = plt.plot(groundtruth,All_mean,'-ro',label = 'BW  Mhz SF ')
ground_truth_plot2 = plt.plot(groundtruth,groundtruth, '-bo',label = 'ground truth')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

plt.figure(num=3,figsize=(10,7),dpi=150)
All_median_plot = plt.plot(groundtruth,All_median,'-ro',label = 'BW Mhz SF ')
ground_truth_plot3 = plt.plot(groundtruth,groundtruth, '-bo',label = 'ground truth')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()