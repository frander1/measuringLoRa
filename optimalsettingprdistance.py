import numpy as np
import pandas as pd
from pandas import DataFrame as df
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.patches as mpatches
#import statsmodels
import scipy
from scipy.stats import sem, t
import matplotlib.pyplot as plt
#import seaborn as sns

#groundtruth was 25, 100, 1133 ,1690, 2607
truth = [25, 100, 1133 ,1690, 2607]

# IF BW 1.6 WORKS - USE THAT since it provides higher datarate and higher precision
# 10 per sf per datarate 100 packets
#
### 25m   SF 5  6  7  8  9  10
BW_16_25m = pd.DataFrame(np.transpose(np.array([[20.5,21.4,21.4,20.1,21.4,21.4,22.2,22.2,21.4,20.5],[21.7,22.6,23.4,23.4,23.4,23.4,24.2,24.2,24.2,24.2],
                                                [24.7,25.5,24.7,25.5,25.1,25.5,25.5,24.7,25.5,24.7],[26.8,26.4,26.4,26.4,26.4,26.4,27.2,26.4,26.4,26.4],
                                                [30.5,29.9,30.2,29.9,29.9,29.9,29.9,29.9,29.9,29.9],[38.5,39.4,39.4,37.8,37.8,38.5,39.2,39.4,39.4,39.4]])),columns=['SF5','SF6','SF7','SF8','SF9','SF10'])

BW_8_25m  = pd.DataFrame(np.transpose(np.array([[25.9,26.7,26.7,26.7,20.0,25.9,26.7,27.4,26.7,25.9],[30.8,31.5,30.8,32.0,30.8,31.5,31.2,32.1,30.8,31.5],
                                                [26.7,26.7,26.7,27.0,26.7,27.3,26.1,26.7,27.2,27.0],[30.4,30.1,30.0,29.6,29.6,30.1,30.1,30.1,29.6,29.9],
                                                [39.1,38.3,38.3,39.1,39.1,38.7,38.3,38.3,39.1,40.4],[63.1,64.2,64.2,64.2,63.6,63.1,63.1,64.2,63.1,62.0]])),columns=['SF5','SF6','SF7','SF8','SF9','SF10'])

BW_4_25m  = pd.DataFrame(np.transpose(np.array([[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],
                                                [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],
                                                [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],[113.0,111.5,112.0,111.0,112.0,107.0,107.0,109.0,108.0,108.0]])),columns=['SF5','SF6','SF7','SF8','SF9','SF10'])
# at sf 10 started seeing packet refused  to send / faileed

### 100m
BW_16_100m = pd.DataFrame(np.transpose(np.array([[96.1,95.6,96.5,96.5,100,96.1,95.2,96.1,96.5,100],[99.2,100,98.9,98.3,98.4,98.9,98.9,98.9,98.9,98.9],
                                                 [99.0,100.5,100,101,101,100,101,100,101,100],[104,103,104,104,103.5,105,105,104,104,104],
                                                 [110,110,110,110,110,110,110,110,110,109.5],[122,122,122.5,123,123,123,122,121,121,122]])),columns=['SF5','SF6','SF7','SF8','SF9','SF10'])

BW_8_100m  = pd.DataFrame(np.transpose(np.array([[98.7,98.3,100,98.7,98.7,99,99.3,99.3,99,99],[111,111,112,111,111,111,111,111,110,112],
                                                 [105,106,107,109,107,109,104,105,105,106],[111,111.5,112,112,111,112,110,111,112.5,111],
                                                 [121,120,120.5,121,122,122,121,121,122,122],[144,144,143.5,143,144.1,142,144,144,144,144.5]])),columns=['SF5','SF6','SF7','SF8','SF9','SF10'])

BW_4_100m  = pd.DataFrame(np.transpose(np.array([[105,106,103.3,105,109,109,107,106,107,106.8],[109,110,111,109,107,110,111,112,112,111],
                                                 [118,116,115,115,114,115,115,117.5,117,117.5],[124.5,125,125,126,125,124,124,125.5,124.5,124],
                                                 [147,146,147.5,146,147,144,145,145,146,144],[180,179,182,182,182,184,183.3,185,183.5,182]])),columns=['SF5','SF6','SF7','SF8','SF9','SF10'])
# maybe 1 / 80 at SF10, (few 0 measured bc packet loss 100%) could not at around 300+ tries get more than 5 measurements

### 500m
BW_16_500m = pd.DataFrame(np.transpose(np.array([[514.3,506.8,518.5,7115,516.3,517.4,512.7,55107,505.7,509.4],[512.2,512,521,510,511,506.7,511,515.9,511,513.2],
                                                 [511.5,515,512,512,512,512,511.9,514,511,512],[518,517,517.5,518,517,516,516,517,517,517],
                                                 [521,521,521,521,522,521,523,522,521,523.5],[533,533,533,531,533,533,532,531,531,532]])),columns=['SF5','SF6','SF7','SF8','SF9','SF10'])
# sf 5 60-70% tries not coming through and only 5% packet return. SF 6 bit better with 40 % tries drop and 20 % packet return
# sf 7 much better at 90 % tries succes and 30-90% packet return % sf 8 same as sf7
# sf 9 as sf 7 and 8 but 80-100% packet return. sf 10 tries begin failing

BW_8_500m  = pd.DataFrame(np.transpose(np.array([[506,509,504,510,510,515.8,510,511,514,512],[522,522,521,521,523.5,523,524,522.5,522,521],
                                                 [518,517,517,518,518,516,517,518,516,518],[521,526,522,521,522,521,520,523,520,521],
                                                 [531,531,532,531,531,531,532,531,532,532.5],[552,552,552,552,554,551,551,553,552,553]])),columns=['SF5','SF6','SF7','SF8','SF9','SF10'])
# tries getting lost a few times at 800 sf 5 but 80% packet return. SF 6 better % tries getting through like 90%
# sf 100 % tries get throug and 90  % packet return.
# sf 9 and 10 almost 100 % packet return

BW_4_500m  = pd.DataFrame(np.transpose(np.array([[514,516.5,516.5,518,514,523,511.5,512.5,516.5,517.5],[519,518,519,521,519,520,519.8,520,518,519],
                                                 [524,524.8,524,525,526,523,524.5,526,525,524.5],[533,534.5,533,534.5,535,535,533,534,535,534],
                                                 [552,551,552,553,553,553,552,553.3,551,554],[589,590,594,None,None,None,None,None,None,None]])),columns=['SF5','SF6','SF7','SF8','SF9','SF10'])
# 3 / 100~150 get through at sf 10 so not good enough

### 1000m
BW_16_1000m = pd.DataFrame(np.transpose(np.array([[None,None,None,None,None,None,None,None,None,None],[None,None,None,None,None,None,None,None,None,None],
                                                  [1137,1143,1138,1138,1137,1136,1132,1141,1138,1136],[1140,1142,1141,1143,1141,1142,1141,1143,1141,1144],
                                                  [1146,1145,1144,1145,1145,1146,1145,1145,1144,1145],[1156,1156,1155,1156,1156,1157,1157,1155,1156,1157]])),columns=['SF5','SF6','SF7','SF8','SF9','SF10'])
# sf 5 not coming through out of 60, same with sf 6
# sf 7 10% tries succes, 5-20% packet return sf8 100 triy succes 5-90% packet return

BW_8_1000m  = pd.DataFrame(np.transpose(np.array([[1130,1127,1887,None,None,None,None,None,None,None],[1145,1140,1149,1144,1152,1153,1146,1138,1148,1149],
                                                  [1142,1143,1141,1143,1141,1142,1144,1142,1144,1141],[1149,1148,1150,1149,1150,1149,1150,1148,1149,1147],
                                                  [1156,1158,1145,1151,1152,1155,1155,1156,1150,1159],[1178,1178,1177,1176,1175,1177,1177,1175,1176,1176]])),columns=['SF5','SF6','SF7','SF8','SF9','SF10'])
# sf 5 3 / 100 try and 1-5% packet return, sf 6 packet return 1-40%
# sf8 maybe 70 % try succes

BW_4_1000m  = pd.DataFrame(np.transpose(np.array([[1118,1133,1124,1079,1120,1161,1440,1110,1127,1137],[1134,1133,1138,1130,1136,1140,1139,1136,1133,1132],
                                                  [1138,1137,1143,1142,1138,1140,1137,1141,1139,1145],[1149,1148,1147,1147,1148,1148,1148,1148,1148,1149],
                                                  [1168,1167,1169,1167,1167,1168,1169,1171,1169,1171],[1195,1194,1200,1202,1199,1198,1199,1198,1190,1192]])),columns=['SF5','SF6','SF7','SF8','SF9','SF10'])
# sf 5 maybe 60% try succcsss sf 6 80% try success


# DTU total højde  = 4.35+(4.41-0.87)
### 1500m
BW_16_1500m = pd.DataFrame(np.transpose(np.array([[None,None,None,None,None,None,None,None,None,None],[None,None,None,None,None,None,None,None,None,None],
                                                  [None,None,None,None,None,None,None,None,None,None],[None,None,None,None,None,None,None,None,None,None],
                                                  [1684,1685,1683,1683,1683,1712,1688,1693,1687,1689],[1670,1671,1670,1671,1668,1671,1671,1673,1670,1670]])),
                           columns=['SF5','SF6','SF7','SF8','SF9','SF10'])
# sf9 10% try syccess sf10 little better but not much maybe 15-20%

BW_8_1500m  = pd.DataFrame(np.transpose(np.array([[None,None,None,None,None,None,None,None,None,None],[None,None,None,None,None,None,None,None,None,None],
                                                  [1690,1688,1688,1689,1695,1697,1693,1692,1691,1690],[1685,1683,1689,1687,1691,1688,1693,1696,1687,1688],
                                                  [1668,1679,1664,1669,1667,1661,1642,1651,1675,1656],[1646,1648,1645,1648,1647,1645,1645,1648,1645,1646]])),columns=['SF5','SF6','SF7','SF8','SF9','SF10'])
# sf 7 5-10 % try succes, sf 8 was 90 %
# 65535 was taken out of sf 9

BW_4_1500m  = pd.DataFrame(np.transpose(np.array([[1682,1,1679,None,None,None,None,None,None,None],[1676,1684,1682,1682,1686,1689,1677,1720,1667,1645],
                                                  [1685,1681,1680,1681,1679,1684,1672,1685,1677,1682],[1665,1669,1669,1664,1677,1672,1680,1678,1670,1679],
                                                  [1645,1645,1643,1643,1644,1637,1643,1652,1649,1639],[1580,1583,1586,1596,1591,1587,1588,1584,1584,1585]])),columns=['SF5','SF6','SF7','SF8','SF9','SF10'])
#SF 5 3 (2 proper) results in 150 tries. SF 6 271 tries total
# sf 8 was completely stable

### 2500m
BW_16_2500m = pd.DataFrame(np.transpose(np.array([[None,None,None,None,None,None,None,None,None,None],[None,None,None,None,None,None,None,None,None,None],
                                                  [None,None,None,None,None,None,None,None,None,None],[None,None,None,None,None,None,None,None,None,None],
                                                  [None,None,None,None,None,None,None,None,None,None],[2584,2581,None,None,None,None,None,None,None,None]])),columns=['SF5','SF6','SF7','SF8','SF9','SF10'])
# sf 10 2 / 150 got through

BW_8_2500m  = pd.DataFrame(np.transpose(np.array([[None,None,None,None,None,None,None,None,None,None],[None,None,None,None,None,None,None,None,None,None],
                                                  [None,None,None,None,None,None,None,None,None,None],[2603,None,None,None,None,None,None,None,None,None],
                                                  [2545,2540,None,None,None,None,None,None,None,None],[2558,2546,2547,2547,2563,2569,2576,2559,2551,2545]])),columns=['SF5','SF6','SF7','SF8','SF9','SF10'])
# sf 8 was 1 / 150
# sF 9 was 2 / 150, sf 10 10/300+

BW_4_2500m  = pd.DataFrame(np.transpose(np.array([[None,None,None,None,None,None,None,None,None,None],[None,None,None,None,None,None,None,None,None,None],
                                                  [None,None,None,None,None,None,None,None,None,None],[2576,1,None,None,None,None,None,None,None,None],
                                                  [2574,2633,None,None,None,None,None,None,None,None],[2517,2511,2498,2517,2518,2523,2511,2521,2502,2507]])),columns=['SF5','SF6','SF7','SF8','SF9','SF10'])
#SF 8 was 1(2) out of 150
#SF 9 was 2 / 150 SF 10 was at most 10% try succes 10/ 100


BW_16_25mean = df.mean(BW_16_25m)
BW_16_100mean = df.mean(BW_16_100m)
BW_16_500mean = df.mean(BW_16_500m)
BW_16_1000mean = df.mean(BW_16_1000m)
BW_16_1500mean = df.mean(BW_16_1500m)
BW_16_2500mean = df.mean(BW_16_2500m)

BW_8_25mean = df.mean(BW_8_25m)
BW_8_100mean = df.mean(BW_8_100m)
BW_8_500mean = df.mean(BW_8_500m)
BW_8_1000mean = df.mean(BW_8_1000m)
BW_8_1500mean = df.mean(BW_8_1500m)
BW_8_2500mean = df.mean(BW_8_2500m)

BW_4_25mean = df.mean(BW_4_25m)
BW_4_100mean = df.mean(BW_4_100m)
BW_4_500mean = df.mean(BW_4_500m)
BW_4_1000mean = df.mean(BW_4_1000m)
BW_4_1500mean = df.mean(BW_4_1500m)
BW_4_2500mean = df.mean(BW_4_2500m)

# ground truth and setup for plots
SF,BW = np.meshgrid([5,6,7,8,9,10],[400,800,1600])
SF1,BW1 = [5,6,7,8,9,10],[400,800,1600]
groundtruth = np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]])

norm2 = plt.Normalize(groundtruth.min(),groundtruth.max())
colors2 = cm.viridis(norm2(groundtruth))
rcount2, ccount2, _  = colors2.shape

# 25 meter
m25mean_err = np.array([(BW_4_25mean-25),(BW_8_25mean-25),(BW_16_25mean-25)]) #subtracting 25 from mean in order to move
                                                                              #data around 0, such that it will be mean                                                                              #error
norm = plt.Normalize(m25mean_err.min(), m25mean_err.max())
colors = cm.viridis(norm(m25mean_err))
rcount, ccount, _ = colors.shape

# 100 meter

# 500 meter

# 1000 meter

# 1500 meter

# 2500 meter

#fig 1

fig1 = plt.figure(num=2,figsize=(10,7),dpi=200)
ax1 = Axes3D(fig1)

surf0 = ax1.plot_surface(BW,SF,groundtruth,rcount=rcount2, ccount=ccount2, facecolors=colors2, shade=False)
surf1 = ax1.plot_surface(BW,SF,m25mean_err,rcount=rcount, ccount=ccount, facecolors=colors, shade=False)
for a,b,c in zip(m25mean_err,BW,SF): # LEGIT GENIUS PLOT
    for i,j,k in zip(a,b,c):
        ax1.plot([j,j],[k,k],[0,i],'--r')
ax1.scatter(BW,SF,m25mean_err,s=25,marker='s',color='b',depthshade=False)

surf0.set_facecolor((0,0,0,0))
surf0.set_edgecolor('green')
surf0.set_linestyle('-')
surf1.set_facecolor((0,0,0,0))
surf1.set_edgecolor((0,0,0,1))
surf1.set_linestyle('-')
surf1.set_linewidth(1)
ax1.set_xlabel('Bandwidth (KHz)')
ax1.set_ylabel('Spreading Factor')
ax1.set_zlabel('Error (m) ')
ax1.set_xticks([400,800,1600])
ax1.view_init(azim=-135)
plt.title('25m Mean Error')
green_patch = mpatches.Patch(color='green', label='No error / Ground Truth')
blue_patch = mpatches.Patch(color='blue', label='Mean Error ')
plt.legend(handles=[green_patch, blue_patch])
plt.show()
