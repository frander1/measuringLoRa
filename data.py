import numpy as np
import pandas as pd
import statsmodels
import scipy
import matplotlib.pyplot as plt
import seaborn as sns

#distance was 100 meters
Distance = 100
NumberOfTestsPerDistance = 10

listline = [20,40,60,80,100,125,150]
list20 = [20,20,20,20,20,20,20,20,20,20]
list40 = [40,40,40,40,40,40,40,40,40,40]
list60 = [60,60,60,60,60,60,60,60,60,60]
list80 = [80,80,80,80,80,80,80,80,80,80]
list100 = [100,100,100,100,100,100,100,100,100,100]
list125 = [125,125,125,125,125,125,125,125,125,125]
list150 = [150,150,150,150,150,150,150,150,150,150]
####################################################################################
# BW 1.6 MHz SF 10

BW1600SF10PacketNumber20 = [121.5,121,122,122,125.5,122.5,123.5,122.5,121,122.6]
BW16SF10Pn20mean = np.mean(BW1600SF10PacketNumber20)

BW1600SF10PacketNumber40 = [123,122.5,123.5,122,123,123.5,123,122,122,123]
BW16SF10Pn40mean = np.mean(BW1600SF10PacketNumber40)

BW1600SF10PacketNumber60 = [124,124,123,125,121,123,122,121,121,121]
BW16SF10Pn60mean = np.mean(BW1600SF10PacketNumber60)

BW1600SF10PacketNumber80 = [121,122,121,123,121.5,123,124,123,124,123]
BW16SF10Pn80mean = np.mean(BW1600SF10PacketNumber80)

BW1600SF10PacketNumber100 = [123,123.5,123,124,123,121.8,123,123,121.5,123]
BW16SF10Pn100mean = np.mean(BW1600SF10PacketNumber100)

BW1600SF10PacketNumber125 = [123,122,123,122,123,122,123,122,123,123]
BW16SF10Pn125mean = np.mean(BW1600SF10PacketNumber125)

BW1600SF10PacketNumber150 = [122,123,122,122,122,122,122,122,121,123]
BW16SF10Pn150mean = np.mean(BW1600SF10PacketNumber150)

BW16SF10meanlist = [BW16SF10Pn20mean,BW16SF10Pn40mean,BW16SF10Pn60mean,BW16SF10Pn80mean,BW16SF10Pn100mean,BW16SF10Pn125mean,BW16SF10Pn150mean]

####################################################################################
# BW 1.6 MHz SF 5

BW1600SF5PacketNumber20 = [96.5,96.5,100.5,96.2,93.3,95.9,96.1,100.5,95.2,94.7]
BW16SF5Pn20mean = np.mean(BW1600SF5PacketNumber20)

BW1600SF5PacketNumber40 = [95.2,96.1,96.1,100.0,100.3,96.5,100.0,96.1,100.0,96.1]
BW16SF5Pn40mean = np.mean(BW1600SF5PacketNumber40)

BW1600SF5PacketNumber60 = [100,100,96.1,96.1,95.2,96.1,96.1,96.1,96.1,96.5]
BW16SF5Pn60mean = np.mean(BW1600SF5PacketNumber60)

BW1600SF5PacketNumber80 = [100,100,96.5,96.1,100,100,100,100,96.1,100]
BW16SF5Pn80mean = np.mean(BW1600SF5PacketNumber80)

BW1600SF5PacketNumber100 = [96.1,100,100.4,101,101,102,101,103,101,103]
BW16SF5Pn100mean = np.mean(BW1600SF5PacketNumber100)

BW1600SF5PacketNumber125 = [100,100,100,96.5,101,100,100,100,96.1,100]
BW16SF5Pn125mean = np.mean(BW1600SF5PacketNumber125)

BW1600SF5PacketNumber150 = [100,101,101,101,96.1,95.2,95.2,96.5,100.5,101]
BW16SF5Pn150mean = np.mean(BW1600SF5PacketNumber150)

BW16SF5meanlist = [BW16SF5Pn20mean,BW16SF5Pn40mean,BW16SF5Pn60mean,BW16SF5Pn80mean,BW16SF5Pn100mean,BW16SF5Pn125mean,BW16SF5Pn150mean]


####################################################################################
# BW 400 MHz SF 10

BW400SF10PacketNumber20 = [185,182.5,184.5,181.5,183,186,184,184,186,183]
BW4SF10Pn20mean = np.mean(BW400SF10PacketNumber20)

BW400SF10PacketNumber40 = [0,0,0,186,186,0,0,184.5,184.5,0]
BW4SF10Pn40mean = np.mean(BW400SF10PacketNumber40)

BW400SF10PacketNumber60 = [184,0,0,0,0,0,0,0,0,0]
BW4SF10Pn60mean = np.mean(BW400SF10PacketNumber60)


# so much packet loss / errors that 0 was registereed. This test did not continue

#BW400SF10PacketNumber80 = [x,x,x,x,x,x,x,x,x,x]

#BW400SF10PacketNumber100 = [x,x,x,x,x,x,x,x,x,x]

#BW400SF10PacketNumber125 = [x,x,x,x,x,x,x,x,x,x]

#BW400SF10PacketNumber150 = [x,x,x,x,x,x,x,x,x,x]

####################################################################################
# BW 400 MHz SF 5

BW400SF5PacketNumber20 = [103.5,102,102.5,101.7,103.4,1,102.5,101.6,105.5,101.9]
BW4SF5Pn20mean = np.mean(BW400SF5PacketNumber20)

BW400SF5PacketNumber40 = [105,105,105,107,104,106,107,103,104,106.5]
BW4SF5Pn40mean = np.mean(BW400SF5PacketNumber40)

BW400SF5PacketNumber60 = [104,106,105,103.5,105,102,104,103,104,104]
BW4SF5Pn60mean = np.mean(BW400SF5PacketNumber60)

BW400SF5PacketNumber80 = [105,104.5,104,105,106,105,105,104,105,103.5]
BW4SF5Pn80mean = np.mean(BW400SF5PacketNumber80)

BW400SF5PacketNumber100 = [104,106,104.6,104,106,104,104.5,104,104,103.5]
BW4SF5Pn100mean = np.mean(BW400SF5PacketNumber100)

BW400SF5PacketNumber125 = [106.3,105,106,104,106,104,107,103,105,103]
BW4SF5Pn125mean = np.mean(BW400SF5PacketNumber125)

BW400SF5PacketNumber150 = [105.5,105.5,105,106,106,105,104,104,103,103]
BW4SF5Pn150mean = np.mean(BW400SF5PacketNumber150)

BW4SF5meanlist = [BW4SF5Pn20mean,BW4SF5Pn40mean,BW4SF5Pn60mean,BW4SF5Pn80mean,BW4SF5Pn100mean,BW4SF5Pn125mean,BW4SF5Pn150mean]

####################################################################################
# BW 1.6 MHz SF 6
BW1600SF6PacketNumber60 = [101,100,100,100,99.2,99,100,100,99.2,98.3]
BW16SF6Pn60mean = np.mean(BW1600SF6PacketNumber60)

BW1600SF6PacketNumber80 = [98.0,100,101,101,102,98.9,98.9,100,98.3,100]
BW16SF6Pn80mean = np.mean(BW1600SF6PacketNumber80)

BW1600SF6PacketNumber100 = [98.9,100,100,100,98.9,100,98.9,100,100,101]
BW16SF6Pn100mean = np.mean(BW1600SF6PacketNumber100)

BW1600SF6PacketNumber125 = [101,98.3,101,100,101,100,101,100,100,98.9]
BW16SF6Pn125mean = np.mean(BW1600SF6PacketNumber125)

BW16SF6meanlist = [BW16SF6Pn60mean,BW16SF6Pn80mean,BW16SF6Pn100mean,BW16SF6Pn125mean]

####################################################################################

# PLOTTING ALL DATA AS POINTS

plt.figure()
plt.plot(list20,BW1600SF10PacketNumber20,'ro')
plt.plot(list40,BW1600SF10PacketNumber40,'ro')
plt.plot(list60,BW1600SF10PacketNumber60,'ro')
plt.plot(list80,BW1600SF10PacketNumber80,'ro')
plt.plot(list100,BW1600SF10PacketNumber100,'ro')
plt.plot(list125,BW1600SF10PacketNumber125,'ro')
plt.plot(list150,BW1600SF10PacketNumber150,'ro',label = 'BW 1600, SF 10')

#

plt.plot(list20,BW1600SF5PacketNumber20,'bo')
plt.plot(list40,BW1600SF5PacketNumber40,'bo')
plt.plot(list60,BW1600SF5PacketNumber60,'bo')
plt.plot(list80,BW1600SF5PacketNumber80,'bo')
plt.plot(list100,BW1600SF5PacketNumber100,'bo')
plt.plot(list125,BW1600SF5PacketNumber125,'bo')
plt.plot(list150,BW1600SF5PacketNumber150,'bo',label = 'BW 1600, SF 5')


#
plt.plot(list20,BW400SF10PacketNumber20,'co')
plt.plot(list40,BW400SF10PacketNumber40,'co')
plt.plot(list60,BW400SF10PacketNumber60,'co',label = 'BW 400, SF 10')
#

plt.plot(list20,BW400SF5PacketNumber20,'yo')
plt.plot(list40,BW400SF5PacketNumber40,'yo')
plt.plot(list60,BW400SF5PacketNumber60,'yo')
plt.plot(list80,BW400SF5PacketNumber80,'yo')
plt.plot(list100,BW400SF5PacketNumber100,'yo')
plt.plot(list125,BW400SF5PacketNumber125,'yo')
plt.plot(list150,BW400SF5PacketNumber150,'yo',label = 'BW 400, SF 5')

#

plt.plot(list60,BW1600SF6PacketNumber60,'go')
plt.plot(list80,BW1600SF6PacketNumber80,'go')
plt.plot(list100,BW1600SF6PacketNumber100,'go')
plt.plot(list125,BW1600SF6PacketNumber125,'go',label = 'BW 1600, SF 6')

plt.plot([0,200],[100,100],'black') # ground truth

# PLOTTING MEANS OF DATA

plt.legend()
#plt.axis([0,200,80,150])
plt.show()
plt.figure()
plt.plot(listline, BW16SF10meanlist,'c', label='BW 1600 SF 10')
plt.plot(listline, BW4SF5meanlist,'g', label='BW 400, SF5')
plt.plot(listline, BW16SF5meanlist,'b',label='BW 1600, SF 5')
plt.plot([60,80,100,125], BW16SF6meanlist,'y',label='BW 1600, SF 6')
plt.plot(listline,[100,100,100,100,100,100,100],'black',label='ground truth') # ground truth

plt.legend()


# suggested setting is BW 1.6 MHz and SF 6 from the above plots