#!/usr/bin/env python
from numpy import *
from pylab import * 
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import SkyCoord, get_sun 
import matplotlib.font_manager as fm

#J0023+0923	5.75	9.3833
#J0030+0451	7.5	4.85
#J0340+4130	55.0 	41.5
#J0613-2000	93.25 	-2.0
#J0645+5158	101.25 	51.9667
#J0931-1902	142.75	-19.0333
#J1012+5307	153.0	53.1167
#J1024-0719	156.0	-7.3167
#J1455-3330	223.75	-33.5
#J1600-3053	240.0	-30.8833
#J1614-2230	243.5	-22.5
#J1640+2224	250.0	22.4
#J1643-1224	250.75	-12.4
#J1713+0747	258.25	7.7833
#J1738+0333	264.5	3.55
#J1741+1351	265.25	13.85
#J1744-1134	266.0	-11.5667
#J1747-4036	266.75	-40.6
#J1832-0836	278.0	-8.6
#J1853+1303	283.25	13.05
#B1855+09	284.25	9.7167
#J1903+0327	285.75	3.45
#J1909-3744	287.25	-37.7333
#J1910+1256	287.5	12.9333
#J1918-0642	289.5	-6.7
#J1923+2515	290.75	25.25
#B1937+21	294.75	21.5667
#J1944+0907	296.0	9.1167
#J1949+3106	297.25	31.1
#B1953+29	298.75	29.1333
#J2010-1323	302.5	-13.3833
#J2017+0603	304.25	6.05
#J2043+1711	310.75	17.1833
#J2145-0750	326.5	-7.8333
#J2214+3000	333.5	30.0
#J2302+4442	345.5	44.7
#J2317+1439	349.25	14.65

hfont = {'fontname':'Helvetica_Reg'}
prop = fm.FontProperties(fname='Helvetica_Reg.ttf')

MJD, DM, error= [], [], []
RA, DEC, psr = [], [], []

psr = loadtxt('positions.txt', unpack=True, usecols=(0,), dtype='str')
RA, DEC = loadtxt('positions.txt', unpack=True, usecols=(1,2,))

try:
    dmxfile = sys.argv[1]
    MJD,DM,error=np.loadtxt(dmxfile, unpack=True, usecols=(0,1,2,))
except IndexError:
	print "file not read" % sys.argv[0]
	sys.exit(1)

pname = raw_input('Which pulsar?     ')
ra1, dec1 = [], []

for j in range(0, len(psr)):
  if pname == psr[j]:
    ra1.append(RA[j])
    dec1.append(DEC[j])
	
DM=np.array(DM)
DM=1000*DM
error=np.array(error)
error=1000*error	
MJD=array(MJD)
myEpoch  = Time(MJD, format='mjd', scale='utc')
#fracyr   = myEpoch.decimalyear % 1
Sunpos   = get_sun(myEpoch)
psr      = SkyCoord(ra=ra1*u.deg, dec=dec1*u.deg)
sunangle = psr.separation(Sunpos)

print min(sunangle)

fig = figure(figsize=(12,10))
subplot(211)
plot(MJD, DM, 'k+')
#plot(MJD, DM, 'ro')
errorbar(MJD, DM, yerr=error,linestyle='None',color='black')
xticks(fontsize=18,fontproperties=prop)
yticks(fontsize=18,fontproperties=prop)
#ylabel(fontsize=20)
xlabel("MJD",fontsize=20,fontproperties=prop)
fig.text(0.15, 0.865, ''+str(pname),fontsize=20,fontproperties=prop)
#title("J1614-2230")fig.text(0.2, 0.85, 'B1855+09')
    
#subplot(312)
#plot(fracyr, DM, 'k+')
#plot(fracyr, DM, 'ro')
#errorbar(fracyr, DM, yerr=error,linestyle='None',color='black')
#ylabel(r'$\Delta$ DM [10$^{-3}$ pc cm$^{-3}$]',fontsize=20)
#xlabel("Fractional year",fontsize=20,fontproperties=prop)
#xticks(fontsize=18,fontproperties=prop)
#yticks(fontsize=18,fontproperties=prop)

#subplot(414)
#plot(fracyr, sunangle, 'b-')
#plot(fracyr, sunangle, 'ro')
#xlabel("Fractional year")
#ylabel("Sun separation ($^\circ$)")
    
subplot(212)
#plot(sunangle, DM, 'b-')
plot(sunangle, DM, 'k+')
errorbar(sunangle, DM, yerr=error,linestyle='None',color='black')
#ylabel(fontsize=20)
xlabel("Sun separation [deg]",fontsize=20,fontproperties=prop)
xticks(fontsize=18,fontproperties=prop)
yticks(fontsize=18,fontproperties=prop)

show()
