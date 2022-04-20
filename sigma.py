#! /usr/bin/env python

from matplotlib import pyplot as PLT
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import sys
import os
from pylab import *
import scipy as sy
from scipy.optimize import curve_fit
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import SkyCoord, get_sun 
import matplotlib.font_manager as fm


prop = fm.FontProperties(fname='/home/mljones1/dmproject/Helvetica_Reg.ttf')

#psr=np.loadtxt('psrnames.txt', unpack=True, usecols=(0,), dtype='str')

try:
    psr = sys.argv[1]
    mjd,dm0,err = np.loadtxt(psr+'.dmx',unpack=True, usecols=(0,1,2))
except IndexError:
	print "file not read" % sys.argv[0]
	sys.exit(1)

dm0=1000*np.array(dm0)
err=1000*np.array(err)

start=0
end=70000
x,y,error=[],[],[]
for i in range(0, len(mjd)):
	if mjd[i] > start and mjd[i] < end:
		x.append(mjd[i])
		y.append(dm0[i])
		error.append(err[i])

x=np.array(x)
y=np.array(y)
error=np.array(error)

#Fit linear function
def func(x, a, b):
  return a*x+b

p0 = sy.array([1,1])
coeffs, matcov = curve_fit(func, x, y, p0)
perr = np.sqrt(np.diag(matcov))
yaj1 = func(x, coeffs[0], coeffs[1])
print "Linear Fit"
print(coeffs)
print(perr)
xl=np.arange(min(x), max(x), 1)
gl= coeffs[0]*xl+coeffs[1]


dm_new = y - yaj1


#sigma=[]
#for i in range(0,36):
#	mjd,dm,error=np.loadtxt(psr[i]+'.dmx', unpack=True, usecols=(0,1,2,))
#	dm=dm*1000
#	error=error*1000
#	sigma.append(np.std(dm))
sigma  = np.std(dm_new)
print sigma	
print max(dm_new)
print max(dm_new)/sigma

ra1 = 250.75	
dec1 = -12.4
MJD=array(x)
myEpoch  = Time(MJD, format='mjd', scale='utc')
#fracyr   = myEpoch.decimalyear % 1
Sunpos   = get_sun(myEpoch)
psr      = SkyCoord(ra=ra1*u.deg, dec=dec1*u.deg)
sunangle = psr.separation(Sunpos)

fig = figure(figsize=(12,10))
subplot(211)
plot(mjd, dm0, 'k+')
#plot(MJD, DM, 'ro')
errorbar(mjd, dm0, yerr=err,linestyle='None',color='black')
xticks(fontsize=18,fontproperties=prop)
yticks(fontsize=18,fontproperties=prop)
#ylabel(fontsize=20)
xlabel("MJD",fontsize=20,fontproperties=prop)
#fig.text(0.15, 0.865, ''+str(pname),fontsize=20,fontproperties=prop)
#title("J1614-2230")fig.text(0.2, 0.85, 'B1855+09')
    
subplot(212) 
#plot(sunangle, DM, 'b-')
plot(sunangle, dm_new, 'k+')
errorbar(sunangle, dm_new, yerr=error,linestyle='None',color='black')
#ylabel(fontsize=20)
xlabel("Sun separation [deg]",fontsize=20,fontproperties=prop)
xticks(fontsize=18,fontproperties=prop)
yticks(fontsize=18,fontproperties=prop)

show()
