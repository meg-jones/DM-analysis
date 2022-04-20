#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pylab import *
import scipy as sy
from scipy.optimize import curve_fit
import scipy.signal

try:
    dmxfile = sys.argv[1]
    x,y,error = np.loadtxt(dmxfile,unpack=True, usecols=(0,1,2))
except IndexError:
	print "file not read" % sys.argv[0]
	sys.exit(1)


#Fit linear
def func(x, a, b):
  return a*x+b

p0 = sy.array([1,1])
coeffs, matcov = curve_fit(func, x, y, p0)
yaj = func(x, coeffs[0], coeffs[1])

#Subtract off function
ysub=y
#for v in range(0,len(x)):
#	 ysub.append(y[v]-yaj[v])

#ysub=np.array(ysub)    
time=[]
m = np.mean(ysub)
center = ysub-m
dm = center/ysub.std()
N=len(x)
for i in range(0,N):
  for j in range(i+1,N):
    time.append(np.fabs(x[j]-x[i]))
    
#Define output frequencies
f = np.linspace(0.01,2*np.pi/300,100000)
pgram = scipy.signal.lombscargle(x, dm, f)
max_pgram = max(pgram)
max_f = f[pgram.argmax()]
period = 2*np.pi/max_f
period = int(period)

#From Horne 1986
No = len(ysub)
Ni = -6.362+1.193*No+0.00098*No**2
F = 1-(1-e**(-max_pgram))**Ni
F = 100*F
prob = "%0.2f" % F

print "Period is %s days" % period
print "False alarm probability is %s percent" % prob


subplot(211)
plot(x, dm, 'k+')
errorbar(x, dm, yerr=error,linestyle='None',color='black')
#xticks(fontsize=18,fontproperties=prop)
#yticks(fontsize=18,fontproperties=prop)
ylabel(r'$\Delta$ DM [10$^{-3}$ pc cm$^{-3}$]')
xlabel("MJD")
    
subplot(212)
plot(f, pgram)
ylabel('Power')
xlabel("Frequency")

plt.show()
