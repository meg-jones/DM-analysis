#! /usr/bin/env python

from matplotlib import pyplot as PLT
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys
import os
from pylab import *
import scipy as sy
from scipy.optimize import curve_fit

mjd=[]
dm=[]

try:
    dmxfile = sys.argv[1]
    data = np.loadtxt(dmxfile)
except IndexError:
	print "file not read" % sys.argv[0]
	sys.exit(1)

for i in range(0,len(data)):
	mjd.append(data[i][0])
	dm.append(data[i][1])

x = data[:,0]
y = data[:,1]
error = data[:,2]
y=1000*y
error=1000*error


#Plot DM variations
#plt.plot(x,y,'+',)
#plt.errorbar(x,y,yerr=error,fmt=None)
#plt.legend()
#plt.xlabel('MJD')
#plt.ylabel('DMX (10$^{-3}$ pc cm$^{-3}$)')
#plt.show()

#Interpolate DM variations
f=interp1d(mjd,dm)

#Specify time lag (in days)
tau=[10,20,30,40,50,60,90,120,150,180,210,240,365,480,540]    

#Find two points with time lag tau
span=int(max(mjd)-min(mjd))
mjdspan=np.linspace(min(mjd),max(mjd),span)
function=[]
#function_err=[]
#f_errors=interp1d(mjd,error1)

for l in range(0,len(tau)):
	dmchange=[]
	for j in range(1,span-tau[l]):
		dmchange.append(f(mjdspan[j+tau[l]])-f(mjdspan[j]))
	dmarray=np.array(dmchange)**2
	function.append(dmarray.mean())

#plt.xscale('log')
#plt.yscale('log')
#plt.plot(tau,function,'+')
#plt.errorbar(tau,function,yerr=function_err,fmt=None)
#plt.xlabel("$\tau$ (days)")
#plt.ylabel("log(Structure Function)")
#plt.savefig('structure.pdf',fmt='pdf')
#plt.show()

#Composite plots
fig = plt.figure()
gs = gridspec.GridSpec(2, 1)
gs.update(hspace=0)
ax1 = plt.subplot(gs[0,0])
ax1.plot(x,y,'k+')
ax1.errorbar(x,y,yerr=error,ecolor='k',fmt=None)
plt.xlabel('MJD')
ax1.xaxis.set_label_position('top') 
ax1.xaxis.tick_top()
plt.ylabel(r'$\Delta$ DM (10$^{-3}$ pc cm$^{-3}$)')
plt.yticks(np.arange(0.5,4.5,0.5))
#setp(ax1.get_xticklabels(),visible=False)
#plt.title('B1855+09')
#setp(ax1.get_yticklabels(),visible=False)
fig.text(0.1, 0.85, 'B1855+09')

ax2 = plt.subplot(gs[1,0])
ax2.plot(tau,function,'k+')
ax2.loglog(tau, function)
plt.xticks([10,20,30,40,50,60,90,120,240,365,540]) 
plt.xlabel(r'$\tau$ (days)')
plt.ylabel(r'D$_{DM}$($\tau$)')

#ax2.errorbar(x1,y1,yerr=error1,fmt=None)
#plt.yticks(np.arange(-0.5,4.0,1))
x_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
ax2.xaxis.set_major_formatter(x_formatter)
#plt.savefig('panel.pdf',fmt='pdf', dpi=2000)
plt.savefig('panel.eps', format='eps', dpi=2000)
#fig.text(0.045, 0.5, 'DMX (10$^{-3}$ pc cm$^{-3}$)', rotation="vertical", va="center")
plt.show()
