#! /usr/bin/env python

from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec
import sys
import os
from pylab import *
import scipy as sy
from scipy.optimize import curve_fit

#hfont = {'fontname':'Helvetica_Reg'}
#prop = fm.FontProperties(fname='/home/mljones1/Desktop/Helvetica_Reg.ttf')
#rcParams['mathtext.default']='regular'

mjd = []
dm = []

try:
    dmxfile = sys.argv[1]
    data = np.loadtxt(dmxfile)
except IndexError:
	print "file not read" % sys.argv[0]
	sys.exit(1)

for i in range(0,len(data)):
	mjd.append(data[i][0])
	dm.append(data[i][1])

#x =mjd-min(mjd)x_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)

x= data[:,0]
y= data[:,1]
error= data[:,2]

y=1000*y
error=1000*error

#Interpolate DM variations
#f=interp1d(mjd,dm)
#days=[]
#days=np.arange(min(mjd),max(mjd),1)

#tau=[10,20,30,40,50,60,90,120,150,180,210,240,365,480,540]    #days

#Find two points with time lag tau
#span=int(max(mjd)-min(mjd))
#mjdspan=np.linspace(min(mjd),max(mjd),span)
#function=[]

#for l in range(0,len(tau)):
#	dmchange=[]  return a*x+b*sin(c*x+d)+g

#	for j in range(0,span-tau[l]):
#		dmchange.append(f(mjdspan[j+tau[l]])-f(mjdspan[j]))
#	dmarray=np.array(dmchange)**2
#	function.applt.plot(x,y,'k+',xx,yy,'k--')
#plt.errorbar(x,y,yerr=error,ecolor='k',fmt=None)
#plt.xaxis.set_label_position('top') 
#ax1.xaxis.tick_top()
#plt.xlabel('MJD', labelpad=10,fontsize=21)
#plt.ylabel(r'$\Delta$ DM [10$^{-3}$ pc cm$^{-3}$]', labelpad=20, fontsize=21)
#plt.xticks(fontsize=20)
#plt.yticks(fontsize=20)
#plt.tick_params(axis='x',pad=0)
#plt.tick_params(axis='y', pad=5)
#fig.text(0.17, 0.84, 'J1614-2230',fontsize=20, fontproperties=prop)
#plt.show()pend(dmarray.mean())


#Fit function
def func(x, a, b, c, d, g):
  return a*x+b*sin(c*x+d)+g

p0 = sy.array([1,1,0.0172,1,1])
coeffs, matcov = curve_fit(func, x, y, p0)

yaj = func(x, coeffs[0], coeffs[1], coeffs[2], coeffs[3],coeffs[4])
print(coeffs)
print(matcov)

xx=np.arange(min(x),max(x),10)
yy=coeffs[0]*xx+coeffs[1]*sin(coeffs[2]*xx+coeffs[3])+coeffs[4]

#Composite plots
#fig = plt.figure(figsize=(10,7.5))
#fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
#ax1 = plt.subplot(211)
plt.plot(x,y,'k+',xx,yy,'k--')
plt.errorbar(x,y,yerr=error,ecolor='k',fmt=None)
#plt.xaxis.set_label_position('top') 
#ax1.xaxis.tick_top()
plt.xlabel('MJD', labelpad=10,fontsize=21)
plt.ylabel(r'$\Delta$ DM [10$^{-3}$ pc cm$^{-3}$]', labelpad=20, fontsize=21)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tick_params(axis='x',pad=0)
plt.tick_params(axis='y', pad=5)
#fig.text(0.17, 0.84, 'J1614-2230',fontsize=20, fontproperties=prop)
plt.show()


#ax2 = plt.subplot(212)
#ax2.plot(tau,function,'k+',ms=10,mew=1.1)
#plt.xlabel(r'$\tau$ [days]', labelpad=7, fontsize=21, fontproperties=prop)
#plt.ylabel(r'D$_{DM}$($\tau$)', labelpad=12, fontsize=21, fontproperties=prop)
#ax2.set_xscale("log")
#ax2.set_yscale("log")
#x_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
#ax2.xaxis.set_major_formatter(x_formatter)
#plt.xticks([10,50,100,200,300,500],fontsize=20,fontproperties=prop) 
#plt.yticks(fontsize=20,fontproperties=prop)
#plt.tick_params(axis='x', pad=5)
#plt.tick_params(axis='y', pad=5)
#plt.show()


