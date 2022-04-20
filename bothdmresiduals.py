#! /usr/bin/env python
import matplotlib.font_manager as fm
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pylab import *
import scipy as sy
from scipy.optimize import curve_fit
import matplotlib.gridspec as gridspec
from astropy.time import Time
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter

prop = fm.FontProperties(fname='/home/mljones1/dmproject/Helvetica_Reg.ttf')
try:
    pname = sys.argv[1]
except IndexError:
	print "file not read" % sys.argv[0]
	sys.exit(1)
dmxfile = pname+'.dmx'
mjd,dm0,err = np.loadtxt(dmxfile,unpack=True, usecols=(0,1,2))
dm0=1000*dm0
err=1000*err


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


#Fit linear + sine function
def func(x, a, b, c, d, e):
  return a*x+b+c*sin(d*x+e)

p0 = sy.array([1,1,1,0.0172,1])
coeffs, matcov = curve_fit(func, x, y, p0)
yaj3 = func(x, coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4])
perr = np.sqrt(np.diag(matcov))
period=2*np.pi/coeffs[3]
perioderr=(2*np.pi/(coeffs[3]-perr[3])-2*np.pi/(coeffs[3]+perr[3]))/2
slope=np.array(coeffs[0])*365.25
print "Both Fit"
print "Period is %s days" % period
print(perioderr)
print "Slope is %s" % slope
print(perr[0]*365.25)
print "Amplitude is %s" % coeffs[2]
print(perr[2])
print(max(x)-min(x))

xx=np.arange(min(x), max(x), 1)
gx = coeffs[0]*xx+coeffs[1]+coeffs[2]*sin(coeffs[3]*xx+coeffs[4])



#Compute reduced chi-squared
N = len(y)
m = np.mean(y)
center = y-m
dm = center/y.std()
chi_red = np.sum(((y-m)/error)**2)/(N-1)
chi1 = "%0.0f" %chi_red
chi_red_fit3 = np.sum(((y-yaj3)/error)**2)/(N-1-5)
chi2 = "%0.0f" %chi_red_fit3


print "Reduced chi-squared without fit is %s" % chi_red
print "Reduced chi-squared with both fit is %s" % chi_red_fit3




#Subtract off function
subtract=[]
for j in range(0,len(y)):
	 subtract.append(y[j]-yaj3[j])




#MJD to Year
t = Time(x, format='mjd')
years = t.decimalyear

y_upper = max(y) + (max(y)-min(y))/6
y_lower = min(y) - (max(y)-min(y))/3.8
subtract_upper = max(subtract) + (max(subtract)-min(subtract))/2.2
subtract_lower = min(subtract) - (max(subtract)-min(subtract))/4
x_upper = max(x) + (max(x)-min(x))/55
x_lower = min(x) - (max(x)-min(x))/55
xmax = Time(x_upper, format='mjd')
xmin = Time(x_lower, format='mjd')

years_upper = xmax.decimalyear
years_lower = xmin.decimalyear


fig = plt.figure(figsize = (10,6.5))
gs = gridspec.GridSpec(2,1)
gs.update(hspace=0)
ax1 = plt.subplot(gs[0,0])
ax1.plot(x, y, 'k+')
ax1.errorbar(x, y, yerr=error,linestyle='None',color='black')
ax1.plot(xx, gx, 'b-')
ax1.xaxis.set_label_position('top') 
ax1.xaxis.tick_top()
plt.xlabel('MJD', fontproperties=prop, fontsize=16)
plt.ylabel(r'  $\Delta$ DM [10$^{-3}$ pc cm$^{-3}$]', fontproperties=prop, fontsize=15)
plt.xticks(fontsize=15,fontproperties=prop)
plt.yticks(fontsize=15,fontproperties=prop)
plt.xlim(x_lower, x_upper)
plt.ylim(y_lower, y_upper)
#ax1.tick_params(axis='x', pad=5)
ax1.tick_params(axis='y', pad=5)
fig.text(0.15, 0.84, r'J2010$-$1323',fontsize=18, fontproperties=prop)
fig.text(0.15,0.53,r'$\mathrm{\chi_r^2}$ = %s'%(chi1),fontproperties=prop,fontsize=16)



ax2 = plt.subplot(gs[1,0])
ax2.plot(years, subtract, 'k+')
ax2.errorbar(years, subtract, yerr=error,linestyle='None',color='black')
plt.xlabel('Year', fontproperties=prop, fontsize=16)
plt.ylabel(r'$\Delta$ DM$-\overline{\rm{DM}}$ [10$^{-3}$ pc cm$^{-3}$]', fontproperties=prop, fontsize=15)
fig.text(0.15,0.44,r'$\mathrm{\chi_r^2}$ = %s'%(chi2),fontproperties=prop,fontsize=16)
plt.yticks(fontsize=15,fontproperties=prop)
#ax2.tick_params(axis='x', pad=10)
ax2.tick_params(axis='y', pad=5)
plt.xticks(fontsize=15,fontproperties=prop)
plt.yticks(fontsize=15,fontproperties=prop)
plt.ticklabel_format(useOffset=False)
plt.xlim(years_lower, years_upper)
plt.ylim(subtract_lower, subtract_upper)
plt.savefig(pname+'trend.pdf',fmt='pdf', dpi=2000)    
plt.show()
