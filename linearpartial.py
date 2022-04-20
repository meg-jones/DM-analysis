#! /usr/bin/env python
import matplotlib.font_manager as fm
from scipy.interpolate import interp1d
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
    dmxfile = pname+'.dmx'
    mjd,dm0,err = np.loadtxt(dmxfile,unpack=True, usecols=(0,1,2))
except IndexError:
	print "file not read" % sys.argv[0]
	sys.exit(1)
dm0=1000*dm0
err=1000*err

#First fit
start=54400
end=55300
x,y,error=[],[],[]
for i in range(0, len(mjd)):
	if mjd[i] > start and mjd[i] < end:
		x.append(mjd[i])
		y.append(dm0[i])
		error.append(err[i])
x=np.array(x)
y=np.array(y)
error = np.array(error)





#Fit linear function
def func(x, a, b):
  return a*x+b

p0 = sy.array([1,1])
coeffs, matcov = curve_fit(func, x, y, p0, maxfev=20000)
perr = np.sqrt(np.diag(matcov))
yaj1 = func(x, coeffs[0], coeffs[1])
slope=np.array(coeffs[0])*365.25
print "Linear Fit"
print "Slope is %s" % slope
print(perr[0]*365.25)
xl=np.arange(min(x), max(x), 1)
gl= coeffs[0]*xl+coeffs[1]



#Second Fit
start=55300
end=70000
x2,y2,error2=[],[],[]
for i in range(0, len(mjd)):
	if mjd[i] > start:
		x2.append(mjd[i])
		y2.append(dm0[i])
		error2.append(err[i])
x2=np.array(x2)
y2=np.array(y2)
error2 = np.array(error2)

def func(x2, a, b):
  return a*x2+b

p0 = sy.array([1,1])
coeffs, matcov = curve_fit(func, x2, y2, p0, maxfev=20000)
perr = np.sqrt(np.diag(matcov))
yaj2 = func(x2, coeffs[0], coeffs[1])
slope=np.array(coeffs[0])*365.25
print "Linear Fit"
print "Slope is %s" % slope
print(perr[0]*365.25)
xl2=np.arange(min(x2), max(x2), 1)
gl2= coeffs[0]*xl2+coeffs[1]



#Subtract off function
subtract=[]
for j in range(0,len(y)):
	 subtract.append(y[j]-yaj1[j])

for m in range(0,len(y2)):
	 subtract.append(y2[m]-np.mean(y2))

#subtract.extend(y2)
#subtract = np.array(subtract)

days = []
days.extend(x)
days.extend(x2)


errors = []
errors.extend(error)
errors.extend(error2)
errors = np.array(errors)


filename=pname+'subtract.txt'	
file=open(filename,'w')
#x subtract
for k in range(0,len(days)):
	file.write(str(days[k])+'  '+str(subtract[k])+'  '+str(errors[k])+'  '+'\n')

file.close()


#MJD to Year
t = Time(days, format='mjd')
years = t.decimalyear

y_upper = max(dm0) + (max(dm0)-min(dm0))/3
y_lower = min(dm0) - (max(dm0)-min(dm0))/4
subtract_upper = max(subtract) + (max(subtract)-min(subtract))/2.2
subtract_lower = min(subtract) - (max(subtract)-min(subtract))/4
x_upper = max(mjd) + (max(mjd)-min(mjd))/55
x_lower = min(mjd) - (max(mjd)-min(mjd))/55
xmax = Time(x_upper, format='mjd')
xmin = Time(x_lower, format='mjd')

years_upper = xmax.decimalyear
years_lower = xmin.decimalyear


#Compute reduced chi-squared
N = len(dm0)
m = np.mean(dm0)
center = dm0-m
dm = center/dm0.std()
m2 = np.mean(subtract)
chi_red = np.sum(((dm0-m)/err)**2)/(N-1)
chi1 = "%0.0f" %chi_red
chi_red_fit = np.sum((subtract-m2/errors)**2)/(len(subtract)-1-2)
chi_red_fit1 = np.sum((yaj1-np.mean(yaj1)/error)**2)/(len(yaj1)-1-2)
chi_red_fit2 = np.sum((yaj2-np.mean(yaj2)/error2)**2)/(len(yaj2)-1-2)
chi2 = "%0.2f" %chi_red_fit

print "Reduced chi-squared without fit is %s" % chi_red,chi_red_fit1,chi_red_fit2
print "Reduced chi-squared with both fit is %s" % chi_red_fit



fig = plt.figure(figsize = (10,6.5))
gs = gridspec.GridSpec(2,1)
gs.update(hspace=0)
ax1 = plt.subplot(gs[0,0])
ax1.plot(mjd, dm0, 'k+')
ax1.errorbar(mjd, dm0, yerr=err,linestyle='None',color='black')
ax1.plot(xl, gl, 'b-')
#plt.plot(xl2, gl2, 'g-')
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
fig.text(0.15, 0.84, r'J1600$-$3053',fontsize=18, fontproperties=prop)
fig.text(0.15,0.53,r'$\mathrm{\chi_r^2}$ = %s'%(chi1),fontproperties=prop,fontsize=16)

ax2 = plt.subplot(gs[1,0])
ax2.plot(years, subtract, 'k+')
ax2.errorbar(years, subtract, yerr=errors,linestyle='None',color='black')
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


