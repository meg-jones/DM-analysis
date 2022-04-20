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
    dmxfile = '/home/mljones1/dmproject/trajectory/'+pname+'.dmx'
    mjd,dm0,err = np.loadtxt(dmxfile,unpack=True, usecols=(0,1,2))
except IndexError:
	print "file not read" % sys.argv[0]
	sys.exit(1)

dm0=1000*dm0
err=1000*err
#First fit


start=0
end=54733
x,y,error=[],[],[]
for i in range(0, len(mjd)):
	if mjd[i] > start and mjd[i] < end:
		x.append(mjd[i])
		y.append(dm0[i])
		error.append(err[i])
x=np.array(x)
y=np.array(y)
error = np.array(error)


#Fit linear + sine function
def func(x, a, b, c, d, e):
  return a*x+b+c*sin(d*x+e)

p0 = sy.array([1,1,1,0.0172,1])
coeffs, matcov = curve_fit(func, x, y, p0, maxfev=20000)
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
print "Amplitude is %s" % coeffs[2]*10
print(perr[2])*10
print(max(x)-min(x))

#Compute reduced chi-squared
N = len(y)
m = np.mean(y)
center = y-m
dm = center/y.std()
chi_red = np.sum(((y-m)/error)**2)/(N-1)
chi1 = "%0.0f" %chi_red
chi_red_fit3 = np.sum(((y-yaj3)/error)**2)/(N-1-5)
chi2 = "%0.0f" %chi_red_fit3
print(chi1)
print(chi2)

xx=np.arange(min(x), max(x), 1)
gx = coeffs[0]*xx+coeffs[1]+coeffs[2]*sin(coeffs[3]*xx+coeffs[4])

sigma = np.mean(error)
slope = abs(slope) #1e-4
A = abs(coeffs[2]) #1e-4
t = sigma/(slope/365.25 + A/58) #days
print "Timescale is %s" % t


#Second Fit
start=54902
end=56594
x2,y2,error2=[],[],[]
for i in range(0, len(mjd)):
	if mjd[i] > start and mjd[i] < end:
		x2.append(mjd[i])
		y2.append(dm0[i])
		error2.append(err[i])
x2=np.array(x2)
y2=np.array(y2)
error2 = np.array(error2)



#Fit linear + sine function
def func(x2, a, b, c, d, e):
  return a*x2+b+c*sin(d*x2+e)

p0 = sy.array([1,1,1,0.0172,1])
coeffs, matcov = curve_fit(func, x2, y2, p0, maxfev=20000)
yaj3b = func(x2, coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4])
perr = np.sqrt(np.diag(matcov))
period=2*np.pi/coeffs[3]
perioderr=(2*np.pi/(coeffs[3]-perr[3])-2*np.pi/(coeffs[3]+perr[3]))/2
slope=np.array(coeffs[0])*365.25
print "Both Fit"
print "Period is %s days" % period
print(perioderr)
print "Slope is %s" % slope
print(perr[0]*365.25)
print "Amplitude is %s" % coeffs[2]*10
print(perr[2]*10)
print(max(x)-min(x))
#Compute reduced chi-squared
N = len(y2)
m2 = np.mean(y2)
center2 = y2-m2
dm2 = center2/y2.std()
chi_red = np.sum(((y2-m2)/error2)**2)/(N-1)
chi1 = "%0.0f" %chi_red
chi_red_fit3 = np.sum(((y2-yaj3b)/error2)**2)/(N-1-5)
chi2 = "%0.0f" %chi_red_fit3
print(chi1)
print(chi2)

xx2=np.arange(min(x2), max(x2), 1)
gx2 = coeffs[0]*xx2+coeffs[1]+coeffs[2]*sin(coeffs[3]*xx2+coeffs[4])

sigma = np.mean(error2)
v
slope= abs(slope)#1e-3
A = abs(coeffs[2]) #1e-3
t = sigma/(slope/365.25 + A/58) #days
print "Timescale is %s" % t



#plt.plot(mjd,dm0,'x',xx,gx,'r-',xx2,gx2,'g-')
#plt.errorbar(mjd,dm0,yerr=err,fmt=None)
#plt.xlabel("MJD (days)")
#plt.ylabel("DMX")
#plt.savefig('fit.pdf',fmt='pdf')
#plt.show()



start=max(x)
end=min(x2)
x3,y3,error3=[],[],[]
for i in range(0, len(mjd)):
	if mjd[i] > start and mjd[i] < end:
		x3.append(mjd[i])
		y3.append(dm0[i])
		error3.append(err[i])
x3 = np.array(x3)
y3 = np.array(y3)
error3 = np.array(error3)


x4,y4,error4=[],[],[]
for i in range(0, len(mjd)):
	if mjd[i] > max(x2):
		x4.append(mjd[i])
		y4.append(dm0[i])
		error4.append(err[i])
x4 = np.array(x4)
y4 = np.array(y4)
error4 = np.array(error4)


#Subtract off function
subtract=[]
for j in range(0,len(y)):
	 subtract.append(y[j]-yaj3[j])

subtract.extend(y3-np.mean(y3))

for m in range(0,len(y2)):
	 subtract.append(y2[m]-yaj3b[m])

subtract.extend(y4-np.mean(y4))

subtract = np.array(subtract)



days = []
days.extend(x)
days.extend(x3)
days.extend(x2)
days.extend(x4)


errors = []
errors.extend(error)
errors.extend(error3)
errors.extend(error2)
errors.extend(error4)
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
N = len(y)
m = np.mean(y)
center = y-m
dm = center/y.std()
m2 = np.mean(subtract)
chi_red = np.sum(((y-m)/error)**2)/(N-1)
chi1 = "%0.0f" %chi_red
chi_red_fit = np.sum((subtract-m2/errors)**2)/(N-1-5)
chi2 = "%0.2f" %chi_red_fit

print "Reduced chi-squared without fit is %s" % chi_red
print "Reduced chi-squared with both fit is %s" % chi_red_fit



fig = plt.figure(figsize = (10,6.5))
gs = gridspec.GridSpec(2,1)
gs.update(hspace=0)
ax1 = plt.subplot(gs[0,0])
ax1.plot(mjd, dm0, 'k+')
ax1.errorbar(mjd, dm0, yerr=err,linestyle='None',color='black')
ax1.plot(xx, gx, 'b-')
plt.plot(xx2, gx2, 'g-')
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
fig.text(0.15, 0.84, r'J1713+0747',fontsize=18, fontproperties=prop)
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


