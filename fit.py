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

prop = fm.FontProperties(fname='/home/mljones1/dmproject/Helvetica_Reg.ttf')
try:
    pname = sys.argv[1]
    dmxfile = pname+'.dmx'
    mjd,dm0,err = np.loadtxt(dmxfile,unpack=True, usecols=(0,1,2))
except IndexError:
	print "file not read" % sys.argv[0]
	sys.exit(1)
dmxfile = pname+'.dmx'
mjd,dm0,err = np.loadtxt(dmxfile,unpack=True, usecols=(0,1,2))
dm0=1000*dm0
err=1000*err


start=55550
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
#print "Linear Fit"
#print(coeffs)
#print(perr)
xl=np.arange(min(x), max(x), 1)
gl= coeffs[0]*xl+coeffs[1]

#Fit sine function
def func(x, c, d, e):
  return c*sin(d*x+e)

p0 = sy.array([1,0.0172,1])
coeffs, matcov = curve_fit(func, x, y, p0)
yaj2 = func(x, coeffs[0], coeffs[1], coeffs[2])
perr = np.sqrt(np.diag(matcov))
period=2*np.pi/coeffs[1]
perioderr=(2*np.pi/(coeffs[1]-perr[1])-2*np.pi/(coeffs[1]+perr[1]))/2
print "Periodic Fit"
print "Period is %s days" % period
#print(perioderr)
#print(coeffs)
#print(perr)

xs=np.arange(min(x), max(x), 1)
gs = coeffs[0]*sin(coeffs[1]*xs+coeffs[2])

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
#print(perioderr)
#print "Slope is %s" % slope
#print(perr[0]*365.25)
#print "Amplitude is %s" % coeffs[2]
#print(perr[2])

xx=np.arange(min(x), max(x), 1)
gx = coeffs[0]*xx+coeffs[1]+coeffs[2]*sin(coeffs[3]*xx+coeffs[4])



#Compute reduced chi-squared
N = len(y)
m = np.mean(y)
center = y-m
dm = center/y.std()
chi_red = np.sum(((y-m)/error)**2)/(N-1)
chi_red_fit1 = np.sum(((y-yaj1)/error)**2)/(N-1-2)
chi_red_fit2 = np.sum(((y-yaj2)/error)**2)/(N-1-4)
chi_red_fit3 = np.sum(((y-yaj3)/error)**2)/(N-1-5)


print "Reduced chi-squared without fit is %s" % chi_red
print "Reduced chi-squared with linear fit is %s" % chi_red_fit1
print "Reduced chi-squared with sinusoidal fit is %s" % chi_red_fit2
print "Reduced chi-squared with both fit is %s" % chi_red_fit3


fig = plt.figure(figsize=(10,4))
plt.plot(x,y,'k+')
plt.plot(xl,gl,'b-')
plt.plot(xs,gs,'k-')
plt.plot(xx,gx,'k-')
plt.xlabel('MJD', labelpad=10, fontsize=21, fontproperties=prop)
plt.ylabel(r'$\Delta$ DM [10$^{-3}$ pc cm$^{-3}$]', labelpad=20, fontsize=21, fontproperties=prop)
plt.xticks(fontsize=20,fontproperties=prop)
plt.yticks(fontsize=20,fontproperties=prop)
plt.tick_params(axis='x', pad=10)
plt.tick_params(axis='y', pad=10)
plt.errorbar(x,y,yerr=error,ecolor='k',fmt=None)
fig.text(0.17, 0.84,''+str(pname),fontsize=20, fontproperties=prop)
#plt.savefig('fit.pdf',fmt='pdf')
plt.show()


#Subtract off function
subtract=[]
for j in range(0,len(y)):
	 subtract.append(y[j]-yaj1[j])

#print(subtract)


#filename=pname+'subtract.txt'	
#file=open(filename,'w')
#x subtract
#for k in range(0,len(x)):
#	file.write(str(x[k])+'  '+str(subtract[k])+'  '+str(error[k])+'  '+'\n')
#file.close()


plt.subplot(211)
plt.plot(x, y, 'k+')
plt.errorbar(x, y, yerr=error,linestyle='None',color='black')
plt.plot(xs, gs, 'b-')
#plt.title(pname, fontsize=14,fontproperties=prop)
plt.ylabel(r'$\Delta$ DM [10$^{-3}$ pc cm$^{-3}$]', fontproperties=prop)
#plt.xlabel('MJD', fontproperties=prop)
plt.xticks(fontsize=20,fontproperties=prop)
plt.yticks(fontsize=20,fontproperties=prop)
#plt.tick_params(axis='x', pad=10)
plt.tick_params(axis='y', pad=10)
plt.set_xticklabels([])
fig.text(0.17, 0.84, 'J0931$-$1902',fontsize=20, fontproperties=prop)
plt.set_xticklabels([])   

plt.subplot(212)
plt.plot(x, subtract, 'k+')
plt.errorbar(x, subtract, yerr=error,linestyle='None',color='black')
plt.ylabel(r'$\Delta$ DM$-$\overline{\rm{DM}} [10$^{-3}$ pc cm$^{-3}$]', fontproperties=prop)
plt.yticks(fontsize=20,fontproperties=prop)
plt.tick_params(axis='y', pad=10)

    
plt.show()

