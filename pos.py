#! /usr/bin/env python

from matplotlib import pyplot as PLT
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import sys
import os
from pylab import *
import scipy as sy


hfont = {'fontname':'Helvetica_Reg'}
prop = fm.FontProperties(fname='/home/mljones1/Desktop/Helvetica_Reg.ttf')

RA, DEC = np.loadtxt('pos.txt',unpack=True, usecols=(0,1))
l, b = np.loadtxt('pos.txt',unpack=True, usecols=(4,5))



x=np.arange(0,24,.1)
ecliptic=23.5*np.sin(2*3.14159/24*x)

x=np.arange(-180,180,1)
ecliptic=23.5*np.sin(2*3.14159/24*x)

near_x=[0.3833,0.5,16.23,20.167,16.7167,21.75,16]
near_y=[9.3833,4.85,-22.5,-13.3833,-12.4,-7.833,-30.8833]

near_l=[9.070,8.91,-64.348,-65.788,-71.087,-121.924,-146.025]
near_b=[6.309,1.446,-10.072,-1.257,9.778,6.491,5.313]

l_neg=[]
for i in range(0,len(l)):
	if l[i]>180:
		l_neg.append(l[i]-360)
	else:
		l_neg.append(l[i])

#Composite plots
fig = plt.figure(figsize=(10,7))
plt.plot(RA,DEC,'k*',ms=8)
plt.plot(near_x, near_y, 'k^',ms=9,label=r'MSPs within 10$^{\circ}$')
plt.plot(x,ecliptic,'k-.',linewidth=2)
plt.xlabel('RA [hrs]', labelpad=10, fontsize=21, fontproperties=prop)
plt.ylabel('DEC [deg]', labelpad=10, fontsize=21, fontproperties=prop)
plt.xticks([0,4,8,12,16,20,24],fontsize=20,fontproperties=prop)
plt.xlim([0,24])
plt.legend(loc='lower left')
plt.yticks(fontsize=20,fontproperties=prop)
plt.tick_params(axis='x', pad=10)
plt.tick_params(axis='y', pad=10)
plt.grid(b=True, which='major', color='0.75', linestyle='-')
plt.tight_layout()
ax = plt.gca()
ax.invert_xaxis()
plt.savefig('positions.pdf',fmt='pdf', dpi=2000)

plt.show()

#Composite plots
fig = plt.figure(figsize=(10,7))
plt.plot(l_neg,b,'k*',ms=8)
plt.plot(near_l, near_b, 'k^',ms=9,label=r'MSPs within 10$^{\circ}$')
#plt.plot(x,ecliptic,'k-.',linewidth=2)
plt.xlabel('l [deg]', labelpad=10, fontsize=21, fontproperties=prop)
plt.ylabel('b [deg]', labelpad=10, fontsize=21, fontproperties=prop)
#plt.xticks([0,4,8,12,16,20,24],fontsize=20,fontproperties=prop)
#plt.xlim([0,24])
plt.legend(loc='lower right')
plt.yticks(fontsize=20,fontproperties=prop)
plt.tick_params(axis='x', pad=10)
plt.tick_params(axis='y', pad=10)
plt.grid(b=True, which='major', color='0.75', linestyle='-')
plt.tight_layout()
ax = plt.gca()
ax.invert_xaxis()
plt.savefig('positions_lb.pdf',fmt='pdf', dpi=2000)

plt.show()
