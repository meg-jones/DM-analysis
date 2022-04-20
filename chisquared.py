#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pylab import *
import scipy as sy


try:
    dmxfile = sys.argv[1]
    data = np.loadtxt(dmxfile)
except IndexError:
	print "file not read" % sys.argv[0]
	sys.exit(1)

x= data[:,0]
y= data[:,1]
error= data[:,2]
time=[]
m = np.mean(y)
center = y-m
dm = center/y.std()
N=len(x)
for i in range(0,N):
  for j in range(i+1,N):
    time.append(np.fabs(x[j]-x[i]))

#Define output frequencies
f = np.linspace(0.005,2*np.pi/min(time),100000)
pgram = signal.lombscargle(x, dm, f)
max_pgram = max(pgram)
max_f = f[pgram.argmax()]
period = 2*np.pi/max_f
period = int(period)

No = len(y)
Ni = -6.362+1.193*No+0.00098*No**2
F = 1-(1-exp(-max_pgram))**Ni
F = 100*F
prob = "%0.2f" % F

print "Period is %s days" % period
print "False alarm probability is %s percent" % prob
