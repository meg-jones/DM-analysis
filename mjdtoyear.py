#! /usr/bin/env python

from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import sys
import os
from pylab import *
import scipy as sy
import math



function MJDtoYMD (mjd_in)
{
    year,month,day,hour,jd,jdi,jdf,l,n = [],[],[],[],[],[],[],[],[]
    
    
    #Julian day
    jd = math.floor(mjd_in) + 2400000.5

    #Integer Julian day
    jdi = math.floor(jd)
    
    #Fractional part of day
    jdf = jd - jdi + 0.5
    
    #Really the next calendar day?
    if jdf >= 1.0:
       jdf = jdf - 1.0
       jdi = jdi + 1
   


    hour = jdf * 24.0
    l = jdi + 68569
    n = math.floor(4*l/146097)
   
    l = math.floor(l)-math.floor((146097*n+3)/4)
    year = math.floor(4000*(l+1)/1461001)
    
    l = l - (math.floor(1461*year/4))+31
    month = math.floor(80*l/2447)
    
    day = l - math.floor(2447*month/80)
    
    l = math.floor(month/11)
    
    month = math.floor (month + 2 - 12 * l);
    year = Math.floor (100 * (n - 49) + year + l);

    if (month < 10)
       month = "0" + month;
       
    if (day < 10)
       day = "0" + day;
    
    //year = year - 1900;
    
    return (new Array (year, month, day));

} 
