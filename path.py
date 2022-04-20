import commands as com
import re
from astropy import coordinates
import astropy.units as u
import numpy as np
import pylab as pl


psr=np.loadtxt('positions.txt', unpack=True, usecols=(0,), dtype='str')
distot, pmratot, pmdectot = np.loadtxt('positions.txt', unpack=True, usecols=(8, 10, 11))
vt = np.loadtxt('velocities.txt', unpack=True, usecols=(1,))
vtranstot = 1000*vt
rajtot, decjtot = np.loadtxt('positions.txt', unpack=True, usecols=(1,2))
 
for i in range(0,36):

	def psrcatlist(pulsarname):
	    # This gets and prints the info from psrcat which you have presumably downloaded, putting it in to a list
	    cmd = './psrcat -db_file "nanograv.db" -o short -nohead -nonumber -c "name raj decj pmra pmdec dist" ' + pulsarname
 	    output = com.getoutput(cmd)
	    L = re.findall(r"[\-w.|/-w-|/-w:|/-w+]+",output)
	    ra = L[1]
	    dec = L[2]
#	    pmra = float(L[3])
#	    pmdec = float(L[4])
	#    vtrans = float(L[5])*1000
#	    dist = float(L[5]) * 3.085677E19
		
	    pmra = pmratot[i]
	    pmdec = pmdectot[i]
	    dist = distot[i]*3.085677E19
      
	    #Read in velocities
	    vtrans = vtranstot[i]     
 
	    #This reformats the RA
	    l1 = ra.split(':')
	    raj = l1[0] +'h'+l1[1]+ 'm'+ l1[2] + 's'
# 	    ra  = rajtot[i]       
	    #This reformats the DEC
	    l2 = dec.split(':')
	    decj = l2[0] + 'd' + l2[1] + 'm' + l2[2] + 's'
#	    dec = decjtot[i]        

	    #This inputs the RA and DEC into an astropy coordinate system for easy conversion to degrees
	    coord = coordinates.ICRS(ra = raj,dec = decj)     
    
	    ra_okay = ra.is_within_bounds('90d','270d')
    
	    return ra, dec, pmra, pmdec, vtrans, dist, ra_okay



	def graphVeff():
	    #pname = raw_input('Enter Pulsar Name (if finished enter q) :  ')
		pname = psr[i]
	    #while pname != 'q':
        
	        #observingfrequency = raw_input('Enter the observing frequency: ')
	        #eta = raw_input('Enter the eta value for your pulsar: ')
        
	        #wavelength = (3 * np.power(10,8)) / (float(observingfrequency) * 1000000)
 	        #eta = float(eta)
	        #psi = np.linspace(0,90,40)
                
	        C = 3e8
        	e = 0.410152374
        	t = 0 
        
        	#setting up the variables we will need later
        	ra, dec, pmra, pmdec, dist, ra_okay = psrcatlist(pname)
        
        	#This calculates Beta 
        	beta  = np.arcsin( np.sin(dec) * np.cos(e) - np.cos(dec) * np.sin(e) * np.sin(ra))
    
        	#From Beta, this calculates Theta
        	theta = (np.arccos( (np.cos(e) - np.sin(beta)*np.sin(dec)) / (np.cos(beta)*np.cos(dec))))
        	if ra_okay == True:
            		theta = -theta
        	print theta  
        
	        #From Theta, this calculates PMLat and PMLong in the ecliptic plane
        	pmeLat = - pmra * np.sin(theta) + pmdec * np.cos(theta)
        	pmeLon = pmra * np.cos(theta) + pmdec * np.sin(theta)
        
        	#This calculates the magnitude
        	pmTot = np.sqrt( (pmeLat * pmeLat) + (pmeLon * pmeLon))
        
        	#Using trig, this translates the the PMs into Velocities
        	vLat = vtrans * pmeLat / pmTot
        	vLon = vtrans * pmeLon /  pmTot
        	vLatAU = vLat * 3.15569E7 / 149597870700
        	vLonAU = vLon * 3.15569E7 / 149597870700
		vPsr = np.sqrt( vLat*vLat + vLon*vLon )
    
        	#calculates the velocity of the earth, first long number is 1 AU (the amplitude of the earths orbit), the last number is the seconds in a year (to convert the whole thing to m/s)
        	vEarLon = 149597870700 * 2 * np.pi * np.sin(beta) * np.cos(2 * np.pi * t)/ 3.15569E7
        	vEarLat = 149597870700 * -2 * np.pi * np.sin(2 * np.pi* t)/ 3.15569E7
        	vEar = np.sqrt( vEarLon * vEarLon + vEarLat * vEarLat )
                
        	s = .5
		time, dm = np.loadtxt(psr[i]+'.dmx',unpack=True, usecols=(0,1))

        	t = (time-min(time))/365.25
        
        	x = np.array((((1-s)*vLonAU*t)+(s*np.cos(2*np.pi*t)))*np.cos(theta) + (((1-s)*vLatAU*t) + (s*np.sin(beta)*np.sin(2*np.pi*t))) * np.sin(-theta))
        	y = np.array(-(((1-s)*vLonAU*t)+(s*np.cos(2*np.pi*t)))*np.sin(-theta) + (((1-s)*vLatAU*t) + (s*np.sin(beta)*np.sin(2*np.pi*t)))*np.cos(theta)) 
	
		#Write positions to file
		filename=psr[i]+'path.txt'
		file=open(filename,'w')
		for k in range(0,len(x)):
			file.write(str(t[k])+'  '+str(x[k])+'  '+str(y[k])+'  '+str(dm[k])+'\n')
		file.close()
        
        	pl.plot(x,y,label=pname)
        	pl.xlabel('Longitudinal Distance (AU)')
        	pl.ylabel('Latitudinal Distance(AU)')
        	#pname = raw_input('Enter another pulsar (if finished enter q):  ')
    
	        pl.legend()
	   #    pl.xlim([-3,3])
	   #    pl.ylim([-10,10])
	        pl.show()
	        pl.close()     

	graphVeff()
