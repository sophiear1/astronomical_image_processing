# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 09:23:19 2021

@author: sar219
"""
#%%Imports
from astropy.io import fits
import numpy as np
import  scipy as sp
import matplotlib.pyplot as plt
import scipy.optimize as op
import numpy.ma as ma
def gauss(x, a, x0, sigma,C):
    return a*np.exp(-(x-x0)**2/(2*sigma**2)) + C
#%%Open File
hdulist = fits.open("H:\Documents\Labs\Year_3\Astro_Imaging\A1_mosaic\A1_mosaic.fits")
header = hdulist[0].header
data = hdulist[0].data
mask = ((3418.46-13.0034)<data) & (data>(3418.46+13.0034))
mask2 = np.invert(mask)
data2 = ma.array(data,mask=mask2, fill_value=0)
print(data2.filled(0))
hdu = fits.PrimaryHDU(data2.filled(0))
hdul = fits.HDUList([hdu])
hdul.writeto('new3.fits')
#%%
#Inital Histogram 
mask_high = (4000 > data) & (2000 < data) # mask 7
#Actual range = 0 to 65535
low_data = data[mask_high]
counts, bins, bars = plt.hist(low_data, bins =700, histtype = 'bar', align = 'mid')
plt.xlim([3300,3600])
plt.show()
#Peak = Average background noise for counts in the image 
#Big spike 
#Fit gaussian Before + After
#Tails = Contain photons from astronomical source 
#Spread = Noise 
#Plot Gaussian
midpoints = np.array([])
for i in range(len(bins)-1):
    mid = (bins[i] + bins[i+1])/2
    midpoints = np.append(midpoints, [mid])
#Midpoints = x, counts= y
par,cov = op.curve_fit(gauss, midpoints, counts, p0 = [400000, 3420, 5, 0])
plt.plot(midpoints,gauss(midpoints,*par), color = 'purple')
plt.hist(low_data, bins = 700)
plt.xlim([3300,3600])
plt.show()
print(par)
print(cov)
# Remove peak
counts_corr = counts
peak = np.max(counts)
loc = np.where(counts == peak)
print(counts_corr)
#
counts_corr[loc] = counts[np.asarray(loc) -1]
par,cov = op.curve_fit(gauss, midpoints, counts_corr, p0 = [400000, 3420, 5, 0])
plt.plot(midpoints,gauss(midpoints,*par), color = 'purple')
plt.hist(low_data, bins = 700)
plt.xlim([3300,3600])
print(par)
plt.show()
#photutil

