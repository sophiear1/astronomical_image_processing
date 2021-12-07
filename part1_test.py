# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 09:33:27 2021

@author: sar219
"""
from astropy.stats import sigma_clipped_stats
from astropy.io import fits
import numpy as np
import  scipy as sp
import matplotlib.pyplot as plt
import scipy.optimize as op
import numpy.ma as ma
def gauss(x, a, x0, sigma,C):
    return a*np.exp(-(x-x0)**2/(2*sigma**2)) + C
hdulist = fits.open("H:\Documents\Labs\Year_3\Astro_Imaging\A1_mosaic\A1_mosaic.fits")
header = hdulist[0].header
data = hdulist[0].data  
#%%
data = data - 3421
hdu = fits.PrimaryHDU(data)
hdul = fits.HDUList([hdu])
#hdul.writeto('new2.fits')
#%%
import matplotlib.colors as colors
hdulist = fits.open("H:\Documents\Labs\Year_3\Astro_Imaging\A1_mosaic\A1_mosaic.fits")
header = hdulist[0].header
data = hdulist[0].data
mask = (3600<data) 
mask = np.invert(mask)
data2 = ma.array(data,mask=mask, fill_value=0)
data2 = data2.filled(0)
fig, ax = plt.subplots(2,1)
ax[0].imshow(data2, cmap='Greys_r')
pcm = ax[1].pcolor(data2, norm=colors.LogNorm(vmin=3600, vmax=65535,
                   cmap='Greys_r')
fig.colorbar(pcm, ax = ax[1], extend = 'max')
hdu = fits.PrimaryHDU(data2)
hdul = fits.HDUList([hdu])
hdul.writeto('new7.fits')
#%%
data = hdulist[0].data
mask = (3480<data) 
#mask = np.invert(mask)
data4 = ma.array(data,mask=mask, fill_value=60000)
data4 = data4.filled(60000)
#%%
import skimage.data as data
from skimage import segmentation, morphology, measure, exposure
from scipy import ndimage as ndi
from skimage.measure import label, regionprops, regionprops_table
def shuffle_labels(labels):
    indices = np.unique(labels[labels != 0])
    indices = np.append(
            [0],
            np.random.permutation(indices)
            )
    return indices[labels]
data1 = hdulist[0].data

t = 3600
data = data1# 65535 - data1
thresholded = (data1 >= t)

distance = ndi.distance_transform_edt(thresholded)
local_maxima = morphology.local_maxima(distance)
fig,ax = plt.subplots(1,1)
maxi_coords = np.nonzero(local_maxima)
local_maxima = morphology.local_maxima(distance)
maxi_coords = np.nonzero(local_maxima)

markers = ndi.label(local_maxima)[0]
labels = segmentation.watershed(data, markers)
labels_masked = segmentation.watershed(thresholded,markers,mask = thresholded, connectivity = 20)
f, (axo,ax1,ax2) = plt.subplots(1,3)
axo.imshow(thresholded,  cmap='Greys_r')
ax1.imshow(np.log(1 + distance),  cmap='Greys_r')
ax2.imshow(shuffle_labels(labels_masked),  cmap='magma')
plt.show()
contours = measure.find_contours(shuffle_labels(labels_masked))
plt.imshow(data1,  cmap='Greys_r')
for c in contours:
    plt.plot(c[:,1],c[:,0], color = 'pink')
#%%
shuffled_masked_labels = shuffle_labels(labels_masked)
mask = shuffled_masked_labels >1
difference = ma.array(data,mask=mask, fill_value=0)
difference = difference.filled(0)

plt.imshow(difference, cmap='Greys_r')
plt.show()

#%%
hdulist = fits.open("H:\Documents\Labs\Year_3\Astro_Imaging\A1_mosaic\A1_mosaic.fits")
#import data
import pandas as pd
plt.imshow(shuffled_masked_labels, cmap = 'Greys_r')
plt.show()
regions = regionprops(shuffled_masked_labels)
#get region properties
props = regionprops_table(shuffled_masked_labels, properties=('label',
                                                 'area',
                                                 'perimeter','slice','bbox'))
#Find and remove the objects with the largest area -
objects = pd.DataFrame(props)
wr = objects.nlargest(315, 'area')
data_corr = hdulist[0].data
for i in range(len(wr)):
    data_corr[wr.iloc[i]['slice']] = 0
    
#data_corr = data_corr - 3600
plt.imshow(data_corr, cmap = 'Greys_r')
plt.show()
plt.imshow(data, cmap = 'Greys_r')
plt.show()
def newbounds(x1,x2,y1,y2):
    width = (x2-x1)/4
    newx2 = x2+width
    newx1 = x1-width
    height = (y2-y1)/4
    newy2 = y2+height
    newy1 = y1-height
    return int(newx1),int(newx2),int(newy1),int(newy2)
expanded_bbox=np.empty([1974,4],np.int64())
for i in range(len(objects)):
    x= objects['bbox-0'][i]
    y= objects['bbox-1'][i]
    z= objects['bbox-2'][i]
    w= objects['bbox-3'][i]
    y1, y2, x1, x2 = newbounds(x,z,y,w)
    expanded_bbox[i]=[x1,x2,y1,y2]
    
data_bbox = hdulist[0].data
objslices = np.array(objects['slice'])
zerotally=np.array([])
all_source_brightness = np.array([])
for i in range(len(expanded_bbox)):
    region = data_bbox[expanded_bbox[i][0]:expanded_bbox[i][1],expanded_bbox[i][2]:expanded_bbox[i][3]]
    eboxint = np.sum(region)
    srcint = np.sum(data_bbox[objslices[i]])
    if srcint == 0:
        zerotally=np.append(zerotally,1)
    if eboxint!=0:
        source_size = data_bbox[objslices[i]].size
        bkg = eboxint-srcint
        bkg_size = len(region)*len(region[0])
        area = bkg_size - source_size
        source_brightness = srcint - (bkg/area)*source_size 
        all_source_brightness = np.append(all_source_brightness, source_brightness)
    

        
        
   
#%%
header = hdulist[0].header
callibration = header['MAGZPT']
call_error = header['MAGZRR']
#%%
hdu = fits.PrimaryHDU(shuffle_labels(labels_masked))
hdul = fits.HDUList([hdu])
hdul.writeto('new6.fits')


#%%
from skimage import feature
edges2 = feature.canny(data4, sigma=0.2)
plt.imshow(edges2)
plt.show()
#%%
plt.imshow(data4)