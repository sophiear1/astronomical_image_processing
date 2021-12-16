# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 14:38:46 2021

@author: sophi
"""
from astropy.stats import sigma_clipped_stats
from astropy.io import fits
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.optimize as op
import numpy.ma as ma
import pandas as pd
import skimage
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
def gauss(x, a, x0, sigma,C):
    return a*np.exp(-(x-x0)**2/(2*sigma**2)) + C
def newbounds(x1,x2,y1,y2):
    width = (x2-x1)/2
    newx2 = x2+width
    newx1 = x1-width
    height = (y2-y1)/2
    newy2 = y2+height
    newy1 = y1-height
    return int(newx1),int(newx2),int(newy1),int(newy2)
def func(file, source_threshold, edges, connect, n_removed, points_min, points_max):
    #file = "A1_mosaic.fits"
    #source_threshold = 3600
    #edges = sobel
    #connect = 20
    #n_removed = 315
    #points_min = 200
    #points_max = 1000
    hdulist = fits.open(file)
    header = hdulist[0].header
    data = hdulist[0].data
    #data = data[100:-100,200:-200]
    calibration = header['MAGZPT']
    call_error = header['MAGZRR'] 
    t = source_threshold # Threshold for the watershed 
    t2 = 3500 # Threshold for finding sources
    data1 = data
    thresholded = (data1 >= t)
    thresholded2 = (data1 >= t2)
    distance = ndi.distance_transform_edt(thresholded2)
    local_maxima = morphology.local_maxima(distance)
    maxi_coords = np.nonzero(local_maxima)
    local_maxima = morphology.local_maxima(distance)
    maxi_coords = np.nonzero(local_maxima)
    markers = ndi.label(local_maxima)[0]
    labels = segmentation.watershed(data, markers)
    edges = skimage.filters.prewitt(data)
    #########################
    #########################
    labels_masked = segmentation.watershed(data,markers, mask = thresholded, connectivity = connect)
    #########################
    #########################
    shuffled_masked_labels = shuffle_labels(labels_masked)
    regions = regionprops(shuffled_masked_labels)
    props = regionprops_table(shuffled_masked_labels, properties=('label','area','perimeter','slice','bbox'))
    objects = pd.DataFrame(props)
    wr = objects.nlargest(n_removed, 'area')
    data_corr = hdulist[0].data
    for i in range(len(wr)):
        data_corr[wr.iloc[i]['slice']] = [0]
    expanded_bbox=np.empty([len(objects['label']),4],np.int64())
    for i in range(len(objects['label'])):
        x1= objects['bbox-0'][i]
        y1= objects['bbox-1'][i]
        x2= objects['bbox-2'][i]
        y2= objects['bbox-3'][i]
        x1,x2,y1,y2 = newbounds(x1,x2,y1,y2)
        expanded_bbox[i]=[x1,x2,y1,y2]   
    data_bbox = data_corr
    objslices = np.array(objects['slice'])
    all_source_brightness = np.array([])
    for i in range(len(expanded_bbox)):
        region = data_bbox[expanded_bbox[i][0]:expanded_bbox[i][1],expanded_bbox[i][2]:expanded_bbox[i][3]]
        eboxint = np.sum(region)
        srcint = np.sum(data_bbox[objslices[i]])
        if eboxint!= 0: 
            source_size = data_bbox[objslices[i]].size
            bkg = eboxint-srcint
            bkg_size = len(region)*len(region[0])
            area = bkg_size - source_size
            source_brightness = srcint - (bkg/area)*source_size 
            if source_brightness > 0:
                all_source_brightness = np.append(all_source_brightness, source_brightness)  
    magdata = calibration -2.5 * np.log10(all_source_brightness)
    mag_sorted = np.sort(magdata)
    number = np.linspace(0, len(magdata),len(magdata))
    cumulative = np.cumsum(number)
    x = mag_sorted
    y = np.log10(cumulative)
    plt.scatter(x, y)
    plt.xlabel('magnitude')
    plt.ylabel('log(count)')
    x_straight = x[points_min:points_max]
    y_straight = y[points_min:points_max]
    m,b = np.polyfit(x_straight, y_straight, 1)
    print(m)
    plt.plot(x_straight, m*x_straight + b, c = 'red')
    plt.show()

func(file = "A1_mosaic.fits", 
     source_threshold = (3421+(13*3)),
     edges = 'sobel',
     connect = 20,
     n_removed =80,
     points_min = 100,
     points_max = 600)  

#%%
#all egdes
def binary(array):
    for i in range(0, len(array)):
        for j in range(0, len(array[0])):
            if array[i][j] !=0:
                array[i][j] = 1
    return array

x =sobel(data)
#%%


hdulist = fits.open("A1_mosaic.fits")
header = hdulist[0].header
data = hdulist[0].data
from skimage.filters import sobel, scharr, prewitt, roberts, laplace, farid
fig, ([ax0, ax1], [ax2, ax3], [ax4, ax5]) = plt.subplots(nrows = 3, ncols = 2)
ax0.imshow(np.log10(sobel(data)-3200), cmap = 'Greys_r')
ax1.imshow(scharr(data), cmap = 'Greys_r')
ax2.imshow(prewitt(data), cmap = 'Greys_r')
ax3.imshow(roberts(data), cmap = 'Greys_r')
ax4.imshow(laplace(data), cmap = 'Greys_r')
ax5.imshow(farid(data), cmap = 'Greys_r')
plt.show()