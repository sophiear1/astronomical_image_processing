# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 15:13:09 2021

@author: sophi

"""
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
hdulist = fits.open("A1_mosaic.fits")
data = hdulist[0].data
x1 = np.linspace(0, len(data[0]), len(data[0])+1)
y1= np.linspace(0, len(data), len(data)+1)

def gauss2d(x,y,sig=1,x_mean=0,y_mean=0):
    front = 1/(2*np.pi*(sig**2))
    power = (((x-x_mean)**2)+((y - y_mean)**2))/(2*(sig**2))
    g = front*np.exp(-1*power)
    return g  

def deVs(I, r, R):
    amp = I
    power = -7.669*(((r/R)**(1/4))-1)
    return amp * np.exp(power)

print(gauss2d(1,1))
print(gauss2d(-1, -1))
print(gauss2d(0.5, 0.5))


#%%

x,y = np.meshgrid(x1, y1, sparse = True)

gaussian_500 =np.empty([len(data),len(data[0])])
for i in range(0,len(data)):
    for j in range(0,len(data[0])):
        gaussian_500[i][j] =gauss2d(x[0][j],y[i],sig = 500, x_mean = len(data[0])/2, y_mean = len(data)/2)

plt.imshow(gaussian_500)
plt.show()
#%%
hdulist = fits.open("A1_mosaic.fits")
data = hdulist[0].data

x1 = np.linspace(1500, 2000, 500).astype(int)
y1= np.linspace(1500, 2000, 500).astype(int)

x,y = np.meshgrid(x1, y1, sparse = True)

gaussian_100 =np.empty([len(data),len(data[0])])
for i in range(0, len(y)):
    for j in range(0,len(x[0])):
        gaussian_100[i+1500,j+1500] =gauss2d(x[0,j],y[i],sig = 50, x_mean = 1750, y_mean = 1750)

plt.imshow(gaussian_100)
plt.show()

#%%
#%%
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
from skimage import filters
def shuffle_labels(labels):
    indices = np.unique(labels[labels != 0])
    indices = np.append(
            [0],
            np.random.permutation(indices)
            )
    return indices[labels]
file = gaussian_100
data = file
source_threshold = filters.threshold_isodata(data)#(5*10**-6)
t = source_threshold
thresholded = (data >= t) 
distance = ndi.distance_transform_edt(thresholded)
local_maxima = morphology.local_maxima(distance)
maxi_coords = np.nonzero(local_maxima)
local_maxima = morphology.local_maxima(distance)
maxi_coords = np.nonzero(local_maxima)
markers = ndi.label(local_maxima)[0]
labels = segmentation.watershed(data, markers)
labels_masked = segmentation.watershed(data,markers, mask = thresholded, connectivity = 10)
shuffled_masked_labels = shuffle_labels(labels_masked)
plt.imshow(shuffled_masked_labels)
plt.show()

#%%


#%%
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
    #data = data[100:-100,200:-200]
    t = source_threshold # Threshold for the watershed 
    t2 = source_threshold#3500 # Threshold for finding sources
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
    magdata = 0 -2.5 * np.log10(all_source_brightness)
    mag_sorted = np.sort(magdata)
    number = np.linspace(0, len(magdata),len(magdata)+1)
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

