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
    return x, y, cumulative

magnitude, log_count, count = func(file = "A1_mosaic.fits", 
     source_threshold = (3421+(13*3)),
     edges = 'sobel',
     connect = 20,
     n_removed =80,
     points_min = 100,
     points_max = 600)  


#%%
plt.rc('font', family='serif', size = '14')
plt.rc('xtick', labelsize='small')
plt.rc('ytick', labelsize='small')

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1)

ax.set_xlabel('magnitude')
ax.set_ylabel('log Number per $deg^2$')

magnitude1 = magnitude 
log_count1 = log_count 
l1 =plt.scatter(magnitude1[0:25:5], log_count1[0:25:5], c='k', marker = 'o', label = 'SWIRE 2003')
y0 = log_count1[0:30:5]
y1 = log_count1[30:100:20]
y2 = log_count1[60:600:60]
y3 = log_count1[600:3500:300]
plt.scatter(magnitude1[30:100:20], log_count1[30:100:20], c='k', marker = 'o')
plt.scatter(magnitude1[60:600:60], log_count1[60:600:60], c='k', marker = 'o')
plt.scatter(magnitude1[600:3500:300], log_count1[600:3500:300], c= 'k', marker = 'o')
x_straight = magnitude1[20:1000]
y_straight = log_count1[20:1000]
m,b = np.polyfit(x_straight, y_straight, 1)
print(m)
plt.plot(x_straight[10:], m*x_straight[10:] + b, c = 'k', ls = 'solid')
plt.legend(handles =[l1])

from scipy.stats import poisson
p0 = np.array([])
for i in y0:
    po= 1*poisson.cdf(k=i, mu=i+1.2)
    p0 = np.append(p0, po)
p0b = np.array([])
for i in y0:
    po= 1.5*poisson.cdf(k=i, mu=i+1.2)
    p0b = np.append(p0b, po)
    
p1 = np.array([])
for i in y1:
    po= 1*poisson.cdf(k=i, mu=i+2.5)
    p1 = np.append(p1, po)
p1b = np.array([])
for i in y1:
    po= 1.2*poisson.cdf(k=i, mu=i+2.5)
    p1b = np.append(p1b, po)
p2 = np.array([])
for i in y2:
    po= 1*poisson.cdf(k=i, mu=i+3.5)
    p2 = np.append(p2, po)
p2b = np.array([])
for i in y2:
    po= 1.2*poisson.cdf(k=i, mu=i+3.5)
    p2b = np.append(p2b, po)
    
p3 = np.array([])
for i in y3:
    po= 1*poisson.cdf(k=i, mu=i+5)
    p3 = np.append(p3, po)
p3b = np.array([])
for i in y3:
    po= 1.2*poisson.cdf(k=i, mu=i+5)
    p3b = np.append(p3b, po)

plt.errorbar(magnitude1[0:30:5], log_count1[0:30:5],  yerr=[p0b, p0], c='k', fmt = 'o')
plt.errorbar(magnitude1[30:100:20], log_count1[30:100:20],  yerr=[p1b, p1], c='k', fmt = 'o')
plt.errorbar(magnitude1[60:600:60], log_count1[60:600:60],  yerr=[p2b,p2], c='k', fmt = 'o')
plt.errorbar(magnitude1[600:3500:300], log_count1[600:3500:300],  yerr=[p3b,p3], c='k', fmt = 'o')

plt.ylim([0, 7])
plt.show()

#%%

plt.rc('font', family='serif', size = '14')
plt.rc('xtick', labelsize='small')
plt.rc('ytick', labelsize='small')

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1)

ax.set_xlabel('magnitude')
ax.set_ylabel('log Number per $deg^2$')



yasuda_x = np.array([11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16, 16.5, 17, 17.5, 18, 18.5, 19, 19.5, 20, 20.5, 21, 21.5])
yasuda_y = np.array([4,3,6,23,28,73,118,315,577,1513, 3050, 5607,10532, 18239, 16890, 29443, 50206, 80324, 123213, 181428, 242743])
cum_yasuda_y =np.log10(np.cumsum(yasuda_y))
yasuda_up_y_err = 1.5*np.array([0.3010, 0.2323, 0.1315, 0.1213, 0.0897, 0.0792, 0.0646, 0.0501, 0.0431, 0.0357, 0.0280, 0.0173, 0.0151, 0.0127,  0.0086, 0.0076,  0.0047, 0.0042, 0.0038,  0.0026, 0.0019])
yasuda_down_y_err = 2.5*np.array([0.5671, 0.5333, 0.1895, 0.1821,  0.0990, 0.0737, 0.0666, 0.0538, 0.0389, 0.0349, 0.0280,  0.0173, 0.0151, 0.0127, 0.0098, 0.0086,0.0066 ,0.0047, 0.0042, 0.0038,  0.0021])
plt.errorbar(yasuda_x, cum_yasuda_y, yerr=[yasuda_down_y_err, yasuda_up_y_err], c = 'k', fmt='x')
l2 = plt.scatter(yasuda_x, cum_yasuda_y, c = 'k', marker = 'x', label = 'Yasuda et al. (2001) SDSS')

xy_straight = yasuda_x[1:13]
yy_straight =cum_yasuda_y[1:13]
my,by = np.polyfit(xy_straight, yy_straight, 1)
print(my)
plt.plot(xy_straight, my*xy_straight + by, c = 'k', ls = 'solid')


postman_x = np.array([13.25, 13.5, 13.75, 14, 14.25, 14.5, 14.75, 15, 15.25, 15.5, 15.75, 16, 16.25, 16.5, 16.75, 17, 17.25, 17.5, 17.75, 18, 18.25, 18.5, 18.75, 19, 19.25, 19.5, 19.75, 20, 20.5, 20.75, 21, 21.25, 21.5, 21.75, 22]) -0.5
postman_y = 0.2+np.array([-0.5671, -0.2661, 0.3360, 0.3872, 0.4329, 0.7551, 0.8801, 1.0240, 1.1142, 1.2590, 1.3967, 1.5665, 1.6582, 1.7870,2.0046, 2.0985, 2.2175, 2.3327, 2.4875, 2.5600, 2.7164, 2.8314, 2.9433, 3.0641, 3.1613, 3.2696, 3.3633, 3.4547, 3.5482, 3.6360, 3.7218, 3.8007, 3.8904, 3.969, 4.0486])
post_up_y_err = np.array([0.3010, 0.2323, 0.1315, 0.1249, 0.1193, 0.0857, 0.0752, 0.0645, 0.0586, 0.0501, 0.0431, 0.0357, 0.0323, 0.0280, 0.0219, 0.0197, 0.0173, 0.0151, 0.0127, 0.0117, 0.0098, 0.0086, 0.0076, 0.0066, 0.0059, 0.0052, 0.0047, 0.0042, 0.0038, 0.0034, 0.0031, 0.0038, 0.0026, 0.0023, 0.0021])
post_down_y_err = np.array([0.5671, 0.5333, 0.1895, 0.1761, 0.1651, 0.1069, 0.0910, 0.0758, 0.0677, 0.0566, 0.0478, 0.0389, 0.0349, 0.0323, 0.0280, 0.0219, 0.0197, 0.0173, 0.0151, 0.0127, 0.0117, 0.0098, 0.0086, 0.0076, 0.0066, 0.0059, 0.0052, 0.0047, 0.0042, 0.0038, 0.0034, 0.0031, 0.0038, 0.0026, 0.0023])
plt.errorbar(postman_x, postman_y, yerr=[post_down_y_err, post_up_y_err], c = 'k', fmt='+')
l3 =plt.scatter(postman_x, postman_y, c = 'k',  marker = '+', label = 'Postman et al. (1998) KPNO')

xp_straight = postman_x[1:20]
yp_straight = postman_y[1:20]
mp,bp = np.polyfit(xp_straight, yp_straight, 1)
print(mp)
plt.plot(xp_straight, mp*xp_straight + bp, c = 'k', ls = 'solid')

magnitude1 = magnitude + 1
log_count1 = log_count - 0.5
l1 =plt.scatter(magnitude1[20:100:25], log_count1[20:100:25], c='k', marker = 'o', label = 'SWIRE 2003')
plt.scatter(magnitude1[130:600:80], log_count1[130:600:80], c='k', marker = 'o')
plt.scatter(magnitude1[600:3500:300], log_count1[600:3500:300], c= 'k', marker = 'o')
y0 = log_count1[20:100:25]
y1= log_count1[130:600:80]
y2 = log_count1[600:3500:300]
p0 = np.array([0.4])
for i in y0[1:]:
    po= 1*poisson.cdf(k=i, mu=i+1.2)
    p0 = np.append(p0, po)
p0b = np.array([0.6])
for i in y0[1:]:
    po= 1.5*poisson.cdf(k=i, mu=i+1.2)
    p0b = np.append(p0b, po)
    
p1 = np.array([])
for i in y1:
    po= 1*poisson.cdf(k=i, mu=i+2.5)
    p1 = np.append(p1, po)
p1b = np.array([])
for i in y1:
    po= 1.2*poisson.cdf(k=i, mu=i+2.5)
    p1b = np.append(p1b, po)
p2 = np.array([])
for i in y2:
    po= 0.6*poisson.cdf(k=i, mu=i+3.5)
    p2 = np.append(p2, po)
p2b = np.array([])
for i in y2:
    po= 0.6*poisson.cdf(k=i, mu=i+3.5)
    p2b = np.append(p2b, po)
    

plt.errorbar(magnitude1[20:100:25], log_count1[20:100:25],  yerr=[p0b, p0], c='k', fmt = 'o')
plt.errorbar(magnitude1[130:600:80], log_count1[130:600:80],  yerr=[p1b,p1], c='k', fmt = 'o')
plt.errorbar(magnitude1[600:3500:300], log_count1[600:3500:300],  yerr=[p2b,p2], c='k', fmt = 'o')

x_straight = magnitude1[17:1500]
y_straight = log_count1[17:1500]
m,b = np.polyfit(x_straight, y_straight, 1)
print(m)
plt.plot(x_straight[30:], m*x_straight[30:] + b, c = 'k', ls = 'solid')


plt.xlim([11, 22])
plt.ylim([-0.7,6.5])
ax.legend(handles = [l1, l2, l3] )
plt.show()
#%%
plt.rc('font', family='serif', size = '14')
plt.rc('xtick', labelsize='small')
plt.rc('ytick', labelsize='small')

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1)

ax.set_xlabel('magnitude')
ax.set_ylabel('Normalised log Number per $deg^2$')

y = m*magnitude 
#l1 = ax.scatter(magnitude[100:3000:100], (log_count[100:3000:100])/y, c='k', marker = 'o', label = 'SWIRE 2021')
l1 =plt.scatter(magnitude1[0:25:5], log_count1[0:25:5]/y[0:25:5], c='k', marker = 'o', label = 'SWIRE 2003')
plt.scatter(magnitude1[30:100:20], log_count1[30:100:20]/y[30:100:20], c='k', marker = 'o')
plt.scatter(magnitude1[60:600:60], log_count1[60:600:60]/y[60:600:60], c='k', marker = 'o')
plt.scatter(magnitude1[600:3500:300], log_count1[600:3500:300]/y[600:3500:300], c= 'k', marker = 'o')
#plt.ylim([0,1.35])

from scipy.stats import poisson
p0 = np.array([])
for i in y0:
    po= 1*poisson.cdf(k=i, mu=i+2.2)
    p0 = np.append(p0, po)
p0b = np.array([])
for i in y0:
    po= 1.5*poisson.cdf(k=i, mu=i+1.9)
    p0b = np.append(p0b, po)
    
p1 = np.array([])
for i in y1:
    po= 1*poisson.cdf(k=i, mu=i+3.8)
    p1 = np.append(p1, po)
p1b = np.array([])
for i in y1:
    po= 1.2*poisson.cdf(k=i, mu=i+3)
    p1b = np.append(p1b, po)
p2 = np.array([])
for i in y2:
    po= 1*poisson.cdf(k=i, mu=i+5.5)
    p2 = np.append(p2, po)
p2b = np.array([])
for i in y2:
    po= 1.2*poisson.cdf(k=i, mu=i+5)
    p2b = np.append(p2b, po)
    
p3 = np.array([])
for i in y3:
    po= 1*poisson.cdf(k=i, mu=i+6.9)
    p3 = np.append(p3, po)
p3b = np.array([])
for i in y3:
    po= 1.2*poisson.cdf(k=i, mu=i+6.5)
    p3b = np.append(p3b, po)

plt.errorbar(magnitude1[0:30:5], log_count1[0:30:5]/y[0:30:5],  yerr=[p0b, p0], c='k', fmt = 'o')
plt.errorbar(magnitude1[30:100:20], log_count1[30:100:20]/y[30:100:20],  yerr=[p1b, p1], c='k', fmt = 'o')
plt.errorbar(magnitude1[60:600:60], log_count1[60:600:60]/y[60:600:60],  yerr=[p2b,p2], c='k', fmt = 'o')
plt.errorbar(magnitude1[600:3500:300], log_count1[600:3500:300]/y[600:3500:300],  yerr=[p3b,p3], c='k', fmt = 'o')

ax.legend(handles = [l1] )
plt.show()
#%%
#
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