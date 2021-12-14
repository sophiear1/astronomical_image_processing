# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 10:29:18 2021

@author: sar219
"""
import numpy as np
import  scipy as sp
import matplotlib.pyplot as plt
import scipy.optimize as op
import numpy.ma as ma

from astropy.stats import sigma_clipped_stats
from astropy.io import fits
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization import simple_norm

from photutils.aperture import CircularAperture
from photutils.detection import DAOStarFinder
from photutils.detection import find_peaks

def gauss(x, a, x0, sigma,C):
    return a*np.exp(-(x-x0)**2/(2*sigma**2)) + C

hdulist = fits.open("H:\Documents\Labs\Year_3\Astro_Imaging\A1_mosaic\A1_mosaic.fits")
header = hdulist[0].header
data = hdulist[0].data  

mean, median, std = sigma_clipped_stats(data, sigma=3.0)
daofind = DAOStarFinder(fwhm=5.0, threshold=10. * std)
sources = daofind(data - median)
positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
apertures = CircularAperture(positions, r=4.)
norm = ImageNormalize(stretch=SqrtStretch())
plt.imshow(data, cmap='Greys', origin='lower', norm=norm,
           interpolation='nearest')
apertures.plot(color='blue', lw=1.5, alpha=0.5)
#%%
mean, median, std = sigma_clipped_stats(data, sigma=3.0)
threshold = median + (5.0 * std)
tbl = find_peaks(data, threshold, box_size=11)

positions = np.transpose((tbl['x_peak'], tbl['y_peak']))
apertures = CircularAperture(positions, r=5.)
norm = simple_norm(data, 'sqrt', percent=99.9)
#plt.imshow(data, cmap='Greys_r', origin='lower', norm=norm,
           #interpolation='nearest')
apertures.plot(color='#0547f9', lw=1.5)
plt.xlim(0, data.shape[1] - 1)
plt.ylim(0, data.shape[0] - 1)
#%%
from photutils.segmentation import make_source_mask
mask = make_source_mask(data, nsigma=2, npixels=5, dilate_size=11)
mean, median, std = sigma_clipped_stats(data, sigma=3.0, mask=mask)
print((mean, median, std)) 

#%%
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
data = hdulist[0].data
sigma_clip = SigmaClip(sigma=3.)
bkg_estimator = MedianBackground()
bkg = Background2D(data, (10, 10), filter_size=(100, 100),
                    sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
hdu = fits.PrimaryHDU(data-bkg.background)
hdul = fits.HDUList([hdu])
hdul.writeto('new5.fits')
#%%
print(bkg.background_median) 
print(bkg.background_rms_median) 
plt.imshow(data)
plt.show() 
plt.imshow(bkg.background, origin='lower', cmap='Greys_r',
            interpolation='nearest')
plt.show()
norm = ImageNormalize(stretch=SqrtStretch())
plt.imshow(data - bkg.background, norm=norm, origin='lower',
            cmap='Greys_r', interpolation='nearest')
plt.show()
#%%
from photutils.psf import IterativelySubtractedPSFPhotometry
my_photometry = IterativelySubtractedPSFPhotometry( 
    finder=my_finder, group_maker=my_group_maker,
    bkg_estimator=my_bkg_estimator, psf_model=my_psf_model,
    fitter=my_fitter, niters=3, fitshape=(7, 7)
    # get photometry results
photometry_results = my_photometry(image=my_image)
# get residual image
residual_image = my_photometry.get_residual_image()