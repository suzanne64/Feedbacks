#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 13:16:45 2018a

@author: suzanne
"""
import os.path
import glob
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import Gsmooth_functions as sm
import time
#from mpl_toolkits.basemap import Basemap
# NEED TO get Cartopy working... see pip install Cartopy online.
import datetime as dt
from netCDF4 import Dataset

smoo_dir = '/Users/suzanne/luanne/cmip5/FiguresforPaper/py/smoothed/'
figs_dir = '/Users/suzanne/luanne/cmip5/FiguresforPaper/py/figs/'
data_dir = '/Users/suzanne/luanne/cmip5/FiguresforPaper/py/data/'

filein = smoo_dir + 'test_lh_sm.nc'
#dataset = Dataset(filein)
#print(dataset.dimensions.keys())
#print(dataset.variables.keys())
#print(dataset.variables['lh_sm'])

# get latent heat out, with OAflux grid
fid = Dataset(filein,'r')
latoa = fid.variables['lat'][:]
lonoa = fid.variables['lon'][:]
year = fid.variables['year'][:]
month = fid.variables['month'][:]
day = fid.variables['day'][:]
tt = [dt.date(yy,mm,dd) for yy, mm, dd in zip(year,month,day)]
lh = fid.variables['lh_sm'][:]
fid.close()

# get sea surface temperature (same grid as lh)
filein = smoo_dir + 'test_sst_sm.nc'
fid = Dataset(filein,'r')
sst = fid.variables['sst_sm'][:]
fid.close()

# get sea surface height out, with Aviso grid
filein = smoo_dir + 'test_sla_sm.nc'
fid = Dataset(filein,'r')
latav = fid.variables['lat'][:]
lonav = fid.variables['lon'][:]
ssh = fid.variables['ssh_sm'][:]
fid.close()

# detrend the data temporally

t0 = np.arange(24)
#plt.figure(1);plt.clf()
#plt.plot(lh[:,20,30],'b')
lh = np.array( [sm.detrendNaN(t0,lh[:,jj,ii]) for jj in np.arange(len(latoa)) for ii in np.arange(len(lonoa))]).reshape((latoa.shape[0],lonoa.shape[0],len(tt)))
lh = np.moveaxis(lh,-1,0)  # move last dim to front: now all-months, lat, lon
#plt.plot(lh[:,20,30],'r')

#plt.figure(2);plt.clf()
#plt.plot(sst[:,20,30],'b')
sst = np.array( [sm.detrendNaN(t0,sst[:,jj,ii]) for jj in np.arange(len(latoa)) for ii in np.arange(len(lonoa))]).reshape((latoa.shape[0],lonoa.shape[0],len(tt)))
sst = np.moveaxis(sst,-1,0)  
#plt.plot(sst[:,20,30],'r')

#plt.figure(3);plt.clf()
#plt.plot(ssh[:,np.abs(np.abs(latoa[20]-latav).argmin()),np.abs(np.abs(lonoa[30]-lonav).argmin())],'b')
ssh = np.array( [sm.detrendNaN(t0,ssh[:,jj,ii]) for jj in np.arange(len(latav)) for ii in np.arange(len(lonav))]).reshape((latav.shape[0],lonav.shape[0],len(tt)))
ssh = np.moveaxis(ssh,-1,0)  
#plt.plot(ssh[:,np.abs(np.abs(latoa[20]-latav).argmin()),np.abs(np.abs(lonoa[30]-lonav).argmin())],'g')

# interpolate ssh onto oaflux grid
ssh = np.array( [sm.interp2dNaN(lonav,latav,ssh[ii,:,:],lonoa,latoa) for ii in np.arange(len(tt))])
#print('closest',latav[np.abs(np.abs(latoa[20]-latav).argmin())],latoa[20])
#print('closest',lonav[np.abs(np.abs(lonoa[30]-lonav).argmin())],lonoa[30])
#plt.plot(ssh[:,20,30],'r')

# get ElNino indicies 
filenino = data_dir + 'MEI timeseries from Dec_Jan 1940_50 up to the present.html'
f = open(filenino,'r')
yrnino = np.array([])
inino = np.array([])

nd = False
for ii, line in enumerate(f):
    if nd:
        cols = line.split()
        yrnino = np.append(yrnino,np.asarray(cols[0],dtype=int))
        inino  = np.append(inino,np.asarray(cols[1:],dtype=float))
        if '2018' in cols[0]:
           break
    if 'YEAR' in line and 'DECJAN' in line:
        nd = True
        
inino.resize(inino.shape[0]//12+1,12)
inino[inino==0]=np.nan
print(yrnino)
# WE'LL INTERPOLATE THIS AND SUBTRACT FROM ___ LATER

# make lowpass filter
from scipy.signal import butter, lfilter, freqz   # freqs is for analog filter, freqz is for digital

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog = False)  # analog = False because we want digital filter (we have regularly sampled data)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=4):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

cutoff = 23.1 # cutoff frequency of the filter in Hz
fs = 30.0     # sampling frequency in Hz
order = 6     # order of filter

# Get the filter coefficients so we can check its frequency response.
b, a = butter_lowpass(cutoff, fs, order)

# Plot the frequency response.
w, h = freqz(b, a, worN=8000)
plt.figure(1);plt.clf()
plt.subplot(2, 1, 1)
plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
plt.axvline(cutoff, color='k')
plt.xlim(0, 0.5*fs)
plt.title("Lowpass Filter Frequency Response")
plt.xlabel('Frequency [Hz]')
plt.grid()

## Demonstrate the use of the filter.
## First make some data to be filtered.
#T = 5.0         # seconds
#n = int(T * fs) # total number of samples
#t = np.linspace(0, T, n, endpoint=False)
## "Noisy" data.  We want to recover the 1.2 Hz signal from this.
#data = np.sin(1.2*2*np.pi*t) + 1.5*np.cos(9*2*np.pi*t) + 0.5*np.sin(12.0*2*np.pi*t)
#
## Filter the data, and plot both the original and filtered signals.
#y = butter_lowpass_filter(data, cutoff, fs, order)
#
#plt.subplot(2, 1, 2)
#plt.plot(t, data, 'b-', label='data')
#plt.plot(t, y, 'g-', linewidth=2, label='filtered data')
#plt.xlabel('Time [sec]')
#plt.grid()
#plt.legend()
#
#plt.subplots_adjust(hspace=0.35)
#plt.show()


# rename
# read in files containing smoothed data
# rho,df,rho95 = sm.correlatem(xx,yy)

#lags, rho = sm.lagcor(x1,x2,12,1)

##smoo_dir = '/Users/suzanne/luanne/cmip5/FiguresforPaper/py/smoothed/'
