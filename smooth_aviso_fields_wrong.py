#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 13:16:45 2018a

@author: suzanne
"""
import os.path
import glob
import numpy as np
import matplotlib.pyplot as plt
import Gsmooth_functions as sm
import time
#from mpl_toolkits.basemap import Basemap
# NEED TO get Cartopy working... see pip install Cartopy online.
import datetime as dt
from netCDF4 import Dataset

# establish where to get and put files
data_dir = '/Users/suzanne/luanne/cmip5/FiguresforPaper/py/data/'
figs_dir = '/Users/suzanne/luanne/cmip5/FiguresforPaper/py/figs/'
smoo_dir = '/Users/suzanne/luanne/cmip5/FiguresforPaper/py/smoothed/'

# user defined parameters, depending on where to smooth and how much smoothing (fwhm) to do.
class smooth_params:
    def __init__(self):
        # set domain for smoothing
        self.minlat = 20.
        self.maxlat = 60.
        self.minlon = -80.
        self.maxlon = -20.
        self.fwhm = 300.  # full-width, half-max for smoothing, in km
        self.lon_r = np.ceil(self.fwhm/110.) * 3;
        self.lat_r = np.ceil(self.fwhm/110.) * 3;
        self.sigma = 1000. * self.fwhm / 2.355
        self.ATL = 1      # 1 means Atlantic ocean, lons -180 to 180.

params = smooth_params()
# filename containing the smoothing weights
file_wts = data_dir + 'smooth_wts_avisogrid_fwhm' + str(np.int(params.fwhm)) + '.nc'

# bring in variable to smooth
filelist=sorted(glob.glob(data_dir + 'dt*.nc'))
month=np.array([])
year =np.array([])
day  =np.array([])
sla=np.array([])

for i,file in enumerate(filelist):
    nc = Dataset(file,mode='r')
#        print(nc.file_format)
#        print(nc.dimensions)  # also nc.dimensions.keys()
#        print(nc.variables)
#        print(nc.variables['time'].units)  
    # append time to mm variable (time is actually month of year)
    month = np.append(month, nc.variables['time'][:])
    year  = np.append(year, np.ones([12,1],dtype=np.int)*np.int(re.findall(r'\d+',file)[1]))
    day = np.append(day, np.ones([12,1])*15)
    if i==0:
        lat = nc.variables['lat'][:]
        lon = nc.variables['lon'][:]
        sla=np.squeeze(nc.variables['sla'][:])  # squeeze because was [1,720,1440]
    else:
        # append geophysical variable to sla variable
        sla = np.dstack((sla,np.squeeze(nc.variables['sla'][:]))) 
        
# make time iterable
tt = [dt.date(np.int(yy),np.int(mm),np.int(dd)) for yy,mm,dd in zip(year,month,day)]

# replace FillValue with nans            
np.place(sla,sla==nc.variables['sla']._FillValue,np.nan)

# change longitudes to -180 to 180 if working in Atlantic
if params.ATL == 1:
    np.place(lon,lon>180,lon[lon>180]-360)
    lon=np.roll(lon,len(lon)//2)                        
    sla=np.roll(sla,len(lon)//2,axis=1)  # sla dims: lat, lon, all-months

sla = np.moveaxis(sla,-1,0)  # move last dim to front: now all-months, lat, lon
# make land mask from the means of sla over time. Ocean = 1, Land = nan
mask = np.mean(sla,axis=0)
mask[~np.isnan(mask)]=1
plt.figure(2);plt.clf()
plt.imshow(mask);plt.colorbar()

levels = np.arange(-0.3,0.31,0.01)
plt.figure(1);plt.clf()
#plt.imshow(sla[0,:,:]);plt.colorbar()
plt.contourf(lon[np.logical_and(lon>=params.minlon,lon<=params.maxlon)],lat[np.logical_and(lat>=params.minlat,lat<=params.maxlat)],
             sla[:,np.logical_and(lat>=params.minlat,lat<=params.maxlat),:][:,:,np.logical_and(lon>=params.minlon,lon<=params.maxlon)][0,:,:],levels);plt.colorbar()
plt.title('Sea level anomaly')

# check if smoothing weights file already exists for particular grid and smoothing parameter 'fwhm'
if os.path.isfile(file_wts):
    print('smoothing weights file exists')
    nc = Dataset(file_wts,mode='r')
    class Object(object):
        pass
    Gauss = Object()        
    Gauss.wts = nc.variables['wts'][:]
    Gauss.lat_wts = nc.variables['lat_wts'][:]
    Gauss.lon_wts = nc.variables['lon_wts'][:]
    Gauss.lat_ctr = nc.variables['lat_ctr'][:]
    Gauss.lon_ctr = nc.variables['lon_ctr'][:]
    
else:
    print('make smoothing weights file!')
    #Gauss_wts, lat_wts, lon_wts, latc, lonc = sm.smoothing_weights(lat,lon,params).make_weights()
    Gauss = sm.smoothing_weights(lat,lon,params).make_weights()
    print('in fields.py',Gauss.lat_ctr.shape,Gauss.lon_ctr.shape)
    print('shapes',Gauss.lat_wts.shape,Gauss.lon_wts.shape,Gauss.wts.shape)
    
    filename = data_dir + 'smooth_wts_avisogrid_fwhm300.nc'
    print(filename)
    nc = Dataset(filename,'w',format='NETCDF4_CLASSIC')
    print(nc.file_format) # CLASSIC data: dims, variables, attributes.
    
    # create dimensions
    sm_width = nc.createDimension('smoothing_width',Gauss.lat_wts.shape[0]) # None means unlimitied 
    lat_center = nc.createDimension('lat_center',Gauss.wts.shape[2])
    lon_center = nc.createDimension('lon_center',1)
    print(nc.dimensions.keys())
    
    # create variables
    var=nc.createVariable('lat_ctr',float,('lat_center',))
    var[:]=Gauss.lat_ctr
    var=nc.createVariable('lon_ctr',float,('lon_center',))
    var[:]=Gauss.lon_ctr
    var=nc.createVariable('lat_wts',float,('smoothing_width','lat_center'),fill_value='NaN')
    var[:]=Gauss.lat_wts
    var=nc.createVariable('lon_wts',float,('smoothing_width',))
    var[:]=Gauss.lon_wts
    var=nc.createVariable('wts',float,('smoothing_width','smoothing_width','lat_center'),fill_value='NaN')
    var[:]=Gauss.wts
    nc.close()


# time the smoothing
t_start = time.perf_counter()

# smooth the monthly clim anomaly, geo, where geo = var minus monthly mean of var
sla_sm, lat_sm, lon_sm = sm.Gsmooth(sla,lat,lon,tt,mask,params,Gauss).smooth_it()  # Gsmooth doesn't use tt 
print('sla_sm size',sla_sm.shape)
fig = plt.figure(33);plt.clf()
levels = np.arange(-0.15,0.16,0.01)
plt.contourf(lon_sm,lat_sm,sla_sm[0,:,:],levels);plt.colorbar()
plt.title('smoothed AVISO, SSH anomaly minus clim means, Jan 2014')
fig.savefig('figs/smoothed_SSH.jpg')
t_stop = time.perf_counter()
print('time perf',(t_stop-t_start)/60)

## parse the tt vector ('iterable' in python parlance)
#year  = np.array([ (dt.datetime(1950,1,1) + dt.timedelta(days=np.asscalar(tt[i]))).year for i in range(len(tt))])
#month = np.array([ (dt.datetime(1950,1,1) + dt.timedelta(days=np.asscalar(tt[i]))).month for i in range(len(tt))])
#day   = np.array([ (dt.datetime(1950,1,1) + dt.timedelta(days=np.asscalar(tt[i]))).day for i in range(len(tt))])

# make netcdf file
filename = smoo_dir + 'test_sla_sm.nc'
print(filename)
nc = Dataset(filename,'w',format='NETCDF4_CLASSIC')
#print(nc.file_format) # CLASSIC data: dims, variables, attributes.

# create dimensions
latitude = nc.createDimension('latitude',lat_sm.shape[0]) # 'None' indicates unlimitied 
longitude = nc.createDimension('longitude',lon_sm.shape[0]) 
time = nc.createDimension('time',len(tt))
print(nc.dimensions.keys())

# create variables
var=nc.createVariable('lat',np.float32,('latitude',))  # variable id, type, dimension(s)
var[:]=lat_sm
var.units = 'degrees_North'
var=nc.createVariable('lon',np.float32,('longitude',))
var[:]=lon_sm
var.units = 'degrees_East'
var=nc.createVariable('year',np.int32,('time',))
var[:]=year
var=nc.createVariable('month',np.int32,('time',))
var[:]=month
var=nc.createVariable('day',np.int32,('time',))
var[:]=day

var=nc.createVariable('ssh_sm',np.float32,('time','latitude','longitude'))
var[:]=sla_sm
var.units = 'meters'
var.land_value = 'NaN'
var.grid = 'Aviso grid: 1/4 degree in lat and lon, centered on the eighth degree'

# global attribute
nc.fwhm = 'Full-width, half-max is ' + str(params.fwhm) + ' km'
nc.close()
