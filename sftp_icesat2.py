#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 10:14:47 2019

@author: suzanne
"""

import pysftp  #ftplib
import numpy as np
import os

# get list of file names in rel205_path
rel205_path='/Volumes/ice2/suzanne/ATL03_rel205/GreenlandSea/'
filenames = [f for f in os.listdir(rel205_path) if os.path.isfile(os.path.join(rel205_path, f))]
print(filenames)
# set strings for path composition
prod = 'ATL03/'
rel = 'rel001/'
fpath = '/data1/' +prod + rel
print(fpath)

# domain name 
try:
    srv = pysftp.Connection(host='scf.gsfc.nasa.gov',username='sdickin1',private_key='/Users/suzanne/.ssh/id_rsa')
#    print(srv.listdir())
    srv.chdir(fpath)
#    print(srv.pwd)
#    print(srv.listdir())
    srv.close()
except Exception as e:
    print('darn', e)

#for i in data:
#    print(i)
#ftp = ftplib.FTP('my.cmems-du.eu')
#ftp.login('sdickinson1','cloudy66,')  
#path = 'Core/SEALEVEL_GLO_PHY_L4_REP_OBSERVATIONS_008_047/' #dataset-duacs-rep-global-merged-allsat-phy-l4'
#ftp.cwd(path)
#ftp.retrlines('LIST')
#path = 'Core/SEALEVEL_GLO_PHY_L4_REP_OBSERVATIONS_008_047/dataset-duacs-rep-global-merged-allsat-phy-l4'
#ftp.cwd(path)
#print(ftp.pwd())
#
##for y in year:
##    ftp.cwd(str(y))
##    dir_list = ftp.nlst()
##    print(dir_list)
##    for f in dir_list:
##        print(f)
##        local_filename = os.path.join('data/', f)
##        file = open(local_filename, 'wb')
##        ftp.retrbinary('RETR '+ f, file.write)
##
##    file.close()
#
#ftp.quit() # This is the “polite” way to close a connection  

