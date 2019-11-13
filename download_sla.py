#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 10:14:47 2019

@author: suzanne
"""

import ftplib
import numpy as np
import os

# set years for download
year=np.arange(2006,2007)

# domain name 
ftp = ftplib.FTP('my.cmems-du.eu')
ftp.login('sdickinson1','cloudy66,')  
path = 'Core/SEALEVEL_GLO_PHY_L4_REP_OBSERVATIONS_008_047/' #dataset-duacs-rep-global-merged-allsat-phy-l4'
ftp.cwd(path)
ftp.retrlines('LIST')
path = 'Core/SEALEVEL_GLO_PHY_L4_REP_OBSERVATIONS_008_047/dataset-duacs-rep-global-merged-allsat-phy-l4'
ftp.cwd(path)
print(ftp.pwd())

#for y in year:
#    ftp.cwd(str(y))
#    dir_list = ftp.nlst()
#    print(dir_list)
#    for f in dir_list:
#        print(f)
#        local_filename = os.path.join('data/', f)
#        file = open(local_filename, 'wb')
#        ftp.retrbinary('RETR '+ f, file.write)
#
#    file.close()

ftp.quit() # This is the “polite” way to close a connection  

