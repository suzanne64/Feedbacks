#  Suzanne Dickinson, Jan 2018
#   Download WaveWatch III SWH and wind fields for use with python 3.


import numpy as np
from ftplib import FTP
import os

# set years for download
year=np.arange(2006,2019)
#year=np.arange(2018,2019) # thru Nov 2018
month=np.arange(1,13)

# domain name 
ftp = FTP('polar.ncep.noaa.gov')
ftp.login()  # don't need user or passwd for anonymous login
ftp.retrlines('LIST')
print(ftp.pwd())

for y in year:
    for m in month:
        dirname = '/history/waves/multi_1/%d%.2d/gribs' % (y,m)
        ftp.cwd(dirname)
        #ftp.retrlines('LIST')
        filehs = 'multi_1.glo_30m.hs.%d%.2d.grb2' % (y,m)
        ftp.retrbinary('RETR %s' % filehs, open(filehs,'wb').write)
        filewind = 'multi_1.glo_30m.wind.%d%.2d.grb2' % (y,m)
        ftp.retrbinary('RETR %s' % filewind, open(filewind,'wb').write)
        fileptp = 'multi_1.glo_30m.ptp.%d%.2d.grb2' % (y,m)
        try:
            with open(fileptp, "wb") as f:            
                ftp.retrbinary('RETR %s' % fileptp, f.write)  # ptp starts feb 2017, if no file we still download empty file? 
        except:
            print(fileptp,'not available')
            os.unlink(fileptp)
        filetp = 'multi_1.glo_30m.tp.%d%.2d.grb2' % (y,m)
        ftp.retrbinary('RETR %s' % filetp, open(filetp,'wb').write)

ftp.quit()




