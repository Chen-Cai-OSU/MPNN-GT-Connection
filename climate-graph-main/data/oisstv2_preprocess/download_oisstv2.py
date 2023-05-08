import os
import sys
from ftplib import FTP
from os.path import join

# ---------------------- Download OISSTv2 data from ftp2.psl.noaa.gov ############
ALL_YEARS = range(1982, 2023)
SMALL = range(2019, 2023)
# Edit here to change the years to download:
YEARS = ALL_YEARS
# Edit the following variables to change the download parameters
#  the data directory can be passed as an argument to the script:
#           python download_oisstv2.py /path/to/data/dir
if len(sys.argv) > 1:
    DATA_DIR = sys.argv[1]
else:
    DATA_DIR = "C:/Users/salvarc/data/oisstv2"
VARIABLE_KEY = 'sst.day.mean'

# Create the data directory if it does not exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Connect to ftp server
ftp = FTP("ftp2.psl.noaa.gov")
ftp.login()
ftp.cwd('Datasets/noaa.oisst.v2.highres')

# Get all files
files = ftp.nlst()
print([f for f in files if VARIABLE_KEY in f])
print(f"Downloading to {DATA_DIR}")
# Download files
for year in YEARS:
    file = f'{VARIABLE_KEY}.{year}.nc'
    # If file exists, skip
    if os.path.exists(join(DATA_DIR, file)):
        print(f'Skipping {file} since it already exists')
        continue
    print(f'Downloading {file}')
    ftp.retrbinary(f'RETR {file}', open(join(DATA_DIR, file), 'wb').write)

# Close connection
ftp.quit()
