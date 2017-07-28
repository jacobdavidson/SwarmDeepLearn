"""
Author: Arthur Mateos
14 July 2017
"""

import pandas as pd
import numpy as np
import os
import csv

def download_jolle_data(filename):
    """download the csv file stored at 'filename'.
    Drop unnecessary columns."""
    data = pd.read_csv(filename, sep=",", header=0, index_col=4)  # 'frame' is the 5th column in the csv
    data = data[['x', 'y']]
    return data

def main():
    data_folders = ['../data/fish/170202_pi11_SOLI_S17_F175_TR',
        '../data/fish/170202_pi11_SOLI_S17_F175_TR',
        '../data/fish/170202_pi11_SOLI_S25_F053_TR',
        '../data/fish/170202_pi11_SOLI_S18_F078_TR',
        '../data/fish/170202_pi11_SOLI_S26_F200_TR',
        '../data/fish/170202_pi11_SOLI_S20_F144_TR',
        '../data/fish/170202_pi11_SOLI_S27_F020_TR',
        '../data/fish/170202_pi11_SOLI_S24_F189_TR',
        '../data/fish/170202_pi11_SOLI_S28_F028_TR']

    for folder in data_folders:
        readfile = os.path.join(folder, 'original.csv')
        savefile = os.path.join(folder, 'pixel_pos.csv')

        data = download_jolle_data(readfile)
        data['frame'] = data.index.values
        data['fishid'] = 1

        # rearrange columns
        data = data[['frame', 'fishid', 'x', 'y']]

        data = np.array(data).T
        data = [row for row in data]
        data[0] = data[0].astype(int)
        data[1] = data[1].astype(int)
        data[2] = data[2]/data[2].max()
        data[3] = data[3]/data[3].max()

        with open(savefile, "w") as f:
            writer = csv.writer(f)
            writer.writerows(data)

if __name__ == '__main__':
    main()
