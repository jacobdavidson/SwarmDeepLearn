"""
Author: Arthur Mateos
7/14/17
"""


import pandas as pd
import numpy as np
import os
import csv

def download_jolle_data(filename):
    """download the csv file stored at 'filename'.
    Drop unnecessary columns."""
    data = pd.read_csv(filename, sep=",", header=0)
    data = data.drop(['Color', 'col'], axis=1)
    return data

def main():
    # names of folders that hold csv's to convert
    data_folders = ['../data/fish/CM1FRE_150324_1147_RP10_S04_G22_P',
        '../data/fish/CM1FRE_150324_1227_RP09_S05_G05_P',
        '../data/fish/CM1FRE_150324_1227_RP10_S05_G20_P',
        '../data/fish/CM1FRE_150324_1307_RP09_S06_G12_P',
        '../data/fish/CM1FRE_150324_1307_RP10_S06_G18_P']

    # Loop through each folder
    for folder in data_folders:
        readfile = os.path.join(folder, 'original.csv')
        savefile = os.path.join(folder, 'pixel_pos.csv')

        data = download_jolle_data(readfile)

        data = np.array(data).T

        # 'Frame' column should have integer values
        data[0] = data[0].astype(int)

        # Relabel fish so each has unique positive integer id
        fish_ids = np.unique(data[1])
        new_fish_id = 1
        for fish_id in fish_ids:
            data[1][data[1]==fish_id] = new_fish_id
            new_fish_id += 1

        # Normalize x values
        data[2] = data[2]/data[2].max()
        # Normalize y values
        data[3] = data[3]/data[3].max()

        # Save newly formatted file
        with open(savefile, "w") as f:
            writer = csv.writer(f)
            writer.writerows(data)

if __name__ == '__main__':
    main()
