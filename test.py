import os

import convert_csv

path = "/Users/pyojisung/Library/CloudStorage/GoogleDrive-gachon.mowa@gmail.com/내 드라이브/data/pcap"
os.chdir(path)
for i in range(308, 1001):
    file_name = "Empty_Ex_Home_" + str(i)
    convert_csv.generate_csv(file_name + ".pcap",
                             "/Users/pyojisung/Library/CloudStorage/GoogleDrive-gachon.mowa@gmail.com/내 드라이브/data/csv/"+file_name + ".csv", 'amplitude')
