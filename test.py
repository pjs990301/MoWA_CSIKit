import os

import convert_csv

# mac
# path = "/Users/pyojisung/Library/CloudStorage/GoogleDrive-gachon.mowa@gmail.com/내 드라이브/data/pcap"

# window
path = "H:\\내 드라이브\\data\\pcap"
os.chdir(path)
for i in range(251, 400):
    file_name = "Lying_Ex_Home_" + str(i)
    convert_csv.generate_csv(file_name + ".pcap",
                             # "/Users/pyojisung/Library/CloudStorage/GoogleDrive-gachon.mowa@gmail.com/내
                             # 드라이브/data/csv/"+file_name + ".csv", 'amplitude')
                             "H:\\내 드라이브\\data\\csv\\" + file_name + ".csv", 'amplitude')
