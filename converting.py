import os

import convert_csv

# mac
# path = "/Users/pyojisung/Library/CloudStorage/GoogleDrive-gachon.mowa@gmail.com/내 드라이브/data/pcap"

# window
convert_csv.generate_csv("C:\\Users\\HOME\\Desktop\\데이터수집\\input_Walking_1"+".pcap",
                         "C:\\Users\\HOME\\Desktop\\데이터수집\\input_Walking_1.csv", 'amplitude')
