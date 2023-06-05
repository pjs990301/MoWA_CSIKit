# -*- coding: utf-8 -*-

import sys
sys.path.append("/usr/lib/python3/dist-packages/")

import os
import ftplib
import schedule 
import time
from datetime import datetime
import requests

def command():
    now = datetime.now()
    file_name = "input_" + now.strftime('%Y_%m_%d_%H_%M_%S') + ".pcap"
    cmd = "tcpdump -i wlan0 dst port 5500 -vv -w " + file_name + " -c 500"
    os.system(cmd)
    return file_name

def upload_file(file_name):
    url = "http://3.37.161.170:8000/pi/CSI"  # 요청을 보낼 URL

    with open(file_name, 'rb') as file:
        files = {'file': file}
        response = requests.post(url, files=files)

    if response.status_code == 200:
        print('파일 업로드 성공')
    else:
        print('파일 업로드 실패')

def job():
    file_name = command()
    upload_file(file_name)

while True:
    schedule.run_pending()
    time.sleep(1)
    schedule.every(15).seconds.do(job)

