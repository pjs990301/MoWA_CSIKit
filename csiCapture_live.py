import sys
import os
import ftplib
import schedule
import time
from datetime import datetime

sys.path.append("/usr/lib/python3/dist-packages/")


def command():
    now = datetime.now()
    file_name = "input_" + now.strftime('%Y_%m_%d_%H_%M_%S') + ".pcap"
    cmd = "tcpdump -i wlan0 dst port 5500 -vv -w " + file_name + " -c 500"
    os.system(cmd)
    return file_name


def ftp(file_name):
    session = ftplib.FTP()
    session.connect('blue-sun.kro.kr', 9002)
    session.login("MoWA", "12345678")

    uploadfile = open(file_name, mode='rb')

    session.encoding = 'utf-8'
    session.storbinary('STOR ' + '/data/input/pcap/' + file_name, uploadfile)
    uploadfile.close()

    session.quit()
    print("complete")


def job():
    file_name = command()
    ftp(file_name)


schedule.every(1).minutes.do(job)

while True:
    schedule.run_pending()
    time.sleep(1)
