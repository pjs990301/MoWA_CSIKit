import os
import ftplib
import schedule
import time
from datetime import datetime


def command():
    now = datetime.now()
    file_name = "input_" + now.strftime('%Y_%m_%d') + ".pcap"
    cmd = "tcpdump -i wlan0 dst port 5500 -vv -w " + file_name + " -c 500"
    os.system(cmd)
    ftp(file_name)


def ftp(file_name):
    session = ftplib.FTP()
    session.connect('blue-sun.kro.kr', 9002)  # 두 번째 인자는 port number
    session.login("MoWA", "12345678")  # FTP 서버에 접속

    uploadfile = open(file_name, mode='rb')  # 업로드할 파일 open

    session.encoding = 'utf-8'
    session.storbinary('STOR ' + '/data/input/input.pcap', uploadfile)  # 파일 업로드

    uploadfile.close()  # 파일 닫기

    session.quit()  # 서버 나가기
    print("전송 완료")


schedule.every(1).minutes.do(command())

while True:
    schedule.run_pending()
    time.sleep(1)
