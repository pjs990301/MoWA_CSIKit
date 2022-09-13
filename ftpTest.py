import ftplib

session = ftplib.FTP()
session.connect('blue-sun.kro.kr', 9002)  # 두 번째 인자는 port number
session.login("MoWA", "12345678")  # FTP 서버에 접속

uploadfile = open('input.pcap', mode='rb')  # 업로드할 파일 open

session.encoding = 'utf-8'
session.storbinary('STOR ' + '/data/input/input.pcap', uploadfile)  # 파일 업로드

uploadfile.close()  # 파일 닫기

session.quit()  # 서버 나가기
print('파일전송함')