import os

# 자동으로 1000번 반복 (Empty)
for i in range(1, 1000):
    command = "tcpdump -i wlan0 dst port 5500 -vv -w " + "Empty_Ex_Home_" + str(i) + ".pcap" + " -c 500"
    os.system(command)

# # stand
# command = "tcpdump -i wlan0 dst port 5500 -vv -w " + "Empty_Ex_Home_" + str(i) + ".pcap" + " -c 500"
# os.system(command)
