import os

for i in range(64, 102):
    command = "tcpdump -i wlan0 dst port 5500 -vv -w " + "Walking_Ex_Home_" + str(i) + ".pcap" + " -c 500"
    print(command)
    os.system(command)