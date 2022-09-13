import os

encoded_parameter = os.popen("mcp -C 1 -N 1 -c 36/80 -m e4:5f:01:c4:c3:7e,88:36:6c:06:40:6a")
os.system("pkill wpa_supplicant")
os.system("ifconfig wlan0 up")
os.system("nexutil -Iwlan0 -s500 -b -l34 -v" + encoded_parameter.read())
os.system("iw dev wlan0 interface add mon0 type monitor")
os.system("ip link set mon0 up")
