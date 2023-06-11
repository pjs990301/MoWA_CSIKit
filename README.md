# MoWA CSIKit
This repository contains Python commands that help make CSI easier to handle on Raspberry Pi.

## How to Use & When to Use
1. If you want to get CSI data from Raspberry Pi, run the following command
    <pre><code>python csiCapture.py</code></pre>
    <br>
2. If you use the Nexmon CSI tool while using Raspberry Pi 4B, there are many things to set up first. However, it can be easily set up through the following code execution.   
    <pre><code>python init.py</code></pre>
    <br>
3. If you need to filter Mac addresses, Change the MAC address after the first line.   
    <pre><code>encoded_parameter = os.popen("mcp -C 1 -N 1 -c 36/80 -m e4:5f:01:c4:c3:7e,88:36:6c:06:40:6a")</code></pre>
    `mcp` supports several other features like filtering data by Mac IDs or by FrameControl byte. 
   Run `mcp -h` to see all available options.        
    <br>
4. If you want to send the collected CSI data from Raspberry Pi to the server, there are two options.    
   - FTP
        <pre><code>python ftpTest.py</code></pre>
        <br>
   - RESTAPI POST method
        <pre><code>python csiCapture_live.py</code></pre>
        You can modify the URL variable and request it to that URL.    
   <br>
5. When the CSI data is extracted, the value of the data has the form of complex number data. But there are times when we want to check this visually.
    <pre><code>converting.py</code></pre>


## Reference
### csiread
The repository provides useful libraries for each CSI extraction tool.
You can use the library if you can easily install it.
<pre><code>pip3 install csiread</code></pre>
Please check the link below for details. 
[Link](https://github.com/citysu/csiread)


## License
Our repository worked on the MIT license.
Please refer to the following [Link](https://github.com/pjs990301/MoWA_CSIKit/blob/main/LICENSE) for more information

## Contact
- Pull Request
- p990301@gachon.ac.kr