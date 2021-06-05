import datetime
import logging
import os
import subprocess
import threading
from time import sleep
import pynvml as nvml
from pynvml.smi import nvidia_smi
from bs4 import BeautifulSoup as bs

bash_command = "/bin/bash -c 'nvidia-smi --query -x'"
sp = subprocess.Popen(bash_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
out, err = sp.communicate()
soup = bs(out, "lxml")

proc_gpu_dict = {}
avail_gpus = []
all_gpus = []
for p in soup.find("nvidia_smi_log").find_all("gpu"):
    all_gpus.append(int(p.find("minor_number").get_text()))

for gp in all_gpus:
    proc_gpu_dict.setdefault(gp, [])

for p in soup.find("nvidia_smi_log").find_all("gpu"):
    for proc in p.find("processes").find_all("process_info"):
        print("local.py: Adding: {%s:%s}" % (p.find("minor_number").get_text(), proc.find("pid").get_text()) )
        proc_gpu_dict[int(p.find("minor_number").get_text())].append(proc.find("pid").get_text())

for x, y in proc_gpu_dict.items():
    if y == []:
        avail_gpus.append(x)
    
proc_gpu_dict = {}

for gp in all_gpus:
    proc_gpu_dict.setdefault(gp, [])

#if all of the gpus have at least one process running on them, allocate minimum memory usage gpu
if len(avail_gpus) == 0:
    for p in soup.find("nvidia_smi_log").find_all("gpu"):
        for proc in p.find("fb_memory_usage").find_all("used"):
            print("local.py: Adding: {%s:%s}" % (p.find("minor_number").get_text(), proc.get_text()) )
            proc_gpu_dict[int(p.find("minor_number").get_text())].append(proc.get_text())

    avail_gpus.append(min(proc_gpu_dict.items(), key=lambda x: x[1])[0])

print("local.py: AVAIL GPUS: %s" % avail_gpus)
print("local.py: ALL GPUS: %s" % all_gpus)
print("proc_gpu_dict:")
print(proc_gpu_dict)

# print( avail_gpus, all_gpus)


