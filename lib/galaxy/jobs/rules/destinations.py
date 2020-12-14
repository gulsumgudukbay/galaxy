import logging
import os
import subprocess

from pynvml.smi import nvidia_smi
from bs4 import BeautifulSoup as bs

log = logging.getLogger(__name__)

gpu_flag = 0
bash_command = "/bin/bash -c 'nvidia-smi'"
sp = subprocess.Popen(bash_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
out, err = sp.communicate()
command_not_found = 'command not found'
if command_not_found.encode() not in out and command_not_found.encode() not in err:
    log.debug("***************out: %s, err: %s, command_not_found.encode: %s ************COMMAND NOT FOUND NOT IN OUT" % (out, err, command_not_found.encode()))
    gpu_flag = 1
    import pynvml as nvml


if gpu_flag == 1:
    nvml.nvmlInit()
    gpu_count = nvml.nvmlDeviceGetCount()

def get_gpu_usage(gpu_id):
    bash_command = "/bin/bash -c 'nvidia-smi --query -x'"
    sp = subprocess.Popen(bash_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = sp.communicate()
    soup = bs(out, "lxml")

    proc_gpu_dict = {}
    avail_gpus = []
    all_gpus = []
    print("PROCESSES IN GPU")
    for p in soup.find("nvidia_smi_log").find_all("gpu"):
        proc_gpu_dict.setdefault((p.find("minor_number").get_text()), []).append("")
        for proc in p.find("processes").find_all("process_info"):
            print("Adding: {%s:%s}" % (p.find("minor_number").get_text(), proc.find("pid").get_text()) )
            proc_gpu_dict.setdefault((p.find("minor_number").get_text()), []).append(proc.find("pid").get_text())

    for x, y in proc_gpu_dict.items():
        all_gpus.append(x)
        print(y)
        if y == [] or not y or y == ['']:
            avail_gpus.append(x)

    print("AVAIL GPUS: %s" % avail_gpus)
    print("ALL GPUS: %s" % all_gpus)

    return avail_gpus, all_gpus
    # return proc_gpu_dict.get(gpu_id), avail_gpus, all_gpus


def dynamic_fun(tool, job):
    flag = 0
    os.path.join('GALAXY_GPU_ENABLED', 'false')
    os.environ['GALAXY_GPU_ENABLED'] = "false"
    gpu_id_to_query = ""
    gpu_dev_to_exec = ""
    all_gps = []
    
    avail_gps = []
    if tool:
        reqmnts = tool.requirements
        for req in reqmnts:
            if req.type == "compute" and req.name == "gpu":
                if req.version and req.version != "":
                    gpu_id_to_query = req.version
                flag = 1
        if gpu_flag == 1 and gpu_count > 0 and flag == 1:
            os.environ['GALAXY_GPU_ENABLED'] = "true"
            
            if gpu_id_to_query != "":
                avail_gps, all_gps = get_gpu_usage(gpu_id_to_query)
                for dev in all_gps:
                    gpu_dev_to_exec += dev
                    if dev != all_gps[-1]: # if not last dev insert ','
                        gpu_dev_to_exec += ","

                if gpu_id_to_query in avail_gps:
                    gpu_dev_to_exec = gpu_id_to_query

                print(gpu_dev_to_exec)
                os.environ['CUDA_VISIBLE_DEVICES'] = gpu_dev_to_exec
                print("CUDA_VISIBLE_DEVICES: %s" % os.environ['CUDA_VISIBLE_DEVICES'])
                

            return "local_gpu"
        else:
            os.environ['GALAXY_GPU_ENABLED'] = "false"
            return "local_cpu"
    else:
        return "local_cpu"
