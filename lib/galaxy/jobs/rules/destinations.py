from galaxy.jobs import JobDestination
from galaxy.jobs.mapper import JobMappingException
import logging
import os
import subprocess

gpu_flag = 0
bash_command = "/bin/bash -c 'nvidia-smi'"
sp = subprocess.Popen(bash_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
out, err = sp.communicate()
command_not_found = 'command not found'
if command_not_found.encode() not in out:
    gpu_flag = 1
    import pynvml as nvml

log = logging.getLogger(__name__)

if gpu_flag == 1:
    nvml.nvmlInit()
    gpu_count = nvml.nvmlDeviceGetCount()

def dynamic_fun(tool,job):
    flag = 0
    os.path.join('GALAXY_GPU_ENABLED', 'false')
    os.environ['GALAXY_GPU_ENABLED'] = "false"

    if tool:
        reqmnts = tool.requirements
        for req in reqmnts:
            if req.type == "compute" and req.name == "gpu":
                flag = 1
        if gpu_flag == 1 and gpu_count > 0 and flag == 1:
            os.environ['GALAXY_GPU_ENABLED'] = "true"
            return "local_gpu"
        else:
            os.environ['GALAXY_GPU_ENABLED'] = "false"
            return "local_cpu"
    else:
        return "local_cpu"