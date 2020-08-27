"""
Job runner plugin for executing jobs on the local system via the command line.
"""
import datetime
import logging
import os
import subprocess
import tempfile
import threading
from time import sleep

from galaxy import model
from galaxy.job_execution.output_collect import default_exit_code_file
from galaxy.util import (
    asbool,
)
from . import (
    BaseJobRunner,
    JobState
)
from .util.process_groups import (
    check_pg,
    kill_pg
)

import sched, time
import sys, signal
import csv
import shutil

import pynvml as nvml

log = logging.getLogger(__name__)
nvml.nvmlInit()
gpu_count = nvml.nvmlDeviceGetCount()

__all__ = ('LocalJobRunner', )

DEFAULT_POOL_SLEEP_TIME = 1
# TODO: Set to false and just get rid of this option. It would simplify this
# class nicely. -John
DEFAULT_EMBED_METADATA_IN_JOB = True


class LocalJobRunner(BaseJobRunner):
    """
    Job runner backed by a finite pool of worker threads. FIFO scheduling
    """
    runner_name = "LocalRunner"

    def __init__(self, app, nworkers):
        """Start the job runner """

        # create a local copy of os.environ to use as env for subprocess.Popen
        self._environ = os.environ.copy()
        self._proc_lock = threading.Lock()
        self._procs = []

        # Set TEMP if a valid temp value is not already set
        if not ('TMPDIR' in self._environ or 'TEMP' in self._environ or 'TMP' in self._environ):
            self._environ['TEMP'] = os.path.abspath(tempfile.gettempdir())

        super().__init__(app, nworkers)
        self._init_worker_threads()

    def __command_line(self, job_wrapper):
        """
        """
        command_line = job_wrapper.runner_command_line
        if job_wrapper.tool:
            flag = 0
            reqmnts = job_wrapper.tool.requirements
            for req in reqmnts:
                if req.type == "compute" and req.name == "gpu":
                    flag = 1
            if gpu_count > 0 and flag == 1:
                # log.info("**************************CL  GPU ENABLED!!!!!**********************************************")
                os.environ['GALAXY_GPU_ENABLED'] = "true"
            else:
                # log.info("**************************CL  GPU DISABLED!!!!!*********************************************")
                os.environ['GALAXY_GPU_ENABLED'] = "false"

        # slots would be cleaner name, but don't want deployers to see examples and think it
        # is going to work with other job runners.
        slots = job_wrapper.job_destination.params.get("local_slots", None) or os.environ.get("GALAXY_SLOTS", None)
        if slots:
            slots_statement = 'GALAXY_SLOTS="%d"; export GALAXY_SLOTS; GALAXY_SLOTS_CONFIGURED="1"; export GALAXY_SLOTS_CONFIGURED; GALAXY_GPU_ENABLED="%s"; export GALAXY_GPU_ENABLED;' % (int(slots)) % os.environ.get("GALAXY_GPU_ENABLED")
        else:
            slots_statement = 'GALAXY_SLOTS="1"; export GALAXY_SLOTS; GALAXY_GPU_ENABLED="%s"; export GALAXY_GPU_ENABLED;' % os.environ.get("GALAXY_GPU_ENABLED")

        job_id = job_wrapper.get_id_tag()
        job_file = JobState.default_job_file(job_wrapper.working_directory, job_id)
        exit_code_path = default_exit_code_file(job_wrapper.working_directory, job_id)
        job_script_props = {
            'slots_statement': slots_statement,
            'command': command_line,
            'exit_code_path': exit_code_path,
            'working_directory': job_wrapper.working_directory,
            'shell': job_wrapper.shell,
        }
        job_file_contents = self.get_job_file(job_wrapper, **job_script_props)
        self.write_executable_script(job_file, job_file_contents)
        return job_file, exit_code_path

    def post_process(self, log_file_path, util_path, util_mem_path, mem_free_path, mem_used_path, pcie_link_gen_cur_path, stats_path):
        # print("\nResults:")
        File_object = open(log_file_path,"r+")
        Lines = File_object.readlines() 
        File_object.close()
        util = []
        util_mem = []
        mem_total = 0
        mem_free = []
        mem_used = []
        pcie_link_gen_max = 0
        pcie_link_gen_cur = []

        # lines2 = [v for i, v in enumerate(Lines) if i % 2 == 1]
        lines2 = [v for i, v in enumerate(Lines) if i % 1 == 0 and i != 0]

        for lin in lines2:
            lin_list = lin.split(",")

            util.append(float(''.join(filter(str.isdigit, lin_list[0]))))
            util_mem.append(float(''.join(filter(str.isdigit, lin_list[1]))))
            mem_total = float(''.join(filter(str.isdigit, lin_list[2])))
            mem_free.append(float(''.join(filter(str.isdigit, lin_list[3]))))
            mem_used.append(float(''.join(filter(str.isdigit, lin_list[4]))))
            pcie_link_gen_max = float(''.join(filter(str.isdigit, lin_list[5])))
            pcie_link_gen_cur.append(float(''.join(filter(str.isdigit, lin_list[6]))))

        # print("Utilization Percentage: Min:%.3f, Max:%.3f, Mean:%.3f" % (min(util) , max(util) , (sum(util)/len(util))))
        # print("%s" % util)
        # print("-------------------------------------------------------------------------------------------------------------------")

        with open(util_path, 'w') as util_file:
            wr = csv.writer(util_file, quoting=csv.QUOTE_ALL)
            wr.writerow(util)


        # print("Memory Utilization Percentage: Min:%.3f, Max:%.3f, Mean:%.3f" % (min(util_mem) , max(util_mem) , (sum(util_mem)/len(util_mem))))
        # print("%s" % util_mem)
        # print("-------------------------------------------------------------------------------------------------------------------")

        with open(util_mem_path, 'w') as util_mem_file:
            wr = csv.writer(util_mem_file, quoting=csv.QUOTE_ALL)
            wr.writerow(util_mem)

        # print("Total Memory [MiB]:")
        # print("%s" % mem_total)
        # print("-------------------------------------------------------------------------------------------------------------------")

        # print("Free Memory [MiB]: Min:%.3f, Max:%.3f, Mean:%.3f" % (min(mem_free) , max(mem_free) , (sum(mem_free)/len(mem_free))))
        # print("%s" % mem_free)
        # print("-------------------------------------------------------------------------------------------------------------------")

        with open(mem_free_path, 'w') as mem_free_file:
            wr = csv.writer(mem_free_file, quoting=csv.QUOTE_ALL)
            wr.writerow(mem_free)

        # print("Used Memory [MiB]: Min:%.3f, Max:%.3f, Mean:%.3f" % (min(mem_used) , max(mem_used) , (sum(mem_used)/len(mem_used))))
        # print("%s" % mem_used)
        # print("-------------------------------------------------------------------------------------------------------------------")

        with open(mem_used_path, 'w') as mem_used_file:
            wr = csv.writer(mem_used_file, quoting=csv.QUOTE_ALL)
            wr.writerow(mem_used)

        # print("PCIe Link Gen Max:")
        # print("%s" % pcie_link_gen_max)
        # print("-------------------------------------------------------------------------------------------------------------------")    
        
        # print("PCIe Link Gen Current: Min:%.3f, Max:%.3f, Mean:%.3f" % (min(pcie_link_gen_cur) , max(pcie_link_gen_cur) , (sum(pcie_link_gen_cur)/len(pcie_link_gen_cur))))
        # print("%s" % pcie_link_gen_cur)
        # print("-------------------------------------------------------------------------------------------------------------------")

        with open(pcie_link_gen_cur_path, 'w') as pcie_link_gen_cur_file:
            wr = csv.writer(pcie_link_gen_cur_file, quoting=csv.QUOTE_ALL)
            wr.writerow(pcie_link_gen_cur)
        
        
        stats_fo = open(stats_path,"w+")
        stats_out_str = "Utilization Percentage: Min:%.3f, Max:%.3f, Mean:%.3f\n" \
                        "Memory Utilization Percentage: Min:%.3f, Max:%.3f, Mean:%.3f\n" \
                        "Total Memory [MiB]: %s\n" \
                        "Free Memory [MiB]: Min:%.3f, Max:%.3f, Mean:%.3f\n" \
                        "Used Memory [MiB]: Min:%.3f, Max:%.3f, Mean:%.3f\n" % (min(util), max(util), (sum(util)/len(util)), 
                        min(util_mem), max(util_mem), (sum(util_mem)/len(util_mem)),
                        mem_total,
                        min(mem_free), max(mem_free), (sum(mem_free)/len(mem_free)),
                        min(mem_used), max(mem_used), (sum(mem_used)/len(mem_used)))
        stats_fo.write(stats_out_str)

    def queue_job(self, job_wrapper):
        if not self._prepare_job_local(job_wrapper):
            return

        stderr = stdout = ''

        # command line has been added to the wrapper by prepare_job()
        command_line, exit_code_path = self.__command_line(job_wrapper)
        job_id = job_wrapper.get_id_tag()
        tool_name = "%s_%s" % (job_wrapper.tool.id, job_id)

        try:
            stdout_file = tempfile.NamedTemporaryFile(mode='wb+', suffix='_stdout', dir=job_wrapper.working_directory)
            stderr_file = tempfile.NamedTemporaryFile(mode='wb+', suffix='_stderr', dir=job_wrapper.working_directory)
            log.debug('({}) executing job script: {}'.format(job_id, command_line))
            
            #gpu_stats
            if os.environ['GALAXY_GPU_ENABLED'] == "true":
                # log.info("**************************SUBMIT  GPU ENABLED!!!!!**********************************************")
                directory = "gpu_util_%s" % tool_name
                parent_dir = os.getcwd()
                path = os.path.join(parent_dir, directory) 
                log_file_path = "gpu_util_%s.log" % tool_name
                log_file_path = os.path.join(path, log_file_path)
                util_path = os.path.join(path, "util.csv")
                util_mem_path = os.path.join(path, "util_mem.csv")
                mem_free_path = os.path.join(path, "mem_free.csv")
                mem_used_path = os.path.join(path, "mem_used.csv")
                pcie_link_gen_cur_path = os.path.join(path, "pcie_link_gen_cur.csv")
                stats_path = os.path.join(path, "stats_%s" % tool_name)

                if os.path.exists(path):
                    shutil.rmtree(path)

                os.mkdir(path)
                if os.path.exists(log_file_path):
                    os.remove(log_file_path)
                if os.path.exists(util_path):
                    os.remove(util_path)
                if os.path.exists(util_mem_path):
                    os.remove(util_mem_path)
                if os.path.exists(mem_free_path):
                    os.remove(mem_free_path)
                if os.path.exists(mem_used_path):
                    os.remove(mem_used_path)
                if os.path.exists(pcie_link_gen_cur_path):
                    os.remove(pcie_link_gen_cur_path)
                if os.path.exists(stats_path):
                    os.remove(stats_path)

                File_object = open(log_file_path,"a+")
                bash_command = "/bin/bash -c 'nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,pcie.link.gen.max,pcie.link.gen.current --format=csv -l 1'"
                sp = subprocess.Popen(bash_command, shell=True, stdout=File_object, stderr=subprocess.PIPE).pid
                # sp_pid = sp.pid
                # out_str = sp.communicate()
                # print("%s" % out_str[0])
                # File_object.write(out_str[0].__str__())
                File_object.close()

            # The preexec_fn argument of Popen() is used to call os.setpgrp() in
            # the child process just before the child is executed. This will set
            # the PGID of the child process to its PID (i.e. ensures that it is
            # the root of its own process group instead of Galaxy's one).
            proc = subprocess.Popen(args=command_line,
                                    shell=True,
                                    cwd=job_wrapper.working_directory,
                                    stdout=stdout_file,
                                    stderr=stderr_file,
                                    env=self._environ,
                                    preexec_fn=os.setpgrp)

            proc.terminated_by_shutdown = False
            with self._proc_lock:
                self._procs.append(proc)

            try:
                job = job_wrapper.get_job()
                # Flush job with change_state.
                job_wrapper.set_external_id(proc.pid, job=job, flush=False)
                job_wrapper.change_state(model.Job.states.RUNNING, job=job)
                self._handle_container(job_wrapper, proc)

                terminated = self.__poll_if_needed(proc, job_wrapper, job_id)
                proc.wait()  # reap
                if terminated:
                    if os.environ['GALAXY_GPU_ENABLED'] == "true":
                        os.kill(sp,9)
                        self.post_process(log_file_path, util_path, util_mem_path, mem_free_path, mem_used_path, pcie_link_gen_cur_path, stats_path)
                    return
                elif check_pg(proc.pid):
                    if os.environ['GALAXY_GPU_ENABLED'] == "true":
                        os.kill(sp,9)
                        self.post_process(log_file_path, util_path, util_mem_path, mem_free_path, mem_used_path, pcie_link_gen_cur_path, stats_path)
                    kill_pg(proc.pid)
            finally:
                with self._proc_lock:
                    if os.environ['GALAXY_GPU_ENABLED'] == "true":
                        os.kill(sp,9)
                        self.post_process(log_file_path, util_path, util_mem_path, mem_free_path, mem_used_path, pcie_link_gen_cur_path, stats_path)
                    self._procs.remove(proc)

            if proc.terminated_by_shutdown:
                if os.environ['GALAXY_GPU_ENABLED'] == "true":
                    os.kill(sp,9)
                    self.post_process(log_file_path, util_path, util_mem_path, mem_free_path, mem_used_path, pcie_link_gen_cur_path, stats_path)
                self._fail_job_local(job_wrapper, "job terminated by Galaxy shutdown")
                return

            stdout_file.seek(0)
            stderr_file.seek(0)
            stdout = self._job_io_for_db(stdout_file)
            stderr = self._job_io_for_db(stderr_file)
            stdout_file.close()
            stderr_file.close()
            log.debug('execution finished: %s' % command_line)
            if os.environ['GALAXY_GPU_ENABLED'] == "true":
                os.kill(sp,9)
                self.post_process(log_file_path, util_path, util_mem_path, mem_free_path, mem_used_path, pcie_link_gen_cur_path, stats_path)
        except Exception:
            if os.environ['GALAXY_GPU_ENABLED'] == "true":
                os.kill(sp,9)
                self.post_process(log_file_path, util_path, util_mem_path, mem_free_path, mem_used_path, pcie_link_gen_cur_path, stats_path)
            log.exception("failure running job %d", job_wrapper.job_id)
            self._fail_job_local(job_wrapper, "failure running job")
            return

        self._handle_metadata_if_needed(job_wrapper)

        job_destination = job_wrapper.job_destination
        job_state = JobState(job_wrapper, job_destination)
        job_state.exit_code_file = default_exit_code_file(job_wrapper.working_directory, job_id)
        job_state.stop_job = False
        self._finish_or_resubmit_job(job_state, stdout, stderr, job_id=job_id)

    def stop_job(self, job_wrapper):
        # if our local job has JobExternalOutputMetadata associated, then our primary job has to have already finished
        job = job_wrapper.get_job()
        job_ext_output_metadata = job.get_external_output_metadata()
        try:
            pid = job_ext_output_metadata[0].job_runner_external_pid  # every JobExternalOutputMetadata has a pid set, we just need to take from one of them
            assert pid not in [None, '']
        except Exception:
            # metadata internal or job not complete yet
            pid = job.get_job_runner_external_id()
        if pid in [None, '']:
            log.warning("stop_job(): %s: no PID in database for job, unable to stop" % job.id)
            return
        pid = int(pid)
        if not check_pg(pid):
            log.warning("stop_job(): %s: Process group %d was already dead or can't be signaled" % (job.id, pid))
            return
        log.debug('stop_job(): %s: Terminating process group %d', job.id, pid)
        kill_pg(pid)

    def recover(self, job, job_wrapper):
        # local jobs can't be recovered
        job_wrapper.change_state(model.Job.states.ERROR, info="This job was killed when Galaxy was restarted.  Please retry the job.")

    def shutdown(self):
        super().shutdown()
        with self._proc_lock:
            for proc in self._procs:
                proc.terminated_by_shutdown = True
                kill_pg(proc.pid)
                proc.wait()  # reap

    def _fail_job_local(self, job_wrapper, message):
        job_destination = job_wrapper.job_destination
        job_state = JobState(job_wrapper, job_destination)
        job_state.fail_message = message
        job_state.stop_job = False
        self.fail_job(job_state, exception=True)

    def _handle_metadata_if_needed(self, job_wrapper):
        if not self._embed_metadata(job_wrapper):
            self._handle_metadata_externally(job_wrapper, resolve_requirements=True)

    def _embed_metadata(self, job_wrapper):
        job_destination = job_wrapper.job_destination
        embed_metadata = asbool(job_destination.params.get("embed_metadata_in_job", DEFAULT_EMBED_METADATA_IN_JOB))
        return embed_metadata

    def _prepare_job_local(self, job_wrapper):
        return self.prepare_job(job_wrapper, include_metadata=self._embed_metadata(job_wrapper))

    def _handle_container(self, job_wrapper, proc):
        if not job_wrapper.tool.produces_entry_points:
            return

        while check_pg(proc.pid):
            if job_wrapper.check_for_entry_points(check_already_configured=False):
                return

            sleep(0.5)

    def __poll_if_needed(self, proc, job_wrapper, job_id):
        # Only poll if needed (i.e. job limits are set)
        if not job_wrapper.has_limits():
            return

        job_start = datetime.datetime.now()
        i = 0
        pgid = proc.pid
        # Iterate until the process exits, periodically checking its limits
        while check_pg(pgid):
            i += 1
            if (i % 20) == 0:
                limit_state = job_wrapper.check_limits(runtime=datetime.datetime.now() - job_start)
                if limit_state is not None:
                    job_wrapper.fail(limit_state[1])
                    log.debug('(%s) Terminating process group %d', job_id, pgid)
                    kill_pg(pgid)
                    return True
            else:
                sleep(DEFAULT_POOL_SLEEP_TIME)
