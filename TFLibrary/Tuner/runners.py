import os
import tempfile
import subprocess


class Runner(object):
    """Runner takes in Parameter instances and execute them"""
    def __init__(self, logdir, tmp_file_prefix="RUNNER_"):
        if not os.path.isdir(logdir):
            # print("Creating ", logdir)
            os.mkdir(logdir)

        self._logdir = logdir
        self._tmp_files = []
        self._tmp_file_prefix = tmp_file_prefix

    def run(self, experiments):
        raise NotImplementedError

    def _create_tmp_file(self):
        tmp_file = tempfile.NamedTemporaryFile(
            prefix=self._tmp_file_prefix,
            dir=self._logdir, delete=False)
        self._tmp_files.append(tmp_file.name)
        return tmp_file.name

    def _clean_tmp_files(self):
        for f in self._tmp_files:
            os.remove(f)


class SyncMultiGPURunner(Runner):
    """Synchronous MultiGPU Runner.

    `SyncMultiGPURunner` takes a set of bash scripts (experiments),
    run them in parallel, and block until all experiments finish.
    """
    def __init__(self, gpus, logdir, print_command=False):
        if not isinstance(gpus, (list, tuple)):
            raise TypeError("`gpus` must be list or tuple")
        if not isinstance(gpus[0], str):
            raise TypeError("`gpus` should be a list of strings")
        
        super(SyncMultiGPURunner, self).__init__(logdir=logdir)
        self._gpus = gpus
        self._print_command = print_command

    @property
    def max_processes(self):
        return len(self._gpus)

    def run(self, experiments):
        if not isinstance(experiments, (list, tuple)):
            raise TypeError(
                "`experiments` must be list or tuple, "
                "found %s" % type(experiments))

        if not isinstance(experiments[0], (list, tuple)):
            raise TypeError(
                "each `experiment` should be a list of commands such that "
                "when joined becomes a valid bash script. "
                "Found type %s" % type(experiments[0]))

        if len(experiments) > len(self._gpus):
            raise ValueError(
                "`experiments` has length %d > `num_gpus`, which has "
                "length %d" % (len(experiments), len(self._gpus)))
        
        # Create temp file
        tmp_file = self._create_tmp_file()
        
        # Run the experiments
        processes = []
        for experiment, gpu_id in zip(experiments, self._gpus):
            # e.g. FileName-0, FileName-2, FileName-3
            fname = "-".join([tmp_file, gpu_id])
            logfile = open(fname + ".log", "w")
            with open(fname, "w") as f:
                f.write("\n".join(experiment))

            # e.g. CUDA_VISIBLE_DEVICES=3 bash FileName-3
            command = "CUDA_VISIBLE_DEVICES=%s bash %s" % (gpu_id, fname)
            if self._print_command:
                print("EXECUTING: \t " + command)
            
            # Launch the processes
            process = subprocess.Popen(
                command, stdout=logfile, shell=True)

            processes.append((process, logfile))

        # Check for completions
        for process, logfile in processes:
            process.wait()
            logfile.close()
