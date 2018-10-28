import os
import tempfile
from TFLibrary.utils import misc_utils
# in the servers I use, the system will detach $HOME
# after certain hours. As a result, executing `parallel ...`
# will raise Permission Denied issue since `parallel` is located
# in `$HOME/bin/parallel`. A workaround is to copy the program
# into another directory that will be detached, so have to
# manually specify the final directory to the command
# e.g. cp `which parallel` $PARALLEL_CMD
# PARALLEL_CMD = "/playpen/home/han/parallel_mirror"
PARALLEL_CMD = "parallel"


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


class BasicRunner(Runner):
    def __init__(self, logdir, print_command=False):
        super(BasicRunner, self).__init__(logdir=logdir)
        self._print_command = print_command

    def run(self, experiments):
        if not isinstance(experiments, (list, tuple)):
            raise TypeError(
                "`experiments` should be a list of commands such that "
                "when joined becomes a valid bash script. "
                "Found type %s" % type(experiments))

        tmp_file = self._create_tmp_file()
        _run_single_command(
            tmp_file, experiments,
            print_command=self._print_command)


class MultiGPURunner(Runner):
    def __init__(self, gpus, logdir, print_command=False):
        if not isinstance(gpus, (list, tuple)):
            raise TypeError("`gpus` must be list or tuple")
        if not isinstance(gpus[0], str):
            raise TypeError("`gpus` should be a list of strings")
        
        super(MultiGPURunner, self).__init__(logdir=logdir)
        self._gpus = gpus
        self._print_command = print_command

    @property
    def num_gpus(self):
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

        tmp_file = self._create_tmp_file()
        gpu_ids = self._gpus[:len(experiments)]
        _run_multiple_commands(
            tmp_file, experiments,
            gpu_ids=gpu_ids,
            print_command=self._print_command)


def _run_single_command(fname, command, print_command=False):
    """Launch the process in a separate screen"""
    with open(fname, "w") as f:
        f.write("\n".join(command))

    command = "bash %s  >%s.log 2>&1 " % (fname, fname)

    if print_command:
        print("EXECUTING: \t " + command)

    misc_utils.run_command(command)


def _run_multiple_commands(fname, commands, gpu_ids=None, print_command=False):
    """http://www.shakthimaan.com/posts/2014/11/27/gnu-parallel/news.html"""
    if not gpu_ids:
        raise ValueError("In Single GPU setting, use _run_single_command")

    if not isinstance(gpu_ids, (list, tuple)):
        raise TypeError("`gpu_ids` must be list of GPU IDs")

    if len(commands) != len(gpu_ids):
        raise ValueError("%d commands != %d gpu_ids" % (
            len(commands), len(gpu_ids)))

    # e.g. FileName-0
    AddGpuIdToFileName = lambda gpu_id: "-".join([fname, gpu_id])

    # e.g. FileName-0, FileName-2, FileName-3
    for command, gpu_id in zip(commands, gpu_ids):
        with open(AddGpuIdToFileName(gpu_id), "w") as f:
            f.write("\n".join(command))

    # https://stackoverflow.com/questions/22187834/gnu-parallel-output-each-job-to-a-different-file
    # quote out the redirect
    command = (  # add --dry-run after `parallel` to test commands
        "%s \'CUDA_VISIBLE_DEVICES=\"{}\" bash %s >%s.log 2>&1\' ::: %s"
        % (PARALLEL_CMD,
           AddGpuIdToFileName("{}"),
           AddGpuIdToFileName("{}"),
           " ".join([i for i in gpu_ids])))

    if print_command:
        print("EXECUTING: \t " + command)

    misc_utils.run_command(command)
