import os
import subprocess
import hydra
from omegaconf import DictConfig, OmegaConf
import submitit
from hydra.core.hydra_config import HydraConfig
import time
import json
import subprocess

slurm_additional_parameters = {
    #"partition": "deepaklong",
    "partition": "all",
    #"time": "2-00:00:00",
    "time": "6:00:00",
    "gpus": 1,
    "cpus_per_gpu": 32,
    "mem": "100g",
    #"exclude": "matrix-1-[4,8,10,12,16],matrix-0-[24,38]",
    # "nodelist": "grogu-1-3"
    "exclude": "grogu-1-14, grogu-1-19, grogu-0-24, grogu-1-[9,24,29], grogu-3-[1,3,5,9,11,25,27], grogu-3-[15,17,19,21,23], grogu-3-29", 
}
import pathlib

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

class WrappedCallable(submitit.helpers.Checkpointable):
    def __init__(self, output_dir, sif_path, python_path, file_path):
        self.output_dir = output_dir
        self.sif_path = sif_path
        self.python_path = python_path
        self.file_path = file_path
        self.p = None

    def __call__(self, checkpoint_path=None):
        """
        wrapped main function for launching diffusion policy code.
        we take in cfg_dict instead of cfg because we need to rebuild the config
        for some reason if we don't do this, the cfg resolve phase fails 
        """
        # launch function in a singularity container: 
        singularity_path = 'singularity'
        cmd = f"{singularity_path} exec --nv {self.sif_path} {self.python_path} {self.file_path} {self.output_dir} \'{str(checkpoint_path)}\'"
        self.p = subprocess.Popen(cmd, shell=True)
        while True:
            pass

    def checkpoint(self, checkpointpath: str) -> submitit.helpers.DelayedSubmission:
        print("sending checkpoint signal")
        import signal
        os.kill(self.p.pid, signal.SIGUSR1)
        print("wait for 30s")
        time.sleep(30)
        print("setup new callable")
        wrapped_callable = WrappedCallable(self.output_dir, self.sif_path, self.python_path, self.file_path)
        checkpoint_path = pathlib.Path(self.output_dir).joinpath('checkpoints', f'latest.ckpt')
        print("RESUBMITTING")
        return submitit.helpers.DelayedSubmission(wrapped_callable, checkpoint_path)
    

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config'))
)
def main(cfg: DictConfig):
    sif_path = "/home/sbahl2/research/neural_mp/neural_mp/containers/neural_mp_bash.sif"
    # Generate the command
    output_dir = HydraConfig.get().runtime.output_dir
    cfg_process = cfg.copy()
    python_cmd = subprocess.check_output("which python", shell=True).decode("utf-8")[:-1]
    executor = submitit.AutoExecutor(
        folder=output_dir + "/%j",
    )
    executor.update_parameters(slurm_additional_parameters=slurm_additional_parameters)
    # executor is the submission interface (logs are dumped in the folder)
    cfg_dict = OmegaConf.to_container(cfg_process, resolve=True)
    # dump cfg_dict to output_dir
    with open(os.path.join(output_dir, 'cfg.json'), 'w') as f:
        json.dump(cfg_dict, f)
    # absolute path to cluster_launch_functions.py
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cluster_launch_functions.py")
    wrapped_callable = WrappedCallable(output_dir, sif_path, python_cmd, file_path)
    t0 = time.time()
    job = executor.submit(wrapped_callable, None)

if __name__ == "__main__":
    main()
