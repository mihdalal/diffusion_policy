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
    "partition": "russ_reserved",
    "time": "72:00:00",
    "gpus": 1,
    "cpus_per_gpu": 20,
    "mem": "62g",
    "exclude": "matrix-1-[4,8,10,12,16],matrix-0-[24,38]",
    "signal": "USR1@30",
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

    def __call__(self, checkpoint_path=None):
        """
        wrapped main function for launching diffusion policy code.
        we take in cfg_dict instead of cfg because we need to rebuild the config
        for some reason if we don't do this, the cfg resolve phase fails 
        """
        # launch function in a singularity container: 
        python_cmd = subprocess.check_output("which python", shell=True).decode("utf-8")[:-1]
        singularity_path = '/opt/singularity/bin/singularity'
        subprocess.run([singularity_path, 'exec', '--nv', self.sif_path, self.python_path, 
                        self.file_path, self.output_dir, str(checkpoint_path)], capture_output=True)

    def checkpoint(self, checkpointpath: str) -> submitit.helpers.DelayedSubmission:
        wrapped_callable = WrappedCallable(self.output_dir, self.sif_path, self.python_path, self.file_path)
        checkpoint_path = pathlib.Path(self.output_dir).joinpath('checkpoints', f'latest.ckpt')
        return submitit.helpers.DelayedSubmission(wrapped_callable, checkpoint_path)
    

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config'))
)
def main(cfg: DictConfig):
    sif_path = "/projects/rsalakhugroup/containers/neural_mp.sif"
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

    print(f"Scheduled {job}.")
    # Wait for the job to be running.
    # while job.state != "RUNNING":
    #     time.sleep(1)

    # Simulate preemption.
    # Tries to stop the job after the first stage.
    # If the job is preempted before the end of the first stage, try to increase it.
    # If the job is not preempted, try to decrease it.
    # time.sleep(300)
    # print(f"preempting {job} after {time.time() - t0:.0f}s")
    # job._interrupt()

    score = job.result()

if __name__ == "__main__":
    main()