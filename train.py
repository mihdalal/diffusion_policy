"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

import subprocess
import sys
import os
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace
import torch.multiprocessing as mp
from hydra.core.hydra_config import HydraConfig
# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

def train(rank, output_dir, world_size, checkpoint_path='None'):
    import sys
    # use line-buffering for both stdout and stderr
    sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

    import hydra
    from omegaconf import OmegaConf
    from diffusion_policy.workspace.base_workspace import BaseWorkspace

    import json
    import os
    import torch
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("medium")
    os.environ['WANDB_API_KEY'] = "010fcba9b0530d8e86f54a8e7e68725a06be7dba"

    with open(os.path.join(output_dir, 'cfg_ddp.json'), 'r') as f:
        cfg_dict = json.load(f)
    cfg = OmegaConf.create(cfg_dict)
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg, output_dir, rank=rank, world_size=world_size)
    if cfg.start_from_checkpoint:
        # basically output_dir is a newly generated directory for the current exp
        # cfg.output_dir holds the old directory from which the checkpoint came from
        # we reset the cfg output_dir to None after so that future checkpoints will 
        # load from the newly generated output directory
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        path = pathlib.Path(cfg.output_dir).joinpath('checkpoints', f'{cfg.ckpt_tag}.ckpt')
        workspace.load_checkpoint(path=path, map_location=map_location)
        workspace.global_step = 0
        workspace.epoch = 0
        workspace._output_dir = output_dir
        cfg.output_dir = None

    if checkpoint_path != 'None':
        workspace.load_checkpoint(checkpoint_path)
    def handler(signum, frame):
        if rank == 0:
            print('Signal handler called with signal', signum)
            workspace.save_checkpoint()
        exit()
    import signal 
    signal.signal(signal.SIGUSR1, handler)
    workspace.run(world_size=world_size, rank=rank, ddp=cfg.ddp.use)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config'))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)
    dataset_name = cfg.task.dataset.dataset_path.split('/')[-1]
    print(subprocess.run(['rsync', '-azvP', f'{cfg.task.dataset.dataset_path}', f'/dev/shm/{dataset_name}'], capture_output=True))
    cfg.task.dataset.dataset_path = f'/dev/shm/{dataset_name}'
    cfg.task.env_runner.dataset_path = f'/dev/shm/{dataset_name}'
    cfg_dict = OmegaConf.to_container(cfg)
    if cfg.output_dir is None or cfg.start_from_checkpoint:
        # if we are not given an output dir -> generate one
        # OR if we are starting from a checkpoint -> generate a new output dir
        # the logic in train.py will handle loading from the old output dir 
        # which is stored in cfg.output_dir
        output_dir = HydraConfig.get().runtime.output_dir
    else:
        output_dir = cfg.output_dir
    import json
    with open(os.path.join(output_dir, 'cfg_ddp.json'), 'w') as f:
        json.dump(cfg_dict, f, indent=4)
    if cfg.ddp.use:
        mp.spawn(train, nprocs=cfg.ddp.num_gpus, args=(output_dir, cfg.ddp.num_gpus))
    else:
        train(0, output_dir, 1)

if __name__ == "__main__":
    main()
