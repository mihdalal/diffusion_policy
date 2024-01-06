"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

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
    workspace: BaseWorkspace = cls(cfg, output_dir)
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
    cfg_dict = OmegaConf.to_container(cfg)
    output_dir = HydraConfig.get().runtime.output_dir
    import json
    with open(os.path.join(output_dir, 'cfg_ddp.json'), 'w') as f:
        json.dump(cfg_dict, f, indent=4)
    if cfg.ddp.use:
        mp.spawn(train, nprocs=cfg.ddp.num_gpus, args=(output_dir, cfg.ddp.num_gpus))
    else:
        train(0, output_dir, 1)

if __name__ == "__main__":
    main()
