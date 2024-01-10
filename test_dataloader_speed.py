"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

from diffusion_policy.dataset.base_dataset import BasePcdDataset
from torch.utils.data import DataLoader
import hydra
from omegaconf import OmegaConf
import pathlib

from tqdm import tqdm

from neural_mp.geometry import TorchCuboids, TorchCylinders, TorchSpheres, construct_mixed_point_cloud_torch
from neural_mp.envs.franka_pybullet_env import render_single_pointcloud
# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config'))
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    dataloader = DataLoader(dataset, **cfg.dataloader)
    with tqdm(dataloader, desc=f"Training epoch {0}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
        for batch_idx, batch in enumerate(tepoch):
            for k in batch['obs']:
                batch['obs'][k] = batch['obs'][k].cuda()

if __name__ == "__main__":
    main()
