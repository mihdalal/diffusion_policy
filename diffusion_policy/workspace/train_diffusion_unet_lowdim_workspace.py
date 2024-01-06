if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)
import torch
import torch._dynamo                                                    
import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import numpy as np
import random
import wandb
import tqdm
import shutil

from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusers.training_utils import EMAModel
import torch.distributed as dist
import torch.nn as nn
OmegaConf.register_new_resolver("eval", eval, replace=True)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group(rank=rank, world_size=world_size)

class TrainDiffusionUnetLowdimWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetLowdimPolicy
        self.model = hydra.utils.instantiate(cfg.policy)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())
        
        self.global_step = 0
        self.epoch = 0

    def run(self, world_size=1, rank=0, ddp=False):
        cfg = copy.deepcopy(self.cfg)

        setup(rank, world_size)

        device = torch.device(rank)
        torch.cuda.set_device(rank)
        self.model.cuda(rank)
        if ddp:
            self.model.model = nn.parallel.DistributedDataParallel(self.model.model, device_ids=[rank])

        self.ema_model: DiffusionUnetLowdimPolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path(tag=cfg.ckpt_tag)
            if lastest_ckpt_path.is_file():
                map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path, map_location=map_location)

        # configure dataset
        dataset: BaseLowdimDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseLowdimDataset)
        if ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                        num_replicas=world_size,
                                                                        rank=rank)
        else:
            train_sampler = None
        train_dataloader = DataLoader(dataset, sampler=train_sampler, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        if ddp:
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset,
                                                                        num_replicas=world_size,
                                                                        rank=rank)
        else:
            val_sampler = None

        val_dataloader = DataLoader(val_dataset, sampler=val_sampler, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure env runner
        if rank == 0:
            env_runner: BaseLowdimRunner
            env_runner = hydra.utils.instantiate(
                cfg.task.env_runner, world_size=world_size,
                output_dir=self.output_dir)
            assert isinstance(env_runner, BaseLowdimRunner)

        # configure logging
        if rank == 0:
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging
            )
            wandb.config.update(
                {
                    "output_dir": self.output_dir,
                },
                allow_val_change=True
            )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        best_val_loss = float('inf')
        best_test_score = 0
        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        ckpt_log_path = os.path.join(self.output_dir, 'checkpoint_logs.txt')
        # create ckpt log path file
        if not os.path.exists(ckpt_log_path):
            with open(ckpt_log_path, 'w') as f:
                f.write('')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # compute loss
                        raw_loss = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        
                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model)

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            if rank == 0:
                                process_train_loss = step_log['train_loss']
                                for src in range(1, world_size):
                                    tensor = torch.zeros(1)
                                    dist.recv(tensor=tensor, src=src)
                                    process_train_loss += tensor.item()
                                process_train_loss /= world_size
                                step_log['train_loss'] = process_train_loss
                                train_losses.remove(raw_loss_cpu)
                                train_losses.append(process_train_loss)
                                # log of last step is combined with validation and rollout
                                wandb_run.log(step_log, step=self.global_step)
                                json_logger.log(step_log)
                            else:
                                tensor = torch.tensor(step_log['train_loss'])
                                dist.send(tensor=tensor, dst=0)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break
                
                if rank == 0:
                    # at the end of each epoch
                    # replace train_loss with epoch average
                    train_loss = np.mean(train_losses)
                    step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                if (self.epoch % cfg.training.rollout_every) == 0 and rank == 0:
                    runner_log = env_runner.run(policy)
                    # log all
                    step_log.update(runner_log)

                    test_score = runner_log['test/mean_score']
                    if test_score > best_test_score and rank == 0:
                        best_test_score = test_score
                        best_ckpt_path = os.path.join(self.output_dir, 'checkpoints', 'best_test_score.ckpt')
                        self.save_checkpoint(path=best_ckpt_path)
                        # save the value and epoch of the best test score to a txt file
                        with open(ckpt_log_path, 'a') as f:
                            f.write(f'Best test score: {best_test_score} at epoch {self.epoch}\n')

                # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                loss = self.model.compute_loss(batch)
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss

                            if val_loss < best_val_loss and rank == 0:
                                best_val_loss = val_loss
                                best_ckpt_path = os.path.join(self.output_dir, 'checkpoints', 'best_val_loss.ckpt')
                                self.save_checkpoint(path=best_ckpt_path)

                                with open(ckpt_log_path, 'a') as f:
                                    f.write(f'Best val loss: {best_val_loss} at epoch {self.epoch}\n')

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0 and rank == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = train_sampling_batch
                        obs_dict = {'obs': batch['obs']}
                        gt_action = batch['action'].cuda(non_blocking=True)
                        
                        result = policy.predict_action(obs_dict)
                        if cfg.pred_action_steps_only:
                            pred_action = result['action']
                            start = cfg.n_obs_steps - 1
                            end = start + cfg.n_action_steps
                            gt_action = gt_action[:,start:end]
                        else:
                            pred_action = result['action_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        # log
                        step_log['train_action_mse_error'] = mse.item()
                        # release RAM
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse
                
                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0 and rank == 0:
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    # topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    # if topk_ckpt_path is not None:
                    #     self.save_checkpoint(path=topk_ckpt_path)
                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                if rank == 0:
                    process_val_loss = step_log.get('val_loss')
                    for src in range(1, world_size):
                        tensor = torch.zeros(1)
                        dist.recv(tensor=tensor, src=src)
                        process_val_loss += tensor.item()
                    process_val_loss /= world_size
                    step_log['val_loss'] = process_val_loss
                    # log of last step is combined with validation and rollout
                    wandb_run.log(step_log, step=self.global_step)
                    json_logger.log(step_log)
                else:
                    tensor = torch.tensor(step_log['val_loss'])
                    dist.send(tensor=tensor, dst=0)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetLowdimWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
