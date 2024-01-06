def function(output_dir, checkpoint_path=None):
    import sys
    # use line-buffering for both stdout and stderr
    sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

    from omegaconf import OmegaConf

    import json
    import os
    from train import train
    import torch.multiprocessing as mp

    os.environ['WANDB_API_KEY'] = "010fcba9b0530d8e86f54a8e7e68725a06be7dba"
    with open(os.path.join(output_dir, 'cfg.json'), 'r') as f:
        cfg_dict = json.load(f)

    cfg = OmegaConf.create(cfg_dict)
    with open(os.path.join(output_dir, 'cfg_ddp.json'), 'w') as f:
        json.dump(cfg_dict, f, indent=4)
    if cfg.ddp.use:
        mp.spawn(train, nprocs=cfg.ddp.num_gpus, args=(output_dir, cfg.ddp.num_gpus, checkpoint_path))
    else:
        train(0, output_dir, 1, checkpoint_path)
    

if __name__ == "__main__":
    import sys
    output_dir = sys.argv[1]
    checkpoint_path = sys.argv[2]
    function(output_dir, checkpoint_path)
