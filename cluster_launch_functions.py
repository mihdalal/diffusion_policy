def function(output_dir, checkpoint_path=None):
    import sys
    # use line-buffering for both stdout and stderr
    sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

    import hydra
    from omegaconf import OmegaConf
    from diffusion_policy.workspace.base_workspace import BaseWorkspace

    import json
    import os
    with open(os.path.join(output_dir, 'cfg.json'), 'r') as f:
        cfg_dict = json.load(f)

    cfg = OmegaConf.create(cfg_dict)
    # initialize the hydra runtime with the config
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg, output_dir)
    import torch
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("medium")
    if checkpoint_path != 'None':
        workspace.load_checkpoint(checkpoint_path)
    def handler(signum, frame):
        print('Signal handler called with signal', signum)
        workspace.save_checkpoint()
        exit()
    import signal 
    signal.signal(signal.SIGUSR1, handler)
    workspace.run()
    

if __name__ == "__main__":
    import sys
    output_dir = sys.argv[1]
    checkpoint_path = sys.argv[2]
    function(output_dir, checkpoint_path)
