from typing import Dict
from diffusion_policy.policy.base_pcd_policy import BasePcdPolicy

class BasePcdRunner:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def run(self, policy: BasePcdPolicy) -> Dict:
        raise NotImplementedError()
