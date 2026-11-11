import os
import random
from typing import Optional, NoReturn

import numpy as np
import torch


def fix_random_seed(seed: Optional[int] = 2021) -> NoReturn:
    if seed is None:
        return

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def fix_all_random_seed(seed: Optional[int] = 2021) -> NoReturn:
    if seed is None:
        return

    # Set CUDA's deterministic workspace before any model or CUDA operation is
    # created.  This is needed for reproducible cuBLAS reductions.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
