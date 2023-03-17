from typing import Union, Callable, List, Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class InverseSquareRootScheduler(LambdaLR):
    def __init__(self, optimizer: Optimizer, lr_lambda: Union[Callable[[int], float], List[Callable[[int], float]]],
                 last_epoch: int = ...) -> None:
        super().__init__(optimizer, lr_lambda, last_epoch)





