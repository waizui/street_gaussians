import numpy as np
import torch


if __name__ == "__main__":
    print(torch.tensor([1, 0, 0, 0]).expand((3, 4)))
