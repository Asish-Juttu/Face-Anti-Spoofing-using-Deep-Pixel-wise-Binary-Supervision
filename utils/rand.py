import numpy as np
import torch
a = torch.randn(4, 4, 3)
print(a)
score = torch.mean(a, axis = (1,2,3))