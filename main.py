from core.solver import SOLVER
import argparse
import torch
import numpy as np
import random
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Training mode and dataset
    parser.add_argument('--train', type=int,default=1,help='train or test')
    parser.add_argument('--base', type=str,default='baseline',help='dataset',choices=['baseline','mdn'])
    parser.add_argument('--gpu', type=int,default=0,help='gpu')

    args = parser.parse_args()
    torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode
    torch.cuda.manual_seed_all(0)  # GPU seed
    torch.manual_seed(seed=10)
    np.random.seed(seed=0)
    random.seed(0)

    sol = SOLVER(args)
    sol.train()