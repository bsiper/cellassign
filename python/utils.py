import numpy as np


def initialize_X(X=None, N=0, verbose=False):
    if X is None:
        if N > 0:
            # X gets matrix with all 1s for N rows
            X = np.ones((N,1))
        else:
            
