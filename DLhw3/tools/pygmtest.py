import pygmtools as pygm
import numpy as np
import jittor as jt

if __name__ == "__main__":
    pygm.BACKEND = "jittor"
    np.random.seed(0)
    s_2d = pygm.utils.from_numpy(np.random.randint(low=0, high=10, size=(3, 3)))
    print(s_2d)
    x = pygm.linear_solvers.sinkhorn(s_2d, backend="jittor")
    print(x)
