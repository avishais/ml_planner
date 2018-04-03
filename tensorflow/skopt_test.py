import numpy as np
from skopt import gp_minimize

def f(x):
    y = ((x[0]-1)**2 + x[1]**2)
    print(x, y)
    return y
    # return (np.sin(5 * x[0]) * (1 - np.tanh(x[0] ** 2)) *
            # np.random.randn() * 0.1)

res = gp_minimize(func=f, dimensions=[(-2.0, 2.0), (-2.0, 2.0)], n_calls=40)

print(res)