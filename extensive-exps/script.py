import numpy as np

a = np.array([1, 2, 3, 4])
b = np.array([2, 3, 4, 5])

l = [a, b]

print(np.std(l, axis=0))

