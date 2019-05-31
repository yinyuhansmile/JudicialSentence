import numpy as np

a = np.zeros((3,4))

a[1,2:] = 1
a[2, 3] = 1
a[0] = 1



print(a.sum(axis=1))
b = a.sum(axis=1)
b = b[np.newaxis, :]
b = np.transpose(b)

print(b)
# a/a.sum(axis=1)
print(a/b)
print(a)
