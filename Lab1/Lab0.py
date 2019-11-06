import time
import numpy as np


def sum_trad():
    start = time.time()
    x = range(10000000)
    y = range(10000000)
    z = []
    for i in range(len(x)):
        z.append(x[i]+y[i])
    return time.time() - start



def sum_numpy():
    start = time.time()
    X = np.arange(10000000)
    Y = np.arange(10000000)
    Z = X+Y
    return time.time() - start

#print('time sum:', sum_trad(), ' time sum numpy:', sum_numpy())


arr = np.array([2,6,5,9], float)
print(arr)
print(type(arr))

arr = np.array([1,2,3], float)
arr.tolist()
list(arr)
print(arr)
print(type(arr))


arr = np.array([1,2,3], float)
arr1 = arr
arr2 = arr.copy() #'''copy needs to be used, the assignment operator will just link the new name to the same original object'''
arr[0] = 0

print(arr)
print(arr1)
print(arr2)


arr = np.array([10,20,33], float)
print(arr)
arr.fill(1)
print(arr)

print(np.random.permutation(3))

print(np.random.normal(0,1,5)) #Permutations from a normal distribution

print(np.random.random(5)) #numbers between 0 and 1

print(np.identity(5, dtype=float)) #identitesmatris

print(np.eye(3, k=1, dtype=float)) # Ones along the k:th diagonal

print(np.ones((2,3),dtype=float))

print(np.zeros(6, dtype=int))

arr = np.array([[13,32,31],[64,25,76]], float)
print(np.zeros_like(arr))
print(np.ones_like(arr))

arr1 = np.array([1,3,2])
arr2 = np.array([3,4,6])

print(np.vstack([arr1,arr2])) # merge two arrays

print(np.random.rand(2,3)) # twodimensional random matrix

print(np.random.multivariate_normal([10, 0], [[3,1], [1,4]], size=[5,]))

arr = np.array([2.,6.,5.,5.,])
print(arr[:3]) #only the first 3 ones



