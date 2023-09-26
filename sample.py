import numpy as np
import numpy.random as npr
import functools
import bitstring
import matplotlib.pyplot as plt

def arr():
    res = npr.rand()
    res_arr = np.array([res, 1 - res])
    return res_arr

def sample(members, idx):
    if idx % 1000 == 0:
        print(idx)
    arr = np.zeros(len(members))
    for idx in range(len(members)):
        arr[idx] = npr.choice(2, p=members[idx])
    arr_str = bitstring.BitArray(bin="".join(map(lambda x: str(int(x)), arr)))
    return arr_str.uint


if __name__ == "__main__":
    members = [arr() for _ in range(6)]

    samples = [sample(members, idx) for idx in range(int(1e5))]

    res = functools.reduce(lambda x, y: np.kron(x, y), members)
    plt.plot(res, alpha=0.5)
    plt.hist(samples, bins=(int(2 ** 6)), alpha=0.5, density=True)
    plt.show()
