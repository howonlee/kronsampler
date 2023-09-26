import bitstring
import functools
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
import operator

def arr():
    res = npr.rand()
    res_arr = np.array([res, 1 - res])
    return res_arr

def sample(members1, members2, idx):
    if idx % 1000 == 0:
        print(idx)
    ###########
    ###########
    ###########
    arr = np.zeros(len(members))
    for idx in range(len(members)):
        arr[idx] = npr.choice(2, p=members[idx])
    arr_str = bitstring.BitArray(bin="".join(map(lambda x: str(int(x)), arr)))
    return arr_str.uint


if __name__ == "__main__":
    members1 = [arr() for _ in range(6)]
    members2 = [arr() for _ in range(6)]

    samples = [sample(members1, members2, idx) for idx in range(int(1e5))]

    res1 = functools.reduce(lambda x, y: np.kron(x, y), members1) * 0.5
    res2 = functools.reduce(lambda x, y: np.kron(x, y), members2) * 0.5
    print((res1 + res2).sum())
    plt.plot(res1 + res2, alpha=0.5)
    # plt.hist(samples, bins=(int(2 ** 6)), alpha=0.5, density=True)
    plt.show()
