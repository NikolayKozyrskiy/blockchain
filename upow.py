import numpy as np
from numpy import random
import functools
from multiprocessing import Pool
import time
import polymod

from sympy import Matrix, isprime
import hashlib


def polyval_p(coef, x, p):
    coef = np.asarray(coef)
    x = np.asarray(x)
    y = np.zeros_like(x)
    for i in range(len(coef)):
        y = (y * x + coef[i]) % p
    return y


class Task(object):
    def __init__(self, x: list, p, threads=1, **kwargs):
        self.p = p
        self.threads = threads

        if len(kwargs) == 0:
            self.k = len(x)
            self.n = x[0].shape[0]
            self.d = x[0].shape[1]
            x = list(map(lambda tmp: tmp.reshape(1, -1), x))
            self.x = np.concatenate(np.concatenate(x))

        else:
            self.x = x
            self.k = kwargs['k']
            self.n = kwargs['n']
            self.d = kwargs['d']

        if self.x.size != (self.n * self.k * self.d):
            raise ValueError

    def __str__(self):
        return ('Shape: (%d %d %d)\nPrime: %d\nValues:\n' % (self.k, self.n, self.d, self.p)) + str(self.x)


class Challenge(object):
    def __init__(self, y: np.array, task: Task):
        self.p = task.p
        self.n = task.n
        self.k = task.k
        self.d = task.d
        self.x = task.x
        self.y = y
        self.threads = task.threads

        if y.shape[0] != (self.k * self.d + 1):
            raise ValueError

        if y.size != (self.n * self.k * self.d)*(self.k * self.d + 1):
            raise ValueError

    def __str__(self):
        return ('Shape: (%d %d %d)\nPrime: %d\nValues:\n' % (self.k, self.n, self.d, self.p)) + str(self.y)

    def save(self, filename):
        np.savez(filename, self.x, self.y, np.array([self.k, self.n, self.d, self.p]))


def load_challenge(filename, threads):
    with np.load(filename) as f:
        x = f['arr_0']
        y = f['arr_1']
        arr = f['arr_2']
        k = arr[0]
        n = arr[1]
        d = arr[2]
        p = arr[3]
    return Challenge(y, Task(x, p, threads, k=k, n=n, d=d))


class Solution(object):
    def __init__(self, tau: list, challenge: Challenge):
        self.p = challenge.p
        self.n = challenge.n
        self.k = challenge.k
        self.d = challenge.d
        self.y = challenge.y
        self.x = challenge.x
        self.tau = tau
        self.threads = challenge.threads

        if self.y.shape[0] != (self.k * self.d + 1):
            raise ValueError

        if self.y.size != (self.n * self.k * self.d)*(self.k * self.d + 1):
            raise ValueError

    def __str__(self):
        return ('Shape: (%d %d %d)\nPrime: %d\nValues:\nTau:' % (self.k, self.n, self.d, self.p)) + str(self.y) + str(self.tau)


def Gen(task: Task):
    r = random.randint(0, task.p, len(task.x))
    #r = np.arange(0, len(task.x), 1, int)
    f = functools.partial(Gen1, x=task.x, r=r, p=task.p)
    with Pool(task.threads) as pool:
        y = np.array( tuple( pool.map( f, range(1, task.k * task.d + 2) ) ) )
    return Challenge(y, task)


def Solve(challenge: Challenge):
    f = functools.partial(Solve1, n=challenge.n, k=challenge.k, p=challenge.p, d=challenge.d)
    with Pool(challenge.threads) as pool:
        ret = Solution(list(pool.map(f, challenge.y)), challenge)
    return ret


def Verify(solution: Solution):
    f = functools.partial(Verify1, n=solution.n, k=solution.k, d=solution.d, p=solution.p)
    with Pool(solution.threads) as pool:
        rets = list( pool.map(f, list(zip(solution.y, solution.tau))))
    return functools.reduce(lambda x, y: x and y, rets)


def Gen1(t, x, r, p):
    return (x + t * r) % p


def Solve1(y_t, k, n, d, p):
    alpha = []

    q_1_s = get_q_s(y_t, k, n, d, p, 0, []) % p
    gOV = functools.reduce(lambda x, y: (x + polyval_p(q_1_s, y, p)) % p, range(n), 0) % p
    tau = [np.array([gOV]), q_1_s]

    for s in range(1, k - 1):
        alpha.append(int(hashlib.sha256(np.concatenate(tau)).hexdigest(), 16) % n)  # todo check hash
        tau.append(get_q_s(y_t, k, n, d, p, s, alpha) % p)
    return tau


def Verify1(y_tau_t, k, n, d, p):
    alpha = []
    y_t = y_tau_t[0]
    tau_t = y_tau_t[1]

    gOV = functools.reduce(lambda x, y: (x + polyval_p(tau_t[1], y, p)) % p, range(n), 0) % p
    if gOV != tau_t[0]:
        return False

    for s in range(0, k - 2):
        alpha.append(int(hashlib.sha256(np.concatenate(tau_t[0:s+2])).hexdigest(), 16) % n)  # todo check
        q_alpha_s = polyval_p(tau_t[s+1], alpha[s], p) % p
        part_sum = functools.reduce(lambda x, y: (x + polyval_p(tau_t[s+2], y, p)) % p, range(n), 0) % p
        if q_alpha_s != part_sum:
            return False
    alpha.append(int(hashlib.sha256(np.concatenate(tau_t[0:k-1])).hexdigest(), 16) % n)
    q_real_coeff = get_q_s(y_t, k, n, d, p, k - 1, alpha) % p
    end_sum = functools.reduce(lambda x, y: (x + polyval_p(q_real_coeff, y, p)) % p, range(n), 0) % p
    q_sum_solver = polyval_p(tau_t[-1], alpha[-1], p) % p
    if q_sum_solver != end_sum:
        return False
    return True


def make_x_vector(k, n, d):
    return list(map(lambda tmp: random.randint(0, 2, (n, d)), range(k)))
    #return list(map(lambda tmp: np.arange(1, n*d+1).reshape(n, d) % 2, range(k)))


# s-номер множества , l - номер коодинаты в векторе y - входной вектор
def get_phi_s_l_polinom(y, k, n, d, p, s, l):
    coeffs = polymod.lagrange(np.arange(0, n, dtype=int), y[s * n * d + l * n: s * n * d + (l + 1) * n], p, n)
    return coeffs


def get_q_s(y, k, n, d, p, s, alpha):
    coeffs_res = np.zeros((n - 1) * d + 1, dtype=int)
    for i in range(n ** (k - s -1)):
        coeffs = np.array([1], dtype=int)

        for l in range(d):
            buf_var = 1
            coord_number_l = l * n
            #подсчет произведения phi l-ых при постоянных alpha_1..alpha_s_minus_1
            for j in range(s + 1, k):
                set_number_j = n * j * d
                vector_number_in_j_set = int(i / (n ** (j - s - 1 )))
                buf_var = (buf_var * y[set_number_j + coord_number_l + vector_number_in_j_set % n]) % p
            for e in range(s):
                set_number_e = n * e * d
                buf_var = (buf_var * y[set_number_e + coord_number_l + alpha[e]]) % p

            coeff_one = ((-1) * get_phi_s_l_polinom(y, k, n, d, p, s, l) * buf_var) % p
            coeff_one[- 1] = (coeff_one[- 1] + 1) % p

            #свертка коэффициентов (q(alpha(1),..,alpha(s-1), x , I(s+1),...,I(k))
            coeffs = np.convolve(coeffs, coeff_one) % p

        #сумма по всем выборкам (I(s+1),..,I(k))
        coeffs_res = (coeffs_res + coeffs) % p

    return coeffs_res


def test1(k, n, threads):
    d = int(np.ceil(np.log2(n) ** 2))
    p = int(np.ceil(n ** np.log2(n)))
    while not isprime(p):
        p += 1
    print("k %d n %d d %d p %d thr %d" % (k, n, d, p, threads))
    x = make_x_vector(k, n, d)
    start = time.time()

    challenge = Gen(Task(x, p, threads=threads))
    gen_time = time.time() - start
    print('gen_time: ', gen_time)

    solution = Solve(challenge)
    solve_time = time.time() - gen_time - start
    print('solve_time: ', solve_time)

    ans = Verify(solution)
    print(ans)
    ver_time = time.time() - gen_time - solve_time - start
    print('ver_time: ', ver_time)


def test():
    for k in range(2, 10):
        for n in range(1, 10):
            for threads in (1, 2, 4, 8, 16):
                test1(k, n, threads)


if __name__ == '__main__':
    test1(3,7,1)

