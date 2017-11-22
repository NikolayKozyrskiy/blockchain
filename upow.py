import numpy as np
from numpy import random
import functools
from multiprocessing import Pool
import time

from sympy import Matrix
import hashlib

WORKERS = 2


def Gen(x, p, k, d):
    r = random.randint(0, p, len(x))
    f = functools.partial(Gen1, x=x, r=r, p=p)
    return np.array( tuple( Pool(WORKERS).map( f, range(1, k*d+2) ) ) )


def Solve(y, n, k, d, p):
    f = functools.partial(Solve1, n=n, k=k, p=p, d=d)
    return list(Pool(WORKERS).map(f, y))


def Verify(y, tau, n, k, d, p):
    f = functools.partial(Verify1, n=n, k=k, d=d, p=p)
    rets = list( Pool(WORKERS).map(f, list(zip(y, tau))))
    return functools.reduce(lambda x, y: x and y, rets)


def Gen1(t, x, r, p):
    return (x + t * r) % p

def Solve1(y_t, n, k, d, p):
    alpha = []

    q_1_s = get_q_s(y_t, n, k, d, p, 0, []) % p
    gOV = functools.reduce(lambda x, y: (x + np.polyval(q_1_s, y)) % p, range(n), 0) % p
    tau = [gOV, q_1_s]

    for s in range(1, k - 1):
        alpha.append(int(hashlib.sha256(np.array([1, 1, 1])).hexdigest(), 16) % n)  # todo check hash
        tau.append(get_q_s(y_t, n, k, d, p, s, alpha) % p)

    return tau


def Verify1(y_tau_t, n, k, d, p):
    alpha = []
    y_t = y_tau_t[0]
    tau_t = y_tau_t[1]

    gOV = functools.reduce(lambda x, y: (x + np.polyval(tau_t[1], y)) % p, range(n), 0) % p
    if gOV != tau_t[0]:
        return False

    for s in range(0, k - 2):
        alpha.append(int(hashlib.sha256(np.array([1, 1, 1])).hexdigest(), 16) % n)  # todo check
        q_alpha_s = np.polyval(tau_t[s+1], alpha[s]) % p
        part_sum = functools.reduce(lambda x, y: (x + np.polyval(tau_t[s+2], y)) % p, range(n), 0) % p
        if q_alpha_s != part_sum:
            return False

    alpha.append(int(hashlib.sha256(np.array([1, 1, 1])).hexdigest(), 16) % n)
    q_real_coeff = get_q_s(y_t, n, k, d, p, k - 1, alpha) % p
    end_sum = functools.reduce(lambda x, y: (x + np.polyval(q_real_coeff, y)) % p, range(n), 0) % p
    q_sum_solver = np.polyval(tau_t[-1], alpha[-1]) % p
    if q_sum_solver != end_sum:
        return False

    return True


def make_x_vector(n, k, d):
    return random.randint(0, 2, n*k*d)

# s-номер множества , l - номер коодинаты в векторе y - входной вектор
def get_phi_s_l_polinom(y, n, k, d, p, s, l):
    Matrix_p = np.zeros((n, n))
    result_coeffs = np.zeros(n)
    for i in range(n):
        for j in range(n):
            Matrix_p[i][j] = (i ** (n-j-1)) % p
    #coeffs - коэффициенты выходного полинома phi_s_l
    coeffs = Matrix(Matrix_p).inv_mod(p) * Matrix(y[s * n * d + l * n: s * n * d + (l + 1) * n] % p)
    for coef in coeffs:
        result_coeffs[i] = coef % p
    return result_coeffs


def get_q_s(y, n, k, d, p, s, alpha):
    coeffs_res = np.zeros((n - 1) * d + 1)
    for i in range(n ** (k - s -1)):
        coeffs = np.array([1])

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

            coeff_one = ((-1) * get_phi_s_l_polinom(y, n, k, d, p, s, l) * buf_var) % p
            coeff_one[- 1] = (coeff_one[- 1] + 1) % p

            #свертка коэффициентов (q(alpha(1),..,alpha(s-1), x , I(s+1),...,I(k))
            coeffs = np.convolve(coeffs, coeff_one) % p

        #сумма по всем выборкам (I(s+1),..,I(k))
        coeffs_res = (coeffs_res + coeffs) % p

    return coeffs_res



if __name__ == '__main__':
    n = 3
    k = 2
    d = 15
    p = 11
    x = make_x_vector(n, d, k)

    start = time.time()

    y = Gen(x, p, k, d)
    print(y)
    gen_time = time.time() - start
    print('gen_time: ', gen_time)

    z = Solve(y, n, k, d, p)
    print(z)
    solve_time = time.time() - gen_time - start
    print('solve_time: ',  solve_time)

    print(Verify(y, z, n, k, d, p))
    ver_time = time.time() - gen_time - solve_time - start
    print('ver_time: ',  ver_time)
