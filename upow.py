import seaborn as sns
from numpy.linalg import solve
import math
import matplotlib.pyplot as plt
import numpy as np
import pylab
import random

import scipy as stats
import scipy.special as special
from scipy.integrate import quad
import pandas as pd
import pprint
import csv
import scipy.special

from sympy import Matrix
import hashlib

sns.set()


def Gen(x, p, k, d):
    gen = []
    r = np.zeros(len(x))
    for i in range(len(r)):
        r[i] = random.randint(0, p)
    for i in range(1, k * d + 1):
        gen.append((r * i) % p + x)
    print("Result of Gen(x, p, n, k, d) =\n", gen)
    return gen


# todo: implement
def Verify(y, n, k, d, p, tau):
    return 0


def make_x_vector(n, k, d):
    v = np.zeros(n * k * d)
    for i in range(n * k * d):
        v[i] = int(random.randint(0, 1))
    return v

# s-номер множества , l - номер коодинаты в векторе y - входной вектор
def get_phi_s_l_polinom(y, n, k, d, p, s, l):
    Matrix_p = np.zeros((n, n))
    result_coeffs = np.zeros(n)
    for i in range(n):
        for j in range(n):
            Matrix_p[i][j] = (i ** j) % p
    #coeffs - коэффициенты выходного полинома phi_s_l
    coeffs = Matrix(Matrix_p).inv_mod(p) * Matrix(y[s * n * d + l * n : s * n * d + (l + 1) * n])
    for i in range(len(coeffs)):
        result_coeffs[i] = coeffs[i] % p
    return result_coeffs


def get_q_s(y, n, k, d, p, s, alpha):
    coeffs = np.array([1])
    buf_var = 1
    coeffs_res = np.zeros((n - 1) * d + 1)
    for i in range(n ** (k - s - 1)):
        coeffs = np.array([1])

        for l in range(d):
            buf_var = 1
            coord_number_l = l * n
            #подсчет произведения phi l-ых при постоянных alpha_1..alpha_s_minus_1
            for j in range(s + 1, k):
                set_number_j = n * j * d
                vector_number_in_j_set = int(i / (n ** (j - s - 1)))
                buf_var = buf_var * y[set_number_j + coord_number_l + vector_number_in_j_set % n]
            for e in range(s):
                set_number_e = n * e * d
                buf_var = buf_var * y[set_number_e + coord_number_l + alpha[e]]

            coeff_one = (-1) * get_phi_s_l_polinom(y, n, k, d, p, s, l) * buf_var % p
            coeff_one[len(coeff_one) - 1] = (coeff_one[len(coeff_one) - 1] + 1) % p
            # print(coeff_one)
            #свертка коэффициентов (q(alpha(1),..,alpha(s-1), x , I(s+1),...,I(k))
            coeffs = np.convolve(coeffs, coeff_one)
            # print(coeffs)

        #сумма по всем выборкам (I(s+1),..,I(k))
        coeffs_res = coeffs_res + coeffs

    return coeffs_res % p


def Solve(y, n, k, d, p):
    gOV = np.zeros(k * d+1)
    q_s_pol_coeffs_quantity = (n - 1) * d + 1
    alpha = []
    tau = []
    # todo: + 1  ????
    x = np.zeros(q_s_pol_coeffs_quantity)

    for t in range(1, k*d+2):
        y_t = y[t * n * k * d : (t + 1) * n * k * d]
        q_1_s = get_q_s(y_t, n, k, d, p, 0, [])
        for i in range(n):
            for j in range(q_s_pol_coeffs_quantity):
                x[j] = (i **(n-j)) % p
            gOV[t] = (gOV[t] + sum(q_1_s * x)) % p
        tau.append(gOV[t])
        tau.append(q_1_s)
        for s in range(1, k - 1):
            alpha.append(int(hashlib.sha256(sum(tau)).hexdigest(), 16) % n)  #### hash
            # print(alpha, "alpha")
            tau.append(get_q_s(y_t, n, k, d, p, s, alpha))
    return tau


# print(q_s([10101,1,2,3,100,5,10,14,12,13,14,15],3,2,2,7,0,[]))
# print("Result of Solver([10101,1,2,3,100,5,10,11,12,13,14,15,101,2,3,3,7110,58,101,11,102,13,14,15],2,3,2,7) =\n", Solver([10101,1,2,3,100,5,10,11,12,13,14,15,101,2,3,3,7110,58,101,11,102,13,14,15],2,3,2,7))
# print (vector(2, 3, 4))