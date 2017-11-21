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


def Gen(x, p, n, k, d):
    gen = []
    r = np.zeros(len(x))
    for i in range(len(r)):
        r[i] = random.randint(0, p)
    for i in range(1, k * d + 1):
        gen.append((r * i) % p + x)
    print("Result of Gen(x, p, n, k, d) =\n", gen)
    return gen


# todo: implement
def Verify(y, p, solver):
    #генерируем /alpha_s s = [1,k-2]
    alpha = random.randint(0, p)
    #отправляем на prover альфа и ждем коэффициенты
    #тут идут вызов функции
    #тут идут коэффициэнты солвера списком(мб отдельная функция)
    for i in range(len(solver)):
        x = 0
        x = x + solver[i]
        print(x)
        if (x == y):
            print('1 test OK')
        else:
            print('1 test FAILURE')
    #проверка коэффициентов на условие


def vector(n, k, d):
    v = np.zeros(n*k*d)
    for i in range(n*k*d):
        v[i] = int(random.randint(0, 1))
    return v


def Phi(y, n, k, d, p, s, l):  # s-номер множества , l - номер коодинаты в векторе y - входной вектор
    Matrix_p = np.zeros((n, n))
    coeffs_array = np.zeros(n)
    for i in range(n):
        for j in range(n):
            Matrix_p[i][j] = (i ** j) % p
    coeffs = Matrix(Matrix_p).inv_mod(p) * Matrix(y[s * n * d + l * n: s * n * d + (l + 1) * n])
    for i in range(len(coeffs)):
        coeffs_array[i] = coeffs[i] % p
    return coeffs_array


def q_s(y, n, k, d, p, s, alpha):
    coeffs = np.array([1])
    A = 1
    coeffs_res = np.zeros((n - 1) * d + 1)
    for i in range(n ** (k - s - 1)):
        coeffs = np.array([1])

        for l in range(d):
            A = 1
            for j in range(s + 1, k):
                A = A * y[n * j * d + l * n + int(i / (n ** (j - s - 1))) % n]
            for e in range(s):
                A = A * y[n * e * d + l * n + alpha[e]]

            coeff_one = Phi(y, n, k, d, p, s, l) * A * (-1) % p
            coeff_one[len(coeff_one) - 1] = (coeff_one[len(coeff_one) - 1] + 1) % p
            # print(coeff_one)
            coeffs = np.convolve(coeffs, coeff_one)
            # print(coeffs)

        coeffs_res = coeffs_res + coeffs

    return coeffs_res % p


def Solve(y, n, k, d, p):
    z = []
    gOV = np.zeros(k * d + 1)
    alpha = []
    tau = []
    x = np.zeros((n - 1) * d + 1)

    for t in range(2):
        q_1_s = q_s(y[t * n * k * d:(t + 1) * n * k * d], n, k, d, p, 0, [])
        for i in range(n):
            for j in range((n - 1) * d + 1):
                x[j] = (i ** j) % p
            gOV[t] = (gOV[t] + sum(q_1_s * x)) % p
        tau.append(gOV[t])
        tau.append(q_1_s)
        for s in range(1, k - 1):
            alpha.append(int(hashlib.sha256(sum(tau)).hexdigest(), 16) % p)  #### hash
            # print(alpha, "alpha")
            tau.append(q_s(y[t * n * k * d:(t + 1) * n * k * d], n, k, d, p, s, alpha))
    return tau


# print(q_s([10101,1,2,3,100,5,10,14,12,13,14,15],3,2,2,7,0,[]))
# print("Result of Solver([10101,1,2,3,100,5,10,11,12,13,14,15,101,2,3,3,7110,58,101,11,102,13,14,15],2,3,2,7) =\n", Solver([10101,1,2,3,100,5,10,11,12,13,14,15,101,2,3,3,7110,58,101,11,102,13,14,15],2,3,2,7))
# print (vector(2, 3, 4))