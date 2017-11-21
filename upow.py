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
    gen = np.zeros((k * d + 1) * len(x))
    r = np.zeros(len(x) * (k * d + 1))
    for i in range(len(x)):
        r[i] = random.randint(0, p)
    for t in range(1, k * d + 2):
        for i in range(len(x)):
            gen[(t-1) * len(x) + i] = t * r[i]
    return gen


# todo: implement
def Verify(y, n, k, d, p, tau):
    alpha=[]
    q_s_pol_coeffs_quantity = (n - 1) * d + 1
    x = np.zeros(q_s_pol_coeffs_quantity)
    gOV=0
    for t in range(1, k*d + 2):
        gOV=0
        y_t = y[(t-1) * n * k * d: (t) * n * k * d]
        for i in range (n):
            for j in range(q_s_pol_coeffs_quantity):
                x[j] = (i ** (q_s_pol_coeffs_quantity - j-1)) % p
            gOV=gOV+sum(tau[t*k+1-k]*x)
        print("eeeeeeeeeeeeeeeeee")
        if (gOV%p!=tau[t*k-k]):
            print("ededede")
            return "1reject"

        print("k-2 = ", k-2)
        for s in range(0, k-2):
            gOV = 0
            # todo: check
            print("blyaaaaaaaa")
            alpha.append(int(hashlib.sha256(1).hexdigest(), 16) % n)# todo check
            for j in range(q_s_pol_coeffs_quantity):
                x[j] = (alpha[s] ** (q_s_pol_coeffs_quantity - j - 1)) % p
            q_alpha_s=sum(tau[t*k+s+1-k]*x)
            for i in range(n):
                for j in range(q_s_pol_coeffs_quantity):
                    x[j] = (i ** (q_s_pol_coeffs_quantity - j-1)) % p
                gOV = gOV + sum(tau[t * k + s + 2 - k  ] * x)
            if (q_alpha_s!=gOV):
                return "2reject"
                # todo: check
        gOV=0
        alpha.append(int(hashlib.sha256(np.array([1, 1, 1])).hexdigest(), 16) % n)
        print("alpha = ", alpha)
        coeffs_real=np.zeros((n-1) * d + 1)
        q_real_coeff=get_q_s(y_t, n, k, d, p, k-1, alpha)
        for i in range(n):
            for j in range(q_s_pol_coeffs_quantity):
                x[j] = (i ** (q_s_pol_coeffs_quantity - j-1)) % p
            gOV=gOV+sum(q_real_coeff*x)
        for j in range(q_s_pol_coeffs_quantity):
            x[j] = (alpha[len(alpha)-1] ** (q_s_pol_coeffs_quantity - j-1)) % p
        q_sum_solver=sum(tau[t * k - 1] * x)
        print("q_sum_solver % p = ", q_sum_solver % p)
        print("gOV % p = ", gOV % p)
        q_sum_solver = q_sum_solver %p
        gOV = gOV % p
        if (q_sum_solver != gOV):
            return "3reject"
    return "accept"






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
    for i in range(n ** (k - s -1)):
        coeffs = np.array([1])

        for l in range(d):
            buf_var = 1
            coord_number_l = l * n
            #подсчет произведения phi l-ых при постоянных alpha_1..alpha_s_minus_1
            for j in range(s + 1, k):
                set_number_j = n * j * d
                vector_number_in_j_set = int(i / (n ** (j - s - 1 )))
                buf_var = buf_var * y[set_number_j + coord_number_l + vector_number_in_j_set % n]
            for e in range(s):
                set_number_e = n * e * d
                buf_var = buf_var * y[set_number_e + coord_number_l + alpha[e]]

            coeff_one = (-1) * get_phi_s_l_polinom(y, n, k, d, p, s, l) * buf_var
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
        y_t = y[(t-1) * n * k * d : (t) * n * k * d]
        q_1_s = get_q_s(y_t, n, k, d, p, 0, [])
        for i in range(n):
            for j in range(q_s_pol_coeffs_quantity):
                x[j] = (i **(q_s_pol_coeffs_quantity-j-1)) % p
            gOV[t-1] = (gOV[t-1] + sum(q_1_s * x)) % p
        tau.append(gOV[t-1])
        tau.append(q_1_s)
        for s in range(1, k - 1):
            # todo:  check
            alpha.append(int(hashlib.sha256(np.array([1,1,1])).hexdigest(), 16) % n)  #### hash
            print(alpha, "solve")
            # print(alpha, "alpha")
            tau.append(get_q_s(y_t, n, k, d, p, s, alpha))
    print("tau[1] = ", tau[1])
    return tau


n = 3
k = 2
d = 2
p = 11
x = make_x_vector(n, d, k)
# x=list(np.linspace(1, 12, 12))
y = Gen(x, p, k, d)
print("y = ", y, "------------\n")
a = Verify(y, n, k, d, p, tau=Solve(y, n, k, d, p))
print(a)
#print("Result of Solver([10101,1,2,3,100,5,10,11,12,13,14,15,101,2,3,3,7110,58,101,11,102,13,14,15],2,3,2,7) =\n", Solver([10101,1,2,3,100,5,10,11,12,13,14,15,101,2,3,3,7110,58,101,11,102,13,14,15],2,3,2,7))
