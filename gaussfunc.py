# -*- coding: utf-8 -*-
import math
import matplotlib.pyplot as plt
import numpy as np
import pywt
import pywavelet as pw


def gauss_function(para):  # H 幅值, T 保留时间, e 宽度参数, p 为左右峰宽比值
    H = para[0]
    T = para[1]
    ea = para[2]
    eb = para[3]

    def inner_func(t_list):        # t 为输入的时间序列
        i = 0
        while t_list[i] < T:
            i += 1

        t1 = t_list[:i]
        t2 = t_list[i:]

        y_list1 = H * np.exp((-1.0 / 2) * ((t1 - T) / ea) ** 2)
        y_list2 = H * np.exp((-1.0 / 2) * ((t2 - T) / eb) ** 2)

        return np.append(y_list1, y_list2)
    return inner_func


def initial_para(H1, T1, P, E, W, Q1, Rs): # P 为 β， E 为 φ
    e1a = P * H1
    e1b = e1a / E

    H2 = H1 * W * (1 - Q1) / Q1
    e2a = e1a / W
    e2b = e1b / W
    T2 = T1 + 2 * Rs * (e2a + e1b)

    return [H1, T1, e1a, e1b], [H2, T2, e2a, e2b]


def get_two_gauss(para1, para2):
    gauss_func1 = gauss_function(para1)
    gauss_func2 = gauss_function(para2)
    return gauss_func1, gauss_func2


def set_noise(time_list):
    noise_para = [400, 0, 300, 300]
    t = np.arange(-1200, 1200, 1)
    noise_func = gauss_function(noise_para)
    noise_val = noise_func(t)

    noise = []
    for i in range(time_list.size):
        rand_num = np.random.random_integers(-999, 999)
        tmp_n = noise_val[rand_num]
        noise.append(tmp_n * (1 if rand_num > 0 else -1))

    #plt.plot(time_list, noise)
    #noise = np.random.randint(-150, 150, time_list.size)
    return noise


def annota_special():
    plt.annotate("start", xy=(-1456, 0), xytext=(-1756, 500), fontsize=16)
    plt.scatter(-1456, 0, color='black')
    plt.annotate("inflection", xy=(-488, 9576), xytext=(-1300, 9576), fontsize=16)
    plt.scatter(-488, 9576, color='black')
    plt.annotate("vertex", xy=(0, 20058), xytext=(-600, 20058), fontsize=16)
    plt.scatter(0, 20058, color='black')
    plt.annotate("valley", xy=(627, 10213), xytext=(427, 9400), fontsize=16)
    plt.scatter(627, 10213, color='black')
    plt.annotate("end", xy=(2300, 0), xytext=(2100, 500), fontsize=16)
    plt.scatter(2300, 0, color='black')


def show_gaus1():
    para1 = [1000, 0, 500, 500]
    para2 = [300, 1200, 300, 200]

    func1, func2 = get_two_gauss(para1, para2)
    time_list = np.arange(-5000, 5000, 1)

    y1 = func1(time_list)
    level = 6
    cwtmatr, freqs = pywt.cwt(y1, np.arange(1, 128), wavelet='gaus1')  # 连续小波变换`

    plt.subplot(211)
    plt.plot(time_list, y1, linewidth=3)
    plt.subplot(212)
    plt.plot(time_list, cwtmatr[level] * 10 + 2000, linewidth=3)

    plt.show()


if __name__ == "__main__":
    para1, para2 = initial_para(20000, 0, 0.02, 1, 1.25, 0.6, 0.8) # H1, T1, P, E, W, Q, Rs
    print(para1, para2)
    func1, func2 = get_two_gauss(para1, para2)

    time_list = np.arange(-2000, 2500, 1)
    y1 = func1(time_list)
    y2 = func2(time_list)
    noise = set_noise(time_list)

    plt.plot(time_list, y1, linewidth=3, ls='--', label='peak1')
    plt.plot(time_list, y2, linewidth=3, ls='-.', label='peak2')
    plt.plot(time_list, np.add(y1, y2), linewidth=3, label='overlapping peak')

    plt.legend(fontsize=16)
    plt.show()






