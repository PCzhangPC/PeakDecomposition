# -*- coding: utf-8 -*-
from gaussfunc import *
import numpy as np
import matplotlib.pyplot as plt
import csv


def slide_filter(val_list, win_leng=31):
    after_filt = []
    for i in range(int(win_leng / 2), int(val_list.size - win_leng + win_leng / 2 + 1)):
        win = val_list[int(i - win_leng / 2): int(i + win_leng - win_leng / 2)]
        after_filt.append(sum(win) / win_leng)

    ret = [after_filt[0]] * int(win_leng / 2)
    ret.extend(after_filt)
    ret.extend([after_filt[-1]] * int(win_leng / 2))
    return np.array(ret)


def window_derivate(val_list):
    diff_win = [-9, -7, -5, -3, -1, 1, 3, 5, 7, 9]
    deri_list = [0] * 5

    for i in range(5, val_list.size - 4):
        deri = 0
        for j in range(-5, 5):
            deri += (diff_win[j + 5] * val_list[i + j])
        deri_list.append(deri)

    deri_list.extend([0] * 4)
    print(deri_list)
    return deri_list


def cal_derivative_by_diff(val_list):
    deri_list = [0]
    for i in range(1, val_list.size):
        deri_list.append(val_list[i] - val_list[i - 1])

    return np.array(deri_list)


def find_reflection_by_diff(cwtmatr, level, time_list): # 二阶导数的过零点
    x_pos = []
    inflec_index = []

    for i in range(1, time_list.size):
        if cwtmatr[level][i] == 0 or cwtmatr[level][i] * cwtmatr[level][i - 1] < 0:

            x_pos.append(time_list[i])

    print(x_pos)


#计算拐点
def find_inflection(cwtmatr, level, time_list):
    x_pos = []
    y_pos = []

    max_dic = {}  # y是key，x是value
    min_dic = {}

    for i in range(time_list.size):
        if 1000 < i < time_list.size - 1000 and (       # 极大值
                (cwtmatr[level][i - 1] < cwtmatr[level][i] and cwtmatr[level][i + 1] < cwtmatr[level][i])):

            max_dic[cwtmatr[level][i]] = time_list[i]

        elif 1000 < i < time_list.size - 1000 and (     # 极小值
                (cwtmatr[level][i - 1] > cwtmatr[level][i] and cwtmatr[level][i + 1] > cwtmatr[level][i])):

            min_dic[cwtmatr[level][i]] = time_list[i]

    x_max_list = sorted(max_dic.keys())   # value的集合
    x_min_list = sorted(min_dic.keys())

    inflection_dict = {}
    flag = False
    if len(x_max_list) + len(x_min_list) >= 4:
        inflection_dict[max_dic[x_max_list[-1]]] = x_max_list[-1]
        inflection_dict[max_dic[x_max_list[-2]]] = x_max_list[-2]

        inflection_dict[min_dic[x_min_list[0]]] = x_min_list[0]
        inflection_dict[min_dic[x_min_list[1]]] = x_min_list[1]

        x_pos = sorted(inflection_dict.keys())
        for key in x_pos:
            y_pos.append(inflection_dict[key])
            plt.plot(key, -inflection_dict[key], 'ro')

        flag = True

    return x_pos, y_pos, flag


#由拐点计算特征数值
def cal_feature1(x_pos, y_pos):
    # 这里是缪华间的构造方式

    ta, tb, tc, td = x_pos
    ha, hb, hc, hd = y_pos

    p1 = (tb - ta) / ((tb - ta) + (td - tc))
    p2 = (tc - tb) / ((tb - ta) + (td - tc))
    p3 = ha / (ha - hd)
    p4 = hb / (ha - hd)
    p5 = hc / (ha - hd)

    return [p1, p2, p3, p4, p5]


def cal_feature2(x_pos, y_pos, time_list, y):
    # 使用熊的方式
    ta, tb, tc, td = x_pos
    ha, hb, hc, hd = y_pos

    t1_tmp = (tb + ta) / 2
    h1_tmp = solve(time_list, y, t1_tmp)
    t2_tmp = (td + tc) / 2
    h2_tmp = solve(time_list, y, t2_tmp)

    p1 = (tb - ta) / (td - tc)
    p2 = ((td + tc) - (tb + ta)) / (tb - ta + td - tc)
    p3 = h1_tmp / h2_tmp
    p4 = hb / ha
    p5 = hd / ha

    return [p1, p2, p3, p4, p5]


def cal_feature3(x_pos, y_pos):
    # 使用自己的方式
    ta, tb, tc, td = x_pos
    ha, hb, hc, hd = y_pos

    p1 = (tb - ta) / (td - tc)
    p2 = (tc - tb) / (td - ta)
    p3 = ha / (ha - hd)
    p4 = hb / (ha - hd)
    p5 = hc / (ha - hd)

    return [p1, p2, p3, p4, p5]


def generate_data_set():
    ##这里是熊的论文的生成方式
    global P_list, E_list, W_list, Q_list, Rs_list

    P_list = np.linspace(0.005, 0.025, 5)
    E_list = np.linspace(0.5, 2.0, 6)
    W_list = np.linspace(0.5, 2.0, 4) # 4 改成了8
    Q_list = np.linspace(0.1, 0.9, 9) # 9 改成了18
    Rs_list = np.linspace(0.6, 1.4, 5)


#计算 t 时间的值
def solve(t_list, y_list, t):
    index = int(t - t_list[0])
    return y_list[index]


def calculate_and_save():
    global P_list, E_list, W_list, Q_list, Rs_list

    fail_num = 0
    index = 0
    fail_para = []

    H1 = 20000
    T1 = 0
    for p in P_list:
        for e in E_list:
            for w in W_list:
                for q in Q_list:
                    for rs in Rs_list:
                        para1, para2 = initial_para(H1, T1, p, e, w, q, rs)
                        func1, func2 = get_two_gauss(para1, para2)

                        time_list = np.arange(-2000, 8000, 1)
                        y1 = func1(time_list)
                        y2 = func2(time_list)
                        peak_func_val = np.add(y1, y2)
                        cwtmatr, freqs = pywt.cwt(peak_func_val, np.arange(1, 10), wavelet='gaus1')  # 连续小波变换`
                        level = 6

                        x_pos, y_pos, flag = find_inflection(cwtmatr, level, time_list)  # 计算拐点

                        index += 1
                        print('index', index)
                        if not flag:
                            fail_num += 1
                            fail_para.append([para1, para2, [p, e, w, q, rs]])
                            print('fail_happend!!!!!!!!!!!!', fail_num)
                            continue
                            # plt.plot(peak_func_val)
                            # plt.plot(time_list, cwtmatr[level])
                            # print(p, e, w, q, rs)
                            # plt.show()

                        #做无因次比值的转换
                        p1, p2, p3, p4, p5 = cal_feature2(x_pos, y_pos, time_list, peak_func_val) #计算特征
                        feature_list = [p1.item(), p2.item(), p3.item(), p4.item(), p5.item(), q]
                        save_csv(feature_list)  # 保存到csv

                        # 不做无因次比值转换
                        # ta, tb, tc, td = x_pos
                        # ha, hb, hc, hd = y_pos
                        # feature_list = [ta, ha, tb, hb, tc, hc, td, hd, q]
                        # save_csv2(feature_list)  #保存到csv

    print('###########################')
    print('all_fail_num: ', fail_num)
    print('###########################')
    for para in fail_para:
        print(para)


def save_csv(feature_list, path = None):
    path = r'C:\Users\Administrator\Desktop\feature_value5.csv'
    out = open(path, 'a', newline='')
    csv_writer = csv.writer(out)
    csv_writer.writerow(feature_list)


def test_one(H1, T1, p, e, w, q, rs):
    H1, T1, p, e, w, q, rs = 20000, 0, 0.02, 1, 0.9, 0.5, 0.6
    para1, para2 = initial_para(H1, T1, p, e, w, q, rs)  # H1, T1, P, E, W, Q, Rs
    func1, func2 = get_two_gauss(para1, para2)

    time_list = np.arange(-2000, 8000, 1)
    y1 = func1(time_list)
    y2 = func2(time_list)

    plt.plot(time_list, y1)
    plt.plot(time_list, y2)
    y = np.add(y1, y2)
    plt.plot(time_list, y)

    s1 = cal_S_by_diff(-2000, 600, y)
    s2 = cal_S_by_diff(600, 4000, y)
    print(s1, s2, s1 / (s1 + s2))

    #level = 90
    # 做一阶导数
    #wavelet = pywt.ContinuousWavelet('mexh')
    #cwtmatr, freqs = pywt.cwt(y, np.arange(1, 100), wavelet='gaus1')  # 连续小波变换`
    #plt.plot(time_list, -cwtmatr[level])

    # x_pos, y_pos, res = find_inflection(cwtmatr, level, time_list)
    # print('x: ', x_pos)
    # print('y: ', y_pos)

    # x_pos, y_pos, res = find_reflection_by_diff(cwtmatr, level, time_list)
    # print('x: ', x_pos)
    # print('y: ', y_pos)

    #二阶导数
    # cwtmatr2, freqs2 = pywt.cwt(y, np.arange(1, 50), wavelet='gaus2')  # 连续小波变换`
    # plt.plot(time_list, -cwtmatr2[6] * 50)
    # find_reflection_by_diff(cwtmatr2, 6, time_list)
    # print()

    # wavelet = pywt.Wavelet('bior1.3')
    # ca, cd1, cd2, cd3, cd4, cd5 = pywt.wavedec(y, wavelet=wavelet, level=5) #离散变换
    # plt.plot(np.arange(-2002, 8002, 2), -cd5*50)
    #
    # # 做第二次微分
    # cwtmatr2, freqs2 = pywt.cwt(cwtmatr[level], np.arange(1, 129), wavelet='mexh')
    # plt.plot(time_list, -cwtmatr2[level])
    #



    # p1, p2, p3, p4, p5 = cal_feature1(x_pos, y_pos)
    #
    # feature_list = [p1.item(), p2.item(), p3.item(), p4.item(), p5.item(), 0.6]
    # save_csv(feature_list)

    #plt.plot(time_list, np.linspace(0, 0, 10000))
    plt.show()


def draw_diff_level(H1, T1, p, e, w, q, rs):
    para1, para2 = initial_para(H1, T1, p, e, w, q, rs)  # H1, T1, P, E, W, Q, Rs
    func1, func2 = get_two_gauss(para1, para2)

    time_list = np.arange(-2000, 3000, 1)
    y1 = func1(time_list)
    y2 = func2(time_list)

    y = np.add(y1, y2)
    plt.plot(time_list, y, label="Overlapping peak", linewidth=3)

    # 做一阶导数
    cwtmatr, freqs = pywt.cwt(y, np.arange(1, 100), wavelet='gaus1')  # 连续小波变换`
    plt.xlabel('t', fontsize=20)
    plt.ylabel('h(t)', fontsize=20)

    plt.plot(time_list, -cwtmatr[5], label="level 5", linewidth=3)
    plt.plot(time_list, -cwtmatr[20], label="level 20", linewidth=3)
    plt.plot(time_list, -cwtmatr[40], label="level 40", linewidth=3)
    plt.plot(time_list, -cwtmatr[70], label="level 70", linewidth=3)

    plt.plot(time_list, np.linspace(0, 0, 5000), color='black')

    x_pos, y_pos, res = find_inflection(cwtmatr, 70, time_list)

    lable_list = ['A', 'B', 'C', 'D']
    for index in range(4):
        plt.annotate(lable_list[index], xy=(x_pos[index], -y_pos[index]),
                     xytext=(x_pos[index] - 20, -y_pos[index] - 2200), fontsize=16)

    print()
    plt.legend(fontsize=16)
    plt.show()


def find_cal_real_pos(H1, T1, p, e, w, q, rs):
    para1, para2 = initial_para(H1, T1, p, e, w, q, rs)  # H1, T1, P, E, W, Q, Rs
    func1, func2 = get_two_gauss(para1, para2)

    time_list = np.arange(-2000, 3000, 1)
    y1 = func1(time_list)
    y2 = func2(time_list)

    y = np.add(y1, y2)
    plt.plot(time_list, y, label="Overlapping peak", linewidth=3)

    level = 90
    # 做一阶导数
    cwtmatr, freqs = pywt.cwt(y, np.arange(1, 100), wavelet='gaus1')  # 连续小波变换`
    plt.xlabel('t', fontsize=20)
    plt.ylabel('h(t)', fontsize=20)

    plt.plot(time_list, -cwtmatr[5], label="level 5", linewidth=3)
    plt.plot(time_list, -cwtmatr[20], label="level 20", linewidth=3)
    plt.plot(time_list, -cwtmatr[40], label="level 40", linewidth=3)
    plt.plot(time_list, -cwtmatr[70], label="level 70", linewidth=3, ls='--')

    plt.plot(time_list, np.linspace(0, 0, 5000), color='black')

    cwtmatr2, freqs2 = pywt.cwt(y, np.arange(1, 100), wavelet='gaus2') # 二级导数

    print('level 5')
    x_pos, y_pos, res = find_inflection(cwtmatr, 5, time_list)
    print('x: ', x_pos)
    find_reflection_by_diff(cwtmatr2, 10, time_list)

    print()
    print('level 20')
    x_pos, y_pos, res = find_inflection(cwtmatr, 20, time_list)
    print('x: ', x_pos)
    find_reflection_by_diff(cwtmatr2, 10, time_list)

    print()
    print('level 40')
    x_pos, y_pos, res = find_inflection(cwtmatr, 40, time_list)
    print('x: ', x_pos)
    find_reflection_by_diff(cwtmatr2, 10, time_list)

    print()
    print('level 70')
    x_pos, y_pos, res = find_inflection(cwtmatr, 70, time_list)
    print('x: ', x_pos)
    find_reflection_by_diff(cwtmatr2, 10, time_list)

    lable_list = ['A', 'B', 'C', 'D']
    for index in range(4):
        plt.annotate(lable_list[index], xy=(x_pos[index], -y_pos[index]),
                     xytext=(x_pos[index] - 20, -y_pos[index] - 2200), fontsize=16)
    print()

    plt.legend(fontsize=16)
    plt.show()


def cal_S_by_diff(start, end, val_list):
    s = 0.0
    for i in range(start, end + 1):
        s += val_list[i + 2000]

    return s


if __name__ == "__main__":
    find_cal_real_pos(20000, 0, 0.02, 1, 0.6, 0.7, 0.6)







