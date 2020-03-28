# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from multiprocessing import Pool
import os

d = datetime.now()
dirname = '/ext1/chaos/2017/ACS_zaraba/'

TMAX = 5000
NMAX = 30

NPMAX = 1000

PR = 0.2
NO = 0.02


def run_auc_n(nois):
    print("n = {:.3f}".format(nois))
    np.random.seed(23497342)
    result = auction(PR, nois)
    return result


def run_auc_p(prio):
    print("p = {:.3f}".format(prio))
    np.random.seed(23497342)
    result = auction(prio, NO)
    return result


def auction(prio, nois):
    v_buy = np.random.randint(1, 11, NMAX)
    v_sel = np.random.randint(1, 11, NMAX)

    price_buy = np.random.randint(0, 5, NMAX)
    price_sel = np.random.randint(10, 20, NMAX)

    results = np.array(([[], [], [], [], [], [], [],[],[]]))

    prev_max_price = 0
    prev_ave_price = 0

    for t in range(TMAX):

        success_buy = np.zeros(NMAX)
        success_sel = np.zeros(NMAX)

        bid = np.array([price_buy + np.trunc(v_buy * np.random.rand(NMAX)) for i in range(NMAX)])
        ask = np.array([price_sel for i in range(NMAX)]).T
        brd = ((bid >= ask) * (np.random.rand(NMAX, NMAX) <= prio) * bid)
        price = np.max(brd, axis=1)

        for i in range(NMAX):
            if price[i] != 0:
                kaitori = np.random.choice(np.where(brd[i] == price[i])[0])
                success_buy[kaitori] += 1
                success_sel[i] = 1

        num_sel = np.size(np.nonzero(success_sel == 1))
        num_buy = np.size(np.nonzero(success_buy > 1))
        max_price = np.max(price)
        ave_price = np.mean(price)
        diff_max_price = max_price - prev_max_price
        prev_max_price = max_price
        diff_ave_price = ave_price - prev_ave_price
        prev_ave_price = ave_price

        results = np.hstack((results, [[prio], [nois], [t], [num_sel], [num_buy], [max_price], [diff_max_price],
                                       [ave_price], [diff_ave_price]]))

        test_price = price_sel + (success_sel == 1) * v_sel - (success_sel == 0) * v_sel
        if np.size(np.nonzero(success_sel == 1)) != 0 and np.size(np.nonzero(success_sel == 0)) != 0:
            min_failure = np.min(price_sel[success_sel == 1])
            v_sel_1 = (test_price >= min_failure) * (success_sel == 1) * (min_failure - price_sel)
            v_sel_1[v_sel_1 < 1] = 1
            max_success = np.max(price_sel[success_sel == 0])
            v_sel_2 = (test_price <= max_success) * (success_sel == 0) * (price_sel - max_success)
            v_sel_2[v_sel_2 < 1] = 1
            v_sel = v_sel * ((v_sel_1 + v_sel_2) == 0) + v_sel_1 + v_sel_2
            test_price = price_sel + (success_sel == 1) * v_sel - (success_sel == 0) * v_sel
        price_sel = (1 - nois * np.random.rand(NMAX)) * test_price

        test_price = price_buy - (success_buy > 0) * v_buy + (success_buy == 0) * v_buy
        if np.size(np.nonzero(success_buy > 0)) != 0 and np.size(np.nonzero(success_buy == 0)) != 0:
            max_failure = np.max(bid[np.random.randint(0, NMAX)][success_buy == 0])
            v_buy_1 = (test_price <= max_failure) * (success_buy > 0) * (max_failure - price_buy)
            v_buy_1[v_buy_1 < 1] = 1
            min_success = np.min(bid[np.random.randint(0, NMAX)][success_buy > 0])
            v_buy_2 = (test_price >= min_success) * (success_buy == 0) * (price_buy - min_success)
            v_buy_2[v_buy_2 < 1] = 1
            v_buy = v_buy * ((v_buy_1 + v_buy_2) == 0) + v_buy_1 + v_buy_2
            test_price = price_buy - (success_buy > 0) * v_buy + (success_buy == 0) * v_buy
        price_buy = (1 - nois * np.random.rand(NMAX)) * test_price

    return results


def plot_time_series():
    result = auction(PR, NO)
    print(result)

    fig = plt.figure()

    # サブプロットを追加
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    ax1.plot(result[0], result[1])
    ax2.plot(result[0], result[2])
    ax3.plot(result[0], result[3])
    ax4.plot(result[0], result[4])

    plt.show()


def bifurcation_map(dirname, flag):

    p = Pool(10)

    fig1 = plt.figure()
    bx1 = fig1.add_subplot(2, 1, 1)
    bx2 = fig1.add_subplot(2, 1, 2)

    #result_all = np.array([[], [], [], [], [], [], []])

    if flag == "n":
        i = np.arange(NPMAX+1, dtype=np.float64)
        i /= NPMAX
        result = p.map(run_auc_n, i)
        result_all = np.hstack(result)
        #result_all = result_sum[:, np.where(result_sum[2] > 1000)]
        bx1.plot(result_all[1], result_all[5], "k.")
        bx2.plot(result_all[1], result_all[6], "k.")
        np.save('{:s}/bf_p_{:.3f}.npy'.format(dirname, PR), result_all)
        np.savetxt('{:s}/bf_p_{:.3f}.dat'.format(dirname, PR), result_all.T, delimiter=" ")
        plt.savefig('{:s}/bf_p_{:.3f}.eps'.format(dirname, PR))
        plt.show()

    elif flag == "p":
        i = np.arange(NPMAX+1, dtype=np.float64)
        i /= NPMAX
        result = p.map(run_auc_p, i)
        result_all = np.hstack(result)
        #result_all = result_sum[:, np.where(result_sum[2] > 1000)]
        bx1.plot(result_all[0], result_all[5], "k.")
        bx2.plot(result_all[0], result_all[6], "k.")
        np.save('{:s}/bf_n_{:.3f}.npy'.format(dirname, NO), result_all)
        np.savetxt('{:s}/bf_p_{:.3f}.dat'.format(dirname, PR), result_all.T, delimiter=" ")
        plt.savefig('{:s}/bf_n_{:.3f}.eps'.format(dirname, NO))
        plt.show()

    else:
        print("Error: Invalid flag")
        exit(-1)


def time_series(dirname):

    result = auction(PR, NO)
    np.save('{:s}/ts_n={:.3f}_p={:.3f}.npy'.format(dirname, NO, PR), result)


def num_bursts_n(nois):

    np.random.seed(23497342)

    max_burst_ave = np.zeros(100)
    num_burst_ave = np.zeros(100)
    max_burst_max = np.zeros(100)
    num_burst_max = np.zeros(100)

    for i in range(100):
        result = auction(PR, nois)
        max_burst_max[i] = np.max(result[5])
        num_burst_max[i] = np.size(np.where(np.absolute(result[6]) > 100))
        max_burst_ave[i] = np.max(result[7])
        num_burst_ave[i] = np.size(np.where(np.absolute(result[8]) > 200))

    output = np.array([[PR], [nois], [np.max(max_burst_max)], [np.sum(num_burst_max)],
                       [np.max(max_burst_ave)], np.sum(num_burst_ave)])
    return output


def num_bursts_p(prio):

    print(prio)

    np.random.seed(23497342)

    max_burst_ave = np.zeros(100)
    num_burst_ave = np.zeros(100)
    max_burst_max = np.zeros(100)
    num_burst_max = np.zeros(100)

    for i in range(10):
        result = auction(prio, NO)
        max_burst_max[i] = np.max(result[5])
        num_burst_max[i] = np.size(np.where(np.absolute(result[6]) > 100))
        max_burst_ave[i] = np.max(result[7])
        num_burst_ave[i] = np.size(np.where(np.absolute(result[8]) > 200))

    output = np.array([[prio], [NO], [np.max(max_burst_max)], [np.sum(num_burst_max)],
                       [np.max(max_burst_ave)], [np.sum(num_burst_ave)]])
    return output


def bursts_map(flag):

    p = Pool(10)

    if flag == "n":
        i = np.arange(NPMAX+1, dtype=np.float64)
        i /= NPMAX
        result = p.map(num_bursts_n, i)
        result_all = np.hstack(result)
        np.save('{:s}/nb_n_p={:.3f}.npy'.format(dirname, PR), result_all)
        # np.savetxt('{:s}/nb_n_p={:.3f}.dat'.format(dirname, PR), result_all.T, delimiter=" ")

    elif flag == "p":
        i = np.arange(NPMAX+1, dtype=np.float64)
        i /= NPMAX
        result = p.map(num_bursts_p, i)
        result_all = np.hstack(result)
        #result_all = result_sum[:, np.where(result_sum[2] > 1000)]
        np.save('{:s}/nb_p_n={:.3f}.npy'.format(dirname, NO), result_all)
        # np.savetxt('{:s}/nb_p_n={:.3f}.dat'.format(dirname, PR), result_all.T, delimiter=" ")

    else:
        print("Error: Invalid flag")
        exit(-1)


if __name__ == '__main__':

    # os.mkdir(dirname)
    time_series(dirname)
    #bifurcation_map(dirname, "p")
    # bursts_map("p")
