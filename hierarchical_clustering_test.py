import os
import random
import pickle
import math
from pathlib import Path

import pandas as pd
import numpy as np
from numpy import trapz
import torch
from torch import nn
import matplotlib
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import single, complete, average, ward, dendrogram

import sktime
import sktime.datasets
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance,  TimeSeriesResampler
from tslearn.utils import to_time_series_dataset

from dtw import dtw
from dtw import accelerated_dtw as a_dtw
from dtaidistance import dtw as dta_dtw #dtw.distance(subItem1, subItem2, psi=1)
from dtaidistance import dtw_ndim


plt.style.use('bmh')
matplotlib.rcParams.update({'font.size': 7})

import warnings
warnings.filterwarnings("error")


# goal is to preprocess the data by "normalizing" it by subtracting self from sector cluster (matching same time of occurance).  After that, the normalzied data can be shuffled and given for training.   Next, testing.  testing is the latest partial time segment.  i do what i did before: get a sector latest time segment, create a cluster, subtract.  subtract every stock. then carry on with testing.

def get_norm_market_cap(mc):
    with open("market_caps" + ".pkl", "rb") as f:
        market_caps = pickle.load(f)

    x = np.array([[mc[0, 0].item()] + market_caps])
    norm_mc = TimeSeriesScalerMeanVariance().fit_transform(x)[0, 0, 0].item()
    return np.ones(mc.shape)*norm_mc


def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    ma = ret[n - 1:] / n
    add_on = np.ones(n-1)*ma[0].item()
    #add_on = np.ones(n-1)*0
    ma = np.concatenate([add_on, ma])
    return ma


def times(arr, n, thres):
    current_level = 0
    previous_level = 0
    intersections = []
    invest_times = []
    dont_invest_times = []
    hop = []
    ma = moving_average(arr[:,0], n)

    # iteration 0
    hop += [[0, arr[0]]]
    current_level = arr[0,:]

    # Iterate over the array
    for i in range(1, len(arr)):

        previous_level = current_level
        current_level = arr[i,:]

        # Condition to check that the graph crosses the origin.
        if ((previous_level[0] < ma[i] and current_level[1] >= ma[i])
            or (previous_level[1] > ma[i] and current_level[0] <= ma[i])):

            # it crossed over. check if hop reached threshold. reset hop
            found = False
            for j, (min_i, min_v) in enumerate(hop):
                for max_i, max_v in hop[::-1][:-j]:
                    if min_v[0] < max_v[1] and abs(max_v[1] - min_v[0]) > thres:
                        found = True
                        break
                else:
                    continue
                break

            if found:
                invest_times += [[list(range(min_i, max_i+1)), arr[min_i:max_i+1]]]
            else:
                x = [j for j, arr_val in hop]
                y = arr[x[0]:x[-1]+1]
                dont_invest_times += [[x, y]]

            intersections += [[i, ma[i]]]
            hop = []
        hop += [[i, arr[i]]]

    return intersections, invest_times, dont_invest_times

# what i want is number of crosses where the condition of min or max point is at a certain threshold.  need to collect set of points since last count.


def calc_beta(stock, market):
    covariance = np.cov(stock, market)
    beta = covariance[0,1]/covariance[1,1]
    return beta

def cap_data_at_n_stds(data):
    for dim in range(data.shape[2]):
        an_array = data[0, :, dim]
        standard_deviation = np.std(an_array)
        max_deviations = 7.5
        max_val = max_deviations * standard_deviation
        min_val = -max_deviations * standard_deviation
        an_array = np.where(an_array > max_val, max_val, an_array)
        an_array = np.where(an_array < min_val, min_val, an_array)
        data[0, :, dim] = an_array
    return data


def get_data(how_far_back, req_interval_dur, sector,
             start_idx, stop_idx, X_dict, include_test_syms="everything"):
    # parameters
    seq_len = stop_idx - start_idx

    # load data and concat tensor files
    data_folder = Path("tensor_data/")
    X = []

    # load all with matching how_far_back req_interval_dur
    for root, dirs, files in os.walk(data_folder):
        for file_n in files:
            ti = "_".join([str(how_far_back), str(req_interval_dur)])
            sym = file_n.split("_")[1]
            if include_test_syms != "everything":
                if sym not in include_test_syms:
                    continue
            if ti in file_n:

                inp = torch.load(os.path.join(root, file_n))

                if inp[0,0,4].item() == sector:
                    inp = inp[:, :, :4]
                    inp = inp.float()
                    batch_d, seq_d, inp_d = inp.size()

                    x = inp.numpy()

                    if x[:, start_idx:stop_idx].shape[1] == stop_idx - start_idx:
                        std = np.std(x[:, start_idx:stop_idx, 0])
                        mean = np.mean(x[:, start_idx:stop_idx, 0])
                        cv = std / mean
                        x[:, start_idx:stop_idx, :3] = TimeSeriesScalerMeanVariance(std=1.0).fit_transform(x[:, start_idx:stop_idx, :3])
                        x[:, start_idx:stop_idx, :3] = cap_data_at_n_stds(x[:, start_idx:stop_idx, :3])
                        x[:, :, 3] = get_norm_market_cap(x[:, :, 3])

                        # moola test
                        #x[:, :, 3] = (x[:, :, 1] - x[:, :, 0]) / mean

                        new_x = x[:, start_idx:stop_idx]
                        #new_x = new_x  - new_x[0, 0, 0]
                        X += [new_x]
                        X_dict[sym] = [cv, new_x]
                    else:
                        print(file_n, "size too small:", x[:, start_idx:stop_idx].shape[1],
                              "!=", stop_idx - start_idx)
    try:
        X = np.concatenate(X, axis=0)
    except:
        import pdb ; pdb.set_trace()
    batch_d, seq_d, inp_d = X.shape
    #batch_idxs = random.sample(range(batch_d), min(75000, batch_d))
    print("collected", batch_d) #, "choosing", len(batch_idxs), "random batch size")
    #X = X[batch_idxs]
    import pdb ; pdb.set_trace()
    return X, X_dict




def stretch(x, path):
    return x[path]

def shrink(stretched_x, path):
    _, unique_idxs = np.unique(path, return_index=True)
    return stretched_x[unique_idxs]



def get_tensors(how_far_back, req_interval_dur, seq_len, how_far_back_in_file,
                include_test_syms="everything"):

    #how_far_back_in_file = 3536  #full

    if how_far_back_in_file <= seq_len:  #test and val data
        stop_idxs = [-1] #[3536]
        start_idxs = [-how_far_back_in_file] #[3536-how_far_back_in_file]
    else:                               # train data
        start_idxs = list(range(how_far_back_in_file, 0, -seq_len))[1:]
        stop_idxs = list(range(how_far_back_in_file, 0, -seq_len))[:-1]

    Xs = []
    X_dict = {}
    resamp_sz = None
    barylines = []
    for sector in range(11):
        for idx in range(len(start_idxs)):
            start_idx = start_idxs[idx]
            stop_idx = stop_idxs[idx]

            X, X_dict = get_data(how_far_back, req_interval_dur, sector,
                                 start_idx, stop_idx, X_dict, include_test_syms)
            print(sector, start_idx, stop_idx, X.shape)

            # create cluster baryline
            km = TimeSeriesKMeans(n_clusters=1, metric="euclidean")  #"dtw")
            cluster_baryline = km.fit(X[:, :, [0]]).cluster_centers_[0]
            barylines += [cluster_baryline[:, 0]]

            print("doing dtw calc")

            new_xs_stretch = []
            new_ys_stretch = []
            new_xs = []
            for ts in X:
                x = ts ; y = cluster_baryline
                #_, wps = dtw_ndim.warping_paths(x, y)
                #paths = dta_dtw.best_path(wps)
                #path_x = [i[0] for i in paths]
                #path_y = [i[1] for i in paths]
                new_x_stretch = x #stretch(x, path_x)
                new_y_stretch = y #stretch(y, path_y)
                new_x = new_x_stretch - new_y_stretch
                new_xs_stretch += [new_x_stretch]
                new_ys_stretch += [new_y_stretch]
                new_xs += [new_x]

            print("plot stuff")
            #plot_stuff(X, cluster_baryline, new_xs_stretch, new_ys_stretch, new_xs)

            #X = to_time_series_dataset(new_xs)
            #if resamp_sz == None:
            #    resamp_sz = int(np.array([i.shape[0] for i in new_xs]).mean())
            #resampled_X = TimeSeriesResampler(sz=resamp_sz).fit_transform(X)
            #Xs += [resampled_X]
            Xs += [X]

    #for baryline in barylines:
    #    plt.plot(baryline)
    #plt.show()

    X_train = np.concatenate(Xs, axis=0)
    print("X_train:", X_train.shape)
    #plot_stuff(X_train, cluster_baryline, new_xs_stretch, new_ys_stretch, new_xs)

    best_stocks = []
    for n in [35]: #[30, 35, 40, 45, 50, 55]:
        for hop_thres in [.40]: #[.36, .40, .44, .48, .52, .56]:
            print("n:", n, "hop_thres:", hop_thres, "...")
            sym, count, cv = plot_crosses(X_dict, n, hop_thres)
            score = count*cv*hop_thres
            best_stocks += [[n, sym, count, cv, hop_thres, score]]
            print("  ", sym, "green count:", count, "cv:", str(cv*100)[:5],
              "hop_thres:", hop_thres, "score:", score)
    best_stocks = sorted(best_stocks, key=lambda x:x[5], reverse=True)
    print("\nbest stocks:")
    for i in range(4):
        n, sym, count, cv, hop_thres, score = best_stocks[i]
        print("  n", n, "hop_thres:", hop_thres, sym, "green count:", count,
              "cv:", str(cv*100)[:5], "score:", score)

    #plot_crosses(X_dict)
    #plot_most_green(X_dict)

    return X_train, X_dict

def plot_crosses(X_dict,  n, hop_thres):
    # crosses
    #n = 50
    #hop_thres = .01
    cv_thres = .1
    plot_count = 10

    best_stocks = []
    for i, (sym, (cv, arr)) in enumerate(X_dict.items()):
        if cv > cv_thres:
            intersections, invest_times, dont_invest_times = times(arr[0,:,:2], n, hop_thres)
            count = len(invest_times)
            if count > 20 or best_stocks == []:
                score = count*cv*hop_thres
                best_stocks += [[i, sym, count, cv, intersections,
                                 invest_times, dont_invest_times, score, arr]]
        if len(best_stocks) > 4000:
            break

    print("   len(best_stocks) to meet cv_thres and count criteria:", len(best_stocks))
    best_stocks = sorted(best_stocks, key=lambda x:x[7])[::-1][:plot_count]

    plt.figure(figsize=(30,10)) #46))
    for idx, (i, sym, count, cv, inters, inv_ts, dinv_ts, scr, arr) in enumerate(best_stocks):
        ax = plt.subplot(plot_count,1,idx+1)
        plt.plot(arr[0,:,0], color="k", alpha=.8, linewidth=.3)
        plt.plot(arr[0,:,1], color="k", alpha=.8, linewidth=.3)
        #plt.plot(np.zeros((arr.shape[0])), color="r", linewidth=1)
        plt.plot(moving_average(arr[0,:,0], n), color="r", linewidth=.1)
        #for x, y in dinv_ts:
        #    plt.plot(x, y[:,0], 'm', alpha=.5, markersize=.5)
        for x, y in inv_ts:
            plt.plot(x, y[:,1], 'g', linewidth=.5)
        for x, y in inters:
            plt.plot(x,y, 'mo', markersize=.3)
        plt.title(sym + "  " + str(cv*100)[:5] + "  " + str(count))
        ax.get_xaxis().set_ticks([])
    plt.savefig('image n' + str(n) + " hop_thres" + str(hop_thres) +  '.png', bbox_inches='tight',pad_inches = 0, dpi=600)
    #plt.show()
    plt.close()
    i, sym, count, cv, inters, inv_ts, dinv_ts, scr, arr = best_stocks[0]
    return sym, count, cv


def plot_most_green(X_dict):
    best_stocks = []
    cv_thres = .05
    plot_count = 5

    for i, (sym, (cv, arr)) in enumerate(X_dict.items()):
        if cv > cv_thres:
            moola = sum(arr[0,:, 3])
        best_stocks += [[i, sym, moola, cv, arr[0,:,:2]]]

    best_stocks = sorted(best_stocks, key=lambda x:x[2])[::-1][:plot_count]

    plt.figure(figsize=(27,16))
    for idx, (i, sym, moola, cv, arr) in enumerate(best_stocks):
        ax = plt.subplot(plot_count,1,idx+1)
        plt.plot(arr[:,0], color="r", alpha=.8, linewidth=1)
        plt.plot(arr[:,1], color="g", alpha=.8, linewidth=1)
        plt.title(sym + "  " + str(cv*100)[:5] + "  " + str(moola))

    plt.savefig('moola.png', bbox_inches='tight',pad_inches = 0, dpi=300)
    plt.show()



def plot_stuff(data, cluster_baryline, new_xs_stretch, new_ys_stretch, new_xs):
    cm = np.corrcoef(data[:, :, 0])

    # kurtosis
    from scipy.stats import kurtosis
    best_idx_to_explore = [[i, kurtosis(arr)] for i, arr in enumerate(data[:, :, 0])]
    best_idx_to_explore = sorted(best_idx_to_explore, key=lambda x:x[1])[:50]
    best_idx_to_explore = [i for (i, j) in best_idx_to_explore]
    best_ijk = best_idx_to_explore

    for i in best_ijk:
        plt.figure(figsize=(18, 6))
        plt.subplot(3,1,1)
        plt.plot(data[i, :, 0])
        plt.plot(cluster_baryline, color="r", linewidth=1)
        plt.subplot(3,1,2)
        matplotlib.pyplot.hist(data[i, :, 0])
        plt.subplot(3,1,3)
        matplotlib.pyplot.hist(cluster_baryline, color="r")
        plt.show()


    # correlation
    """
    best_idx_to_explore = [[i, sum(arr)] for i, arr in enumerate(cm)]
    best_idx_to_explore = sorted(best_idx_to_explore, key=lambda x:x[1])[:500]
    best_idx_to_explore = [i for (i, j) in best_idx_to_explore]
    n_ts = cm.shape[0]
    min_total_corr = -1
    best_ijk = []
    for ii, i in enumerate(best_idx_to_explore):
        print(ii, "of", len(best_idx_to_explore))
        for jj, j in enumerate(best_idx_to_explore[ii+1:]):
            if cm[i, j] > -.1:
                continue
            for kk, k in enumerate(best_idx_to_explore[ii+jj+1:]):
                if cm[i, k] > -.1 or cm[j, k] > -.1:
                    continue
                for m in best_idx_to_explore[ii+jj+kk+1:]:
                    total_corr = cm[i, j] + cm[i, k] + cm[i, m] + cm[j, k] + cm[j, m] + cm[k, m]
                    if total_corr < min_total_corr:
                        print(cm[i, j], cm[i, k], cm[i, m], cm[j, k], cm[j, m], cm[k, m])
                        print()
                        min_total_corr = total_corr
                        best_ijk = [i, j, k, m]
    """

    plt.figure(figsize=(22, 8))
    plt.subplot(4, 1, 1)
    #betas = [[i, calc_beta(ts[:, 0], cluster_baryline[:, 0])] for i, ts in enumerate(data)]
    #betas = sorted(betas, key=lambda x:x[1])
    #print("low beta", betas[0][1])
    #print("high beta", betas[-1][1])
    for i in best_ijk:
        plt.plot(data[i, :, 0], alpha=.3)
    #plt.plot(data[betas[-1][0]][:, 0], color="g", alpha=.8)
    plt.plot(cluster_baryline, color="b", linewidth=2.5)
    plt.title("original data and euclid baryline")
    #plt.ylim(-3, 4)
    plt.tick_params(labelright=True)

    plt.subplot(4, 1, 2)
    for ts in data:
        beta = calc_beta(ts[:, 0], cluster_baryline[:, 0])
        if beta < .6:
            ts = ts[:, 0] - cluster_baryline[:, 0]
            plt.plot(ts, alpha=.2)
    plt.title("original data - euclid baryline")

    plt.subplot(4, 1, 3)
    for ts in data:
        beta = calc_beta(ts[:, 0], cluster_baryline[:, 0])
        if beta > 1.3:
            plt.plot(ts[:, 0], alpha=.2)
    plt.plot(cluster_baryline, color="b", linewidth=2.5)
    plt.title("original data and euclid baryline")
    #plt.ylim(-3, 4)
    plt.tick_params(labelright=True)

    plt.subplot(4, 1, 4)
    for ts in data:
        beta = calc_beta(ts[:, 0], cluster_baryline[:, 0])
        if beta > 1.3:
            ts = ts[:, 0] - cluster_baryline[:, 0]
            plt.plot(ts, alpha=.2)
    plt.title("original data - euclid baryline")



    """
    plt.subplot(4, 1, 3)
    for idx in range(len(new_xs_stretch)):
        plt.plot(new_xs_stretch[idx][:, 0], color="r", alpha=.2)
        plt.plot(new_ys_stretch[idx][:, 0], color="b", alpha=.2)
    plt.title("stretched data and stretched euclid baryline")

    plt.subplot(4, 1, 4)
    for ts in new_xs:
        plt.plot(ts[:, 0], color="r", alpha=.2)
    plt.title("stretched data - stretched euclid baryline ")
    """
    plt.show()


if __name__ == "__main__":

    how_far_back = 126144000
    req_interval_dur = 1
    seq_len = 3500

    X_train, X_dict = get_tensors(how_far_back, req_interval_dur, seq_len, seq_len)

    name = "hierarchical_train_X"
    with open(name + ".pkl", "wb") as f:
        pickle.dump(X_train, f, protocol=pickle.HIGHEST_PROTOCOL)



