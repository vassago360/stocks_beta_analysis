import os
import random
from pathlib import Path
import math
import pickle

import numpy
import matplotlib.pyplot as plt

import torch
from torch import nn
norm = nn.functional.normalize

from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler
from tslearn.early_classification import NonMyopicEarlyClassifier



class Early_CLF(object):
    def __init__(self, X_train, n_clusters):
        self.X_train = X_train
        self.n_clusters = n_clusters
        self.clust_it(self.X_train, self.n_clusters)
        self.non_myopic_classifer(self.X_train, self.y_train, self.n_clusters)

    def clust_it(self, X_train, n_clusters):
        # data
        batch_d, seq_d, inp_d = X_train.shape
        import pdb ; pdb.set_trace()  # confirm batch_d
        self.X_train = X_train[torch.randperm(batch_d)]

        # scale/resample data and define train and cluster
        self.km = TimeSeriesKMeans(n_clusters=n_clusters, verbose=True)
        self.y_train = self.km.fit_predict(self.X_train)
        self.barylines = self.km.cluster_centers_[:, :, 0]

    def plot_clusters(self):
        # plot cluster
        plt.figure()
        plt.title("-- Clusters --")
        sqt_n_classes = math.ceil(math.sqrt(self.n_clusters))
        for xi in range(sqt_n_classes):
            for yi in range(sqt_n_classes*xi, sqt_n_classes*(xi+1)):
                if yi < self.n_clusters - 1:
                    plt.subplot(sqt_n_classes, sqt_n_classes, yi + 1)
                    for xx in self.X_train[self.y_train == yi]:
                        plt.plot(xx[:, 0].ravel(), "k-", alpha=.2)
                    plt.plot(self.km.cluster_centers_[yi, :, 0].ravel(), "r-")
                    plt.xlim(0, self.X_train.shape[1])
                    plt.ylim(-4, 4)
        plt.show()

    def non_myopic_classifer(self, X_train, y_train, n_clusters):
        self.early_clf = NonMyopicEarlyClassifier(n_clusters=n_clusters,
                                            cost_time_parameter=5e-2,
                                            lamb=1e2,
                                            random_state=0) #1e-3
        self.early_clf.fit(X_train, y_train)

    def save(self, name):
        with open(name + ".pkl", "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)


    def predict(self, x, future_pred_len):
        preds, times = self.early_clf.predict_class_and_earliness(x)
        y_pred = preds[0]
        t = times[0]
        baryline = self.barylines[y_pred].ravel()

        if baryline[t:-future_pred_len].shape[0] > 0:
            uptrend_score = baryline[-future_pred_len:].mean() - baryline[t:-future_pred_len].mean()
        else:
            uptrend_score = baryline[-future_pred_len:].mean() - baryline[-future_pred_len]

        y = x[0, t+1:, 0].ravel()
        y_hat = self.barylines[y_pred, t+1:].ravel()
        mae_fn = nn.L1Loss()
        mae = mae_fn(torch.FloatTensor(y), torch.FloatTensor(y_hat)).item()

        return x[0, :, 0], baryline, t, uptrend_score, mae



 #####################
 ##### chopupnpackagetensors stuff

def last_segment(inp, seq_len):
    # seq_len is desired length to grab

    seq_d, batch_d, inp_d = inp.size()
    start_at = seq_d - seq_len
    inp[-seq_len:]

def break_into_segments(inp, seq_len):
    # seq_len is desired length to break each instance into

    seq_d, batch_d, inp_d = inp.size()
    num_spaces = seq_len // 4 # spacing between start sequences
    stop_at = seq_d - seq_len

    train_inp = torch.zeros(seq_len, 0, inp_d)
    for inst_idx in range(batch_d):
        for seq_idx in range(0, stop_at, num_spaces):
            new_row = inp[seq_idx:seq_idx+seq_len, inst_idx, :].unsqueeze(1)
            train_inp = torch.cat((train_inp, new_row), dim=1)

    if train_inp.size(1) > 45:
        rand_idxs = random.sample(range(train_inp.size(1)), 45)
        train_inp = train_inp[:, rand_idxs]

    return train_inp


def get_n_chopup_tensors(how_far_back, req_interval_dur,
                         desired_seq_len, future_pred_len):
    # load data and concat tensor files
    data_folder = Path("tensor_data/")

    train_inp = torch.zeros(desired_seq_len, 0, 4)
    test_data_seq_len = desired_seq_len - future_pred_len
    test_inp = torch.zeros(test_data_seq_len, 0, 4)
    test_syms = []

    # load all with matching how_far_back req_interval_dur
    for root, dirs, files in os.walk(data_folder):
        for file_n in files:
            ti = "_".join([str(how_far_back), str(req_interval_dur)])
            if ti in file_n and "unnorm" in file_n:
                inp = torch.load(os.path.join(root, file_n))

                # pre_process and initial format data
                inp = inp.float()
                inp = inp.transpose(1, 0)

                #limit train data to [:-desired_seq_len]. other stuff for validat
                inp = inp[:-desired_seq_len]

                # create train
                file_train_inp = break_into_segments(inp, desired_seq_len)
                try:
                    train_inp = torch.cat((train_inp, file_train_inp), dim=1)
                except:
                    import pdb ; pdb.set_trace()

                # create last segment (test)
                sym = file_n.split("_")[1]
                if inp.size(0) > test_data_seq_len:
                    test_inp = torch.cat((test_inp, inp[-test_data_seq_len:]), dim=1)
                    test_syms += [sym]
                else:
                    print(sym, "size too small:", inp.size(0), "<", test_data_seq_len)

    seq_d, batch_d, inp_d = train_inp.size()
    desired_batch_idxs = random.sample(range(batch_d), 75000)
    print("collected", batch_d, "choosing", len(desired_batch_idxs), "random batch size")
    train_inp = train_inp[:, desired_batch_idxs]
    return train_inp, test_inp, test_syms



def get_last_tensors(inp_train, how_far_back, req_interval_dur,
                     seq_len, include_test_syms):
    # load data and concat tensor files
    data_folder = Path("tensor_data/")
    data = {}

    # load all with matching how_far_back req_interval_dur
    for root, dirs, files in os.walk(data_folder):
        for file_n in files:
            ti = "_".join([str(how_far_back), str(req_interval_dur)])
            sym = file_n.split("_")[1]
            if ti in file_n and sym in include_test_syms:

                inp = torch.load(os.path.join(root, file_n))
                inp = inp.float()
                inp = inp.transpose(1, 0)
                market_cap = inp[-seq_len:, :, 3]

                val = market_cap[0, 0].item()
                market_cap_norm = torch.FloatTensor( [val]*inp_train.size(0) ).unsqueeze(1)
                norm_stuff = torch.cat((market_cap_norm, inp_train[:, :, 3]), dim=1)
                market_cap_norm = norm(norm_stuff, dim=1)[:seq_len, 0]

                inp = inp.transpose(0, 1)
                batch_d, seq_d, inp_d = inp.size()

                x = TimeSeriesScalerMeanVariance().fit_transform( inp[:, -seq_len:] )
                x[0, :, 3] = market_cap_norm

                if x.shape[1] >= seq_len:
                    data[sym] = x
                else:
                    print(sym, "size too small:", x.shape[1], "<", seq_len)
    return data


#####################
##### clustering, tslearn stuff

def clust_it(inp_train_val, inp_test, future_pred_len, n_clusters):
    # data
    inp_train_val = inp_train_val.transpose(0, 1)
    batch_d, seq_d, inp_d = inp_train_val.size()
    inp_train_val = inp_train_val[torch.randperm(batch_d)]
    inp_train = inp_train_val[:-30]
    inp_val = inp_train_val[-30:]
    inp_test = inp_test.transpose(0, 1)

    # scale/resample data and define train and cluster
    X_train = TimeSeriesScalerMeanVariance().fit_transform(inp_train)
    km = TimeSeriesKMeans(n_clusters=n_clusters, verbose=True)
    y_train = km.fit_predict(X_train)

    # scale and define validation
    X_val = TimeSeriesScalerMeanVariance().fit_transform(inp_val)

    # scale test and add zeros to end
    X_test = TimeSeriesScalerMeanVariance().fit_transform(inp_test)
    d0, d1, d2 = X_test.shape
    X_test = numpy.concatenate((X_test, numpy.zeros((d0, future_pred_len, d2))), axis=1)
    barylines = km.cluster_centers_[:, :, 0]

    return km, barylines, X_train, y_train, X_val, X_test


def plot_clust(X_train, y_train):
    # plot cluster
    plt.figure()
    n_classes = len(set(y_train))
    sqt_n_classes = int(math.sqrt(n_classes))
    for xi in range(sqt_n_classes):
        for yi in range(sqt_n_classes*xi, sqt_n_classes*(xi+1)):
            plt.subplot(sqt_n_classes, sqt_n_classes, yi + 1)
            for xx in X_train[y_train == yi]:
                plt.plot(xx[:, 0].ravel(), "k-", alpha=.2)
            plt.plot(km.cluster_centers_[yi, :, 0].ravel(), "r-")
            plt.xlim(0, X_train.shape[1])
            plt.ylim(-4, 4)
    plt.show()


def non_myopic_classifer(X_train, y_train, n_clusters):
    early_clf = NonMyopicEarlyClassifier(n_clusters=n_clusters,
                                        cost_time_parameter=5e-2,
                                        lamb=1e2,
                                        random_state=0) #1e-3
    early_clf.fit(X_train, y_train)
    return early_clf


def lowest_mae_runway_tsidxs(X_test, preds, times, barylines, test_syms, num_zeros_at_end, ts_idxs):
    # get predictions where the overlap ( t+1 to -num_zeros_at_end ) between
    # pred and what's already happened (the 2 lines) have lowest mae
    mae_fn = nn.L1Loss()

    mae_tsidxs = []
    for ts_idx in ts_idxs:
        time_series = X_test[ts_idx]
        t = times[ts_idx]
        y = time_series[t+1:-num_zeros_at_end, 0].ravel()

        t = times[ts_idx]
        y_pred = preds[ts_idx]
        y_hat = barylines[y_pred, t+1:-num_zeros_at_end].ravel()

        mae = mae_fn(torch.FloatTensor(y), torch.FloatTensor(y_hat)).item()
        min_len_to_compare = 6
        if y.shape[0] < min_len_to_compare:
            mae += 10
        mae_tsidxs += [[mae, ts_idx, test_syms[ts_idx]]]

    sorted_mae_tsidxs = sorted(mae_tsidxs, key=lambda x:x[0])
    tsidxs_syms = [(tsidx, sym) for (mae, tsidx, sym) in sorted_mae_tsidxs]
    return tsidxs_syms




def plot_non_myopic_prediction(X_test, preds, times, barylines,
                               test_syms, num_zeros_at_end):
    assert len(test_syms) == X_test.shape[0]

    #ts_idxs = [i for (i, t) in enumerate(times) if start_t < t < end_t]
    ts_idxs = list(range(len(times)))
    tsidxs_syms = lowest_mae_runway_tsidxs(X_test, preds, times, barylines,
                                           test_syms, num_zeros_at_end, ts_idxs)
    tsidxs_syms = tsidxs_syms[:100]

    # start plot
    plt.figure()
    plt.title("Test Set")
    sqt_n = math.ceil(math.sqrt(len(tsidxs_syms)))
    for xi in range(sqt_n):
        for yi in range(sqt_n*xi, sqt_n*(xi+1)):
            if yi < len(tsidxs_syms):
                plt.subplot(sqt_n, sqt_n, yi + 1)
                ts_idx = tsidxs_syms[yi][0]
                sym = tsidxs_syms[yi][1]
                t = times[ts_idx]

                # plot line and prediction line
                time_series = X_test[ts_idx]
                y_pred = preds[ts_idx]
                plt.plot(numpy.arange(0, time_series.shape[0] - num_zeros_at_end),
                        time_series[:-num_zeros_at_end, 0].ravel(), linewidth=1.5)
                #plt.plot(time_series[:, 0].ravel(), linewidth=1.5)
                plt.plot(numpy.arange(t+1, time_series.shape[0]),
                        barylines[y_pred, t+1:].ravel(), "r-", linestyle="dashed")
                plt.text(0.05, 1.02, sym, transform=plt.gca().transAxes)
                plt.axvline(x=t, linewidth=.5)
                plt.axvline(x=time_series.shape[0] - num_zeros_at_end, linewidth=.5)
                plt.xlim(0, time_series.shape[0] - 1)
    plt.show()
    import pdb ; pdb.set_trace()



def plot_non_myopic_prediction_validation(X_val, preds, times, barylines, future_pred_len):
    ts_idxs = list(range(len(times)))

    # start plot
    plt.figure()
    plt.title("Validation Set")
    sqt_n = math.ceil(math.sqrt(len(ts_idxs)))
    for xi in range(sqt_n):
        for yi in range(sqt_n*xi, sqt_n*(xi+1)):
            if yi < len(ts_idxs):  #don't want to try to plot something that doesn't exist
                plt.subplot(sqt_n, sqt_n, yi + 1)
                ts_idx = ts_idxs[yi]
                t = times[ts_idx]

                # plot line and prediction line
                time_series = X_val[ts_idx]
                y_pred = preds[ts_idx]
                plt.plot(time_series[:, 0].ravel(), linewidth=1.5)
                plt.plot(numpy.arange(t+1, time_series.shape[0]),
                        barylines[y_pred, t+1:].ravel(), "r-", linestyle="dashed")
                plt.axvline(x=t, linewidth=.5)
                plt.axvline(x=time_series.shape[0] - future_pred_len, linewidth=.5)
                plt.xlim(0, time_series.shape[0] - 1)
    plt.show()





if __name__ == "__main__":
    pass

