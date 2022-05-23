import numpy as np
import pandas as pd
import requests
from pathlib import Path
import os, time
import pickle

from sklearn.preprocessing import RobustScaler
import torch

import mplfinance as mpf

import numpy
import matplotlib.pyplot as plt

from tslearn.clustering import TimeSeriesKMeans
from tslearn.early_classification import NonMyopicEarlyClassifier
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, \
    TimeSeriesResampler

from utils import *
from symbols import symbols, delisted
from hierarchical_clustering_test import get_tensors


plt.rcParams.update({'font.size': 5})


# parameters
num_movers = 15
hr = 3600

### pandas extra functions ###

def roll(df, w):
    # stack df.values w-times shifted once at each stack
    roll_array = np.dstack([df.values[i:i+w, :] for i in range(len(df.index) - w + 1)]).T
    # roll_array is now a 3-D array and can be read into
    # a pandas panel object
    panel = pd.Panel(roll_array,
                     items=df.index[w-1:],
                     major_axis=df.columns,
                     minor_axis=pd.Index(range(w), name='roll'))
    # convert to dataframe and pivot + groupby
    # is now ready for any action normally performed
    # on a groupby object
    return panel.to_frame().unstack().T.groupby(level=0)


####  Query and store price data functions  ####

def created_secs_ago(filename):
    return time.time() - os.path.getmtime(filename)

def get_sector(symbol):
    secs_ago_cut_off = 360000*hr

    # attempt to load symbol from memory if exists and aren't too old
    data_folder = Path("summary_data/")
    data_folder.mkdir(parents=True, exist_ok=True)
    filename_suffix = "_".join(["", "summary"]) + ".pkl"

    sector_names = {'Basic Materials':0, 'Communication Services':1, 'Consumer Cyclical':2, 'Consumer Defensive':3, 'Energy':4, 'Financial Services':5, 'Healthcare':6, 'Industrials':7, 'Real Estate':8, 'Technology':9, 'Utilities':10, 'NA':11}
    filename = data_folder / (symbol + filename_suffix)
    if filename.exists() and created_secs_ago(filename) < secs_ago_cut_off:
        try:
            filename = data_folder / (symbol + filename_suffix)
            with open(filename, "rb") as f:
                JSONContent = pickle.load(f)
            sector = JSONContent["summaryProfile"]["sector"]
        except:
            print('error in JSON: no meta data for', symbol, 'not sure what to use.')
            sector = "NA"
    else:
        print('no JSON: meta data for', symbol, '')
        sector = "NA"
    return sector_names[sector]

def get_market_cap(symbol):
    secs_ago_cut_off = 360000*hr

    # attempt to load symbol from memory if exists and aren't too old
    data_folder = Path("summary_data/")
    data_folder.mkdir(parents=True, exist_ok=True)
    filename_suffix = "_".join(["", "summary"]) + ".pkl"

    filename = data_folder / (symbol + filename_suffix)
    if filename.exists() and created_secs_ago(filename) < secs_ago_cut_off:
        try:
            filename = data_folder / (symbol + filename_suffix)
            with open(filename, "rb") as f:
                JSONContent = pickle.load(f)
            market_cap = JSONContent["summaryDetail"]["marketCap"]["raw"]
            market_cap = math.log(market_cap)
        except:
            print('no meta data for', symbol, 'using 25')
            market_cap = 18.0
    else:
        print('no meta data for', symbol, 'using 25')
        market_cap = 18.0
    return market_cap

def save_market_caps(price_dfs):
    market_caps = []
    for sym in price_dfs.keys():
        mc = price_dfs[sym].iloc[0]['marketCap']
        if type(mc) == type( price_dfs['a'].iloc[0]['marketCap'] ):
            market_caps += [ price_dfs[sym].iloc[0]['marketCap'] ]
    with open("market_caps", "wb") as f:
        pickle.dump(market_caps, f, protocol=pickle.HIGHEST_PROTOCOL)



def get_summary(all_symbols):
    url = "https://yh-finance.p.rapidapi.com/stock/v2/get-summary"

    headers = {
    'x-rapidapi-host': "yh-finance.p.rapidapi.com",
    'x-rapidapi-key': ""
    }

    assert headers['x-rapidapi-key'] != ""

    secs_ago_cut_off = 360000*hr

    # attempt to load symbol from memory if exists and aren't too old
    data_folder = Path("summary_data/")
    data_folder.mkdir(parents=True, exist_ok=True)
    filename_suffix = "_".join(["", "summary"]) + ".pkl"
    price_dfs = {}
    loaded_symbols = []
    for symbol in all_symbols:
        filename = data_folder / (symbol + filename_suffix)
        if filename.exists() and created_secs_ago(filename) < secs_ago_cut_off:
            price_df = pd.read_pickle(filename)
            price_dfs[symbol] = price_df
            loaded_symbols.append(symbol)
    for symbol in loaded_symbols:
        all_symbols.remove(symbol)

    # chunk and query symbols
    print("in memory:", len(loaded_symbols), loaded_symbols)
    print("to query:", len(all_symbols), all_symbols)

    for symbol in all_symbols:

        querystring = {"symbol":symbol,"region":"US"}

        try:
            print("collecting data on", symbol, "...")
            response = requests.request("GET", url, headers=headers, params=querystring)
            JSONContent = response.json()
        except Exception as e:
            print(symbol, ": unable to get request or problem parsing the json data")
            print(e)
            #print(JSONContent)
            continue

        filename = data_folder / (symbol + filename_suffix)
        with open(filename, "wb") as f:
            pickle.dump(JSONContent, f, protocol=pickle.HIGHEST_PROTOCOL)





def get_charts(all_symbols, interval, rang):  #gets more data (close, volume), but 5x more queries :(
    url = "https://yh-finance.p.rapidapi.com/market/get-charts"

    headers = {
    'x-rapidapi-host': "yh-finance.p.rapidapi.com",
    'x-rapidapi-key': "898a9b8ea9msh260ac17e712cfb5p182b9bjsn0882bdf47130"
    }

    # 'x-rapidapi-key': '5bf93aef49msh25cc8b486ea4bf2p187f01jsn12bd81dfdeb0'

    if interval == "60m":
        secs_ago_cut_off = 1300*hr
    else:
        secs_ago_cut_off = 360000*hr

    # attempt to load symbol from memory if exists and aren't too old
    data_folder = Path("chart_data/")
    data_folder.mkdir(parents=True, exist_ok=True)
    filename_suffix = "_v2_".join(["", interval, rang]) + ".pkl"
    price_dfs = {}
    loaded_symbols = []
    for symbol in all_symbols:
        filename = data_folder / (symbol + filename_suffix)
        if filename.exists() and created_secs_ago(filename) < secs_ago_cut_off:
            price_df = pd.read_pickle(filename)
            price_dfs[symbol] = price_df
            loaded_symbols.append(symbol)
    for symbol in loaded_symbols:
        all_symbols.remove(symbol)

    # chunk and query symbols
    print("in memory:", len(loaded_symbols), loaded_symbols)
    print("to query:", len(all_symbols), all_symbols)

    for symbol in all_symbols:

        querystring = {"region":"US", "lang":"en", "symbol":symbol,
                       "interval":interval,"range":rang}

        try:
            print("collecting data on", symbol, "...")
            response = requests.request("GET", url, headers=headers, params=querystring)
            JSONContent = response.json()
            JSONContent["chart"]["result"][0]["indicators"]
        except Exception as e:
             print(symbol, ": unable to get request or problem parsing the json data")
             print(JSONContent)
             continue

        if 'error' not in JSONContent:
            try:
                symbol = JSONContent["chart"]["result"][0]["meta"]["symbol"].lower()
                timestamp_data = JSONContent["chart"]["result"][0]["timestamp"]
                open_data =  JSONContent["chart"]["result"][0]["indicators"]["quote"][0]["open"]
                close_data = JSONContent["chart"]["result"][0]["indicators"]["quote"][0]["close"]
                low_data = JSONContent["chart"]["result"][0]["indicators"]["quote"][0]["low"]
                high_data = JSONContent["chart"]["result"][0]["indicators"]["quote"][0]["high"]
                volume_data = JSONContent["chart"]["result"][0]["indicators"]["quote"][0]["volume"]

                if not [val for val in low_data if val is not None]:
                    print("warning:", symbol, "has no data.  removing it.")
                    continue

                if len(timestamp_data) != len(low_data):
                    print("warning: len diff ts", len(timestamp_data), "low", len(low_data))
                    timestamp_data = timestamp_data[:len(low_data)]

                data = {"timestamp": timestamp_data,
                        "open": open_data,
                        "close": close_data,
                        "low": low_data,
                        "high": high_data,
                        "volume": volume_data}

                price_df = pd.DataFrame(data)
                filename = data_folder / (symbol + filename_suffix)
                price_df.to_pickle(filename)
                price_dfs[symbol] = price_df
            except:
                print("something happened creating the df")
                import pdb ; pdb.set_trace()
        else:
            print("error fetching data!")
            import pdb ; pdb.set_trace()

    return price_dfs









##### Preprocess prices: transform dataFrame functions #####

def preprocess_prices(price_dfs, how_far_back, req_interval_dur):
    if price_dfs == {}:
        return price_dfs

    if True:
        for symbol, _ in price_dfs.items():

            #if symbol in ernest_stocks:

                print("preprocess", symbol, req_interval_dur, "...")

                # fill in missing values
                price_dfs[symbol] = interpolate(price_dfs[symbol])

                # limit it to how far back i want
                price_dfs[symbol] = transf_limit_how_far_back(price_dfs[symbol], how_far_back)

                ## transform prices to match the requested interval duration
                #price_dfs[symbol] = transf_is(price_dfs[symbol], req_interval_dur)
                #save_df_as_tensors(price_dfs[symbol], symbol, how_far_back,
                #                req_interval_dur, "tensor_data")

                # quicker alternative to transform prices to match requested interval
                # add in extra meta data

                # try to add column fill in
                price_dfs[symbol]["marketCap"] = get_market_cap(symbol)

                # try to add column fill in
                price_dfs[symbol]["sector"] = get_sector(symbol)

                tensor = resample(price_dfs[symbol], req_interval_dur)
                save_tensor(tensor, symbol, how_far_back, req_interval_dur, "tensor_data")

    kk

    rids = [1] #[27, 9, 3, 1]  #[180000, 144000, 72000, 36000]:
    cluster_types = [90, 130] #[90, 130, 210] #[90, 110]
    how_far_back = 126144000
    desired_seq_len = 130 #111
    future_pred_len = 25  #needs to be < desired_seq_len

    #### TRAIN
    if False:
        for n_clusters in cluster_types:
            for req_interval_dur in rids:
                print('now clustering..', n_clusters, req_interval_dur)

                #(train_inp, test_inp,
                # test_syms) = get_n_chopup_tensors(how_far_back, req_interval_dur,
                #                                     desired_seq_len, future_pred_len)

                with open("hierarchical_train_X" + ".pkl", "rb") as f:
                    train_inp = pickle.load(f)

                print(' training...')
                name = "_".join(["earlyclf", str(how_far_back), str(req_interval_dur),
                                str(desired_seq_len), str(n_clusters)])
                early_clf = Early_CLF(train_inp, n_clusters)
                try:
                    early_clf.save(name)
                except:
                    import pdb ; pdb.set_trace()



    #### TEST
    # get some symbol names
    if False:

        #train_inp, _, _ = get_n_chopup_tensors(how_far_back, req_interval_dur,
        #                                       desired_seq_len, future_pred_len)

        random_syms = [random.choice(list(price_dfs)) for _ in range(400)]
        #syms = list(set(random_syms)) # + ernest_stocks
        #syms = list(price_dfs)[506:]
        syms = symbols
        #syms = "everything"
        print("len(syms):", len(syms))
        all_clf_data = {}
        for sym in syms:
            all_clf_data[sym] = []

        #val scores
        clf_val_scores = {}
        for n_clusters in cluster_types:
            for req_interval_dur in rids:
                clf_val_scores[str(n_clusters) + "_" + str(req_interval_dur)] = []

        for n_clusters in cluster_types:
            for req_interval_dur in rids:
                print()
                print(n_clusters, req_interval_dur, "...")

                name = "_".join(["earlyclf", str(how_far_back), str(req_interval_dur),
                                str(desired_seq_len), str(n_clusters)])

                with open(name + ".pkl", "rb") as f:
                    early_clf = pickle.load(f)

                #early_clf.plot_clusters()

                # testing:  get data, predict, put everything in a dict
                test_seq_len = desired_seq_len - future_pred_len
                _, test_data = get_tensors(how_far_back, req_interval_dur, desired_seq_len,
                                             test_seq_len, syms)
                syms = list(test_data)  # update syms based on what data we got

                _, val_data = get_tensors(how_far_back, req_interval_dur, desired_seq_len,
                                            desired_seq_len, syms)

                syms = list(val_data)
                # add data and filter syms based on baryline
                syms_scores = {}
                for sym in syms:

                    x_val = val_data[sym]
                    x_val, baryline_val, t_val, _, mae = early_clf.predict(x_val, future_pred_len)
                    clf_val_scores[str(n_clusters) + "_" + str(req_interval_dur)] += [mae]

                    #"""
                    x_test = test_data[sym]
                    d0, d1, d2 = x_test.shape
                    x_test = numpy.concatenate([x_test,
                                                numpy.zeros((d0, future_pred_len, d2))], axis=1)
                    x_test, baryline, t, score, _ = early_clf.predict(x_test, future_pred_len)

                    name = "_".join([sym, str(req_interval_dur),
                                    str(desired_seq_len), str(n_clusters)])
                    all_clf_data[sym] += [[name, x_test, baryline, t, score,
                                           x_val, baryline_val, t_val]]

                    syms_scores[sym] = score

                for _ in range(2):
                    mean_score = numpy.min(list(syms_scores.values()))
                    syms_to_remove = [k for (k, v) in syms_scores.items() if v == mean_score]
                    if len(syms_to_remove) > 0:
                        syms.remove(syms_to_remove[0])
                        syms_scores.pop(syms_to_remove[0])
                        all_clf_data.pop(syms_to_remove[0])
                #[all_clf_data.pop(sym) for sym in syms_to_remove]
                print("syms count:", len(syms), "removed", len(syms_to_remove))

        all_clf_data = list(all_clf_data.values())
        all_clf_data = sorted(all_clf_data, key=lambda x:sum([i[4] for i in x]), reverse=True)
        #"""

        # calc which early_clf has the best validation prediction
        print()
        clf_val_scores = sorted(clf_val_scores.items(), key=lambda x:numpy.mean(x[1]))
        for clf_name, val_scores in clf_val_scores:
            print(clf_name, "\t", len(val_scores), numpy.mean(val_scores))
        print()

        try:
            all_clf_data = all_clf_data[:20]
        except:
            import pdb ; pdb.set_trace()

        # plot
        plt.figure()
        plt.title("Test Set")
        num_cols = len(all_clf_data)
        num_rows = len(rids)*len(cluster_types)
        for col_i, sym_clf_data in enumerate(all_clf_data):
            for row_i, spec_clf_data in enumerate(sym_clf_data):
                plt.subplot(num_rows, num_cols, row_i*num_cols + col_i + 1)
                name, time_series, baryline, t, score, x_val, baryline_val, t_val = spec_clf_data

                ## plot line, validation prediction line, and future prediction line
                # chart 0 to end of collection and added 0's start
                plt.plot(numpy.arange(0, time_series.shape[0] - future_pred_len),
                        time_series[:-future_pred_len].ravel(), linewidth=1.5)
                # plot baryline_val[t_val:] shifted left by future_pred_len
                # (val is to the left by future_pred_len)
                # i need to pick a start_plot_idx, then plot baryline_val[t_val:]
                # start_plot_idx = t_val - future_pred_len
                # if start_plot_idx is -5, start_plot_idx = 0 and cut 5 from front: x[5:]
                val_pred_line = baryline_val[t_val:]
                start_plot_idx = t_val - future_pred_len
                if start_plot_idx < 0:
                    start_plot_idx = 0
                    val_pred_line = val_pred_line[abs(start_plot_idx):]
                plt.plot(numpy.arange(start_plot_idx,
                                      start_plot_idx + val_pred_line.ravel().shape),
                         val_pred_line.ravel(), "g-", linestyle="dashed")
                # plot future prediction line
                plt.plot(numpy.arange(t, time_series.shape[0]),
                        baryline[t:].ravel(), "r-", linestyle="dashed")
                plt.text(0.00, 1.02, name + "_" + str(score)[:3], transform=plt.gca().transAxes)
                plt.axvline(x=t, color="r", linewidth=.5)
                plt.axvline(x=start_plot_idx, color="g", linewidth=.5)
                plt.xlim(0, time_series.shape[0] - 1)
        plt.show()

    kk


    # only keep stocks that have a minimum number of prices
    price_dfs = {k:v for k,v in price_dfs.items() if v.shape[0] > 3}

    return price_dfs


def interpolate(df):
    # first replace 0's with nans
    df["open"].replace(to_replace=[0.0], value=np.nan, inplace=True)
    df["close"].replace(to_replace=[0.0], value=np.nan, inplace=True)
    df["low"].replace(to_replace=[0.0], value=np.nan, inplace=True)
    df["high"].replace(to_replace=[0.0], value=np.nan, inplace=True)

    # interpolate nans
    try:
        r = df.dropna().iloc[0,:]
    except:
        import pdb ; pdb.set_trace()
    df.iloc[0] = [df.iloc[0]["timestamp"], r["open"], r["close"], r["low"], r["high"], r["volume"]]
    df = df.interpolate()
    df["timestamp"] = df["timestamp"].astype(int)

    # check for 0.0s
    min_price = df["low"].min()
    max_price = df["high"].max()

    if (min_price == 0.0) or (max_price == 0.0):
        import pdb ; pdb.set_trace()
    if df["low"].isna().sum() > 0:
        import pdb ; pdb.set_trace()

    return df


def _nearest_idx(df, timestamp_wanted, prev_idx):
    return df.iloc[(df[prev_idx:]["timestamp"] - timestamp_wanted).abs().argsort()].index[0]+prev_idx


def transf_limit_how_far_back(df, how_far_back):
    # last_timestamp - how_far_back
    start_timestamp_wanted = df["timestamp"].iloc[-1] - how_far_back
    # return df starting at the nearest timestamp to start_timestamp_wanted
    start_idx = _nearest_idx(df, start_timestamp_wanted, 0)
    return df.iloc[start_idx:,:].reset_index(drop=True)

def transf_is(old_df, req_interval_dur): # change interval duration shape and add percent diff column
    new_df = pd.DataFrame(columns=["timestamp", "open", "close", "low", "high", "percent diff", "volume"])
    new_df_insert_idx = 0
    start_timestamp = old_df["timestamp"].iloc[0]
    last_timestamp = old_df["timestamp"].iloc[-1]

    if isinstance(start_timestamp, float) or isinstance(last_timestamp, float):
        import pdb ; pdb.set_trace()

    prev_idx = 0


    for begin_timestamp in range(start_timestamp, last_timestamp, req_interval_dur):
        begin_idx = _nearest_idx(old_df, begin_timestamp, prev_idx)
        end_idx = _nearest_idx(old_df, begin_timestamp + req_interval_dur, prev_idx)
        prev_idx = begin_idx

        # time
        timestamp = old_df.iloc[begin_idx,:]["timestamp"]

        # open, close
        opn = old_df.iloc[end_idx,:]["open"]
        close = old_df.iloc[end_idx,:]["close"]

        # low, high
        min_price = old_df.iloc[begin_idx:end_idx+1,:]["low"].min()
        max_price = old_df.iloc[begin_idx:end_idx+1,:]["high"].max()
        percent_diff = (max_price - min_price)/min_price

        # volume
        volume = old_df.iloc[begin_idx:end_idx+1,:]["volume"].sum()

        new_df.loc[new_df_insert_idx] = [timestamp, opn, close, min_price, max_price, percent_diff, volume]
        new_df_insert_idx += 1

        #if new_df.dropna().shape[0] != new_df.shape[0]:
        #    import pdb ; pdb.set_trace()

    new_df["timestamp"] = new_df["timestamp"].astype(int)
    return new_df



def resample(df, req_interval_dur):  # similar to transf_is, change df # rows based on rid
    # to tensor
    df['timestamp'] = pd.to_datetime(df['timestamp'],unit='s')
    df = df.set_index('timestamp')
    r = df.resample(str(req_interval_dur) + "H")

    df = r.agg({"low": np.min, "high": np.max, "volume": np.sum,
                "marketCap": np.min, "sector": np.min}).dropna()
    df = df[["low", "high", "volume", "marketCap", "sector"]]

    return torch.tensor(df.to_numpy()).unsqueeze(0)



def save_tensor(tensor, symbol, how_far_back, req_interval_dur, folder_name):
    # save
    data_folder = Path(folder_name)
    data_folder.mkdir(parents=True, exist_ok=True)
    filename_suffix = "_".join([symbol, str(how_far_back), str(req_interval_dur), ".pt"])
    torch.save(tensor, data_folder / ("unnorm_" + filename_suffix))
    print(how_far_back, req_interval_dur, filename_suffix, tensor.size())


def save_df_as_tensors(df, symbol, how_far_back, req_interval_dur, folder_name):
    # normalize
    X_unnorm = []
    X_robust = []
    X_tornor = []
    df = df[["low", "high", "volume"]]
    #df = df[["low", "high", "percent diff", "volume"]]
    X_unnorm += [torch.tensor(df.to_numpy()).unsqueeze(0)]
    x_robust = RobustScaler().fit_transform( df.to_numpy() )
    X_robust += [torch.tensor(x_robust).unsqueeze(0)]
    x_tornor = torch.nn.functional.normalize( torch.tensor(df.to_numpy()), dim=0 )
    X_tornor += [x_tornor.unsqueeze(0)]
    X_unnorm = torch.cat(X_unnorm, dim=0)
    X_robust = torch.cat(X_robust, dim=0)
    X_tornor = torch.cat(X_tornor, dim=0)

    # save
    data_folder = Path(folder_name)
    data_folder.mkdir(parents=True, exist_ok=True)
    filename_suffix = "_".join([symbol, str(how_far_back), str(req_interval_dur), ".pt"])
    torch.save(X_unnorm, data_folder / ("unnorm_" + filename_suffix))
    #torch.save(X_robust, data_folder / ("robust_" + filename_suffix))
    #torch.save(X_tornor, data_folder / ("tornor_" + filename_suffix))

    print(how_far_back, req_interval_dur, filename_suffix, "X_robust", X_robust.size())







### Analysis functions ###

def num_crosses(df):  # for each interval_dur, scan the range of prices to get max hor crosses
    start_timestamp = df["timestamp"].iloc[0]
    last_timestamp = df["timestamp"].iloc[-1]

    interval_count = 3
    interval_dur = (last_timestamp - start_timestamp) // interval_count

    if isinstance(start_timestamp, float) or isinstance(last_timestamp, float):
        import pdb ; pdb.set_trace()

    total_crosses = 0
    for begin_timestamp in range(start_timestamp, last_timestamp, interval_dur):
        begin_idx = _nearest_idx(df, begin_timestamp)
        end_idx = _nearest_idx(df, begin_timestamp + interval_dur)

        sub_df = df.iloc[begin_idx:end_idx+1,:]

        min_price = sub_df["low"].min()
        max_price = sub_df["high"].max()

        if min_price == max_price:
            continue

        max_count = 0
        for price in np.arange(min_price, max_price, (max_price - min_price)/10):
            bool_vals = (sub_df["low"] <= price) & (price <= sub_df["high"])
            try:
                count = bool_vals.value_counts().loc[True]
            except:
                # looks like it did a quick drop or spike up that's not continuous between adjacent timestamps
                count = 1  # an imaginary cross that happened but not precisely if you know what i mean
                #print("warning: 1 or more imaginary cross for df:", id(df))
            max_count = max(max_count, count)

        total_crosses += max_count

    return total_crosses

def calc_rank(l):
    num_map = {j: i for i, j in enumerate(sorted(set(l)))}
    return [num_map[n] for n in l]





### Parameters processing ###

class Options():
    def __init__(self, symbols, add_options):

        self.symbols = symbols

        self.how_far_backs = []
        self.req_interval_durs = []
        self.param_names = []
        self.price_dfs = []

        for o in add_options:
            self.add(o)

    def add(self, option):

        if option == 1:
            hfb = 2*24*730*hr
            rid = 1 #hr
            pn = "24mo | 1 hr"
            pdf_n = "price_dfs_60m" ; pdf_i = "60m" ; pdf_r = "2y"

        if option == 2:
            hfb = 2*24*730*hr
            rid = 2 #2*hr
            pn = "24mo | 2 hr"
            pdf_n = "price_dfs_60m" ; pdf_i = "60m" ; pdf_r = "2y"

        if option == 3:
            hfb = 2*24*730*hr
            rid = 3 #3*hr
            pn = "24mo | 3 hr"
            pdf_n = "price_dfs_60m" ; pdf_i = "60m" ; pdf_r = "2y"

        if option == 5:
            hfb = 2*24*730*hr
            rid = 5 #5*hr
            pn = "24mo | 5 hr"
            pdf_n = "price_dfs_60m" ; pdf_i = "60m" ; pdf_r = "2y"

        if option == 8:
            hfb = 2*24*730*hr
            rid = 8 #8*hr
            pn = "24mo | 8 hr"
            pdf_n = "price_dfs_60m" ; pdf_i = "60m" ; pdf_r = "2y"

        if option == 9:
            hfb = 2*24*730*hr
            rid = 9 #8*hr
            pn = "24mo | 8 hr"
            pdf_n = "price_dfs_60m" ; pdf_i = "60m" ; pdf_r = "2y"

        if option == 13:
            hfb = 2*24*730*hr
            rid = 13 #13*hr
            pn = "24mo | 13 hr"
            pdf_n = "price_dfs_60m" ; pdf_i = "60m" ; pdf_r = "2y"

        if option == 20:
            hfb = 2*24*730*hr
            rid = 20 #20*hr
            pn = "24mo | 20 hr"
            pdf_n = "price_dfs_60m" ; pdf_i = "60m" ; pdf_r = "2y"

        if option == 27:
            hfb = 2*24*730*hr
            rid = 27 #20*hr
            pn = "24mo | 20 hr"
            pdf_n = "price_dfs_60m" ; pdf_i = "60m" ; pdf_r = "2y"

        self.how_far_backs += [hfb]
        self.how_far_backs = [int(i) for i in self.how_far_backs]

        self.req_interval_durs += [rid]
        self.req_interval_durs = [int(i) for i in self.req_interval_durs]

        self.param_names += [pn]

        print("get_charts", len(self.symbols))
        pdf = get_charts(self.symbols[:], pdf_i, pdf_r)
        print("preprocess_prices", self.how_far_backs[-1], self.req_interval_durs[-1])
        pdf = preprocess_prices(pdf, self.how_far_backs[-1], self.req_interval_durs[-1])

        self.price_dfs += [pdf]


    def __iter__(self):
        return self

    def __next__(self):
        if self.how_far_backs:
            return (self.how_far_backs.pop(0), self.req_interval_durs.pop(0),
                    self.param_names.pop(0), self.price_dfs.pop(0))
        else:
            raise StopIteration

time_r = 1.0

if __name__ == "__main__":

    # parameters
    num_movers = 15
    hr = 3600

    # remove any symbols I don't want to query yahoo for
    symbols = [i.lower() for i in symbols]          # make lowercase
    symbols = list(set(symbols))                    # remove duplicates
    symbols = [i for i in symbols if '-' not in i]  # remove symbols w/ hyphens
    symbols = [i for i in symbols if len(i) < 5]    # remove symbols w/ >4 chars
    symbols = list(set(symbols) - set(delisted))    # remove known delisted symbols
    symbols = sorted(symbols)

    print("num symbols to be queried:", len(symbols))
    print()

    #get_summary(symbols)

    options = Options(symbols, [1])


    print("\n\n\n\n\n ========== RESULTS ==========")
    print()
    print()

    # analysis
    for how_far_back, req_interval_dur, param_name, price_dfs in options:

        # find largest medians
        largest_medians = []
        for symbol, price_df in price_dfs.items():
            median_diff = price_df["percent diff"].median()
            largest_medians.append([symbol, median_diff])
        largest_medians = sorted(largest_medians, key=lambda x:x[1])[::-1]


        # find most cross count
        most_crosses = [[symbol, num_crosses(price_df)] for symbol, price_df in price_dfs.items()]
        most_crosses = sorted(most_crosses, key=lambda x:x[1])[::-1]

        # find lowest current price
        perc_from_means = []
        for symbol, price_df in price_dfs.items():
            perc_from_mean = price_df.iloc[-1]["high"] / price_df["high"].mean()
            perc_from_means.append([symbol, perc_from_mean])
        perc_from_means = sorted(perc_from_means, key=lambda x:x[1])


        # results
        print()
        print("last", param_name, "intervals:")
        print()

        # largest medians
        print("largest medians:")
        lm_syms = []
        for i, (sym, val) in enumerate(largest_medians[:50]):
            cc = [v for (s, v) in most_crosses if s == sym][0]
            lcp = [v for (s, v) in perc_from_means if s == sym][0]
            lm_syms += [sym]
            sym += "     " ; sym = sym[:5]
            if i < 10:
              print("  %s %.3f | %.3f | %i" % (sym, val, lcp, cc))
        print()

        # lowest current price
        print("lowest current price:")
        lcp_syms = []
        for i, (sym, val) in enumerate(perc_from_means[:50]):
            lm = [v for (s, v) in largest_medians if s == sym][0]
            cc = [v for (s, v) in most_crosses if s == sym][0]
            lcp_syms += [sym]
            sym += "     " ; sym = sym[:5]
            if i < 10:
              print("  %s %.3f | %.3f | %i" % (sym, lm, val, cc))
        print()

        # intersection of lm and lcp
        print("intersection:")
        #inter_syms = list( set(lm_syms).intersection(set(lcp_syms)) )
        inter_syms = [sym for sym in lm_syms if sym in lcp_syms]
        most_attract_symbols = inter_syms[:5]
        for sym in inter_syms[:5]:
            lm = [v for (s, v) in largest_medians if s == sym][0]
            cc = [v for (s, v) in most_crosses if s == sym][0]
            lcp = [v for (s, v) in perc_from_means if s == sym][0]

            sym += "     " ; sym = sym[:5]
            print("  %s %.3f | %.3f | %i" % (sym, lm, lcp, cc))

        print("\n\n\n")
        print("---------------------")
        print("Stocks to buy", "(", param_name, "range ):  ", most_attract_symbols)
        print("---------------------")
        print("\n\n\n")
