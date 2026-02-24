#!/usr/bin/env python3

import datetime as dt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import gurobipy as gp
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import pandas as pd
from scipy.stats import linregress
# from statsmodels.tsa.stattools import coint
from statsmodels.tsa.stattools import adfuller

#matplotlib.use('Qt5Agg', force=True) disable for Colab

CROSSING_MEAN = 0.5
CROSSING_MAX = 2.5
ORIG_AMOUNT = 10000
SAVE_FILE = None
RISK_FREE_RATE = 4/(100)      # 4-week T-bill return rate
TX_COST = 0.001
LAMBDA = 0.5

def build_prob_cons(longs, shorts, prices, expected_returns,
                     expected_risk, trading_weights, risk_vec,
                    index, LAMBDA, tc):
    
    global CROSSING_MAX, CROSSING_MEAN

    # XXX: Make the model
    model = gp.Model('portfolio')

    # XXX: The weight vars to optimise
    lws = [model.addVar(name='l!%s!%s' % (v[0], v[1]), lb=0, ub=1)
           for v in longs]
    sws = [model.addVar(name='s!%s!%s' % (v[0], v[1]), lb=-1, ub=0)
           for v in shorts]
    model.update()              # Added the vars to the model
    # model.addVars(lws + sws)

    # print('lws: ', lws, 'sws: ', sws)

    # XXX: Get the expected returns for the currencies being traded
    return_vec = [None]*len(shorts)
    for i, v in enumerate(shorts):
        if v[1] == v[0].split('_')[1]:
            return_vec[i] = -expected_returns[v[0]][0]
        else:
            return_vec[i] = -expected_returns[v[0]][1]

    # XXX: Make the return part of the optimisation objective
    max_ret_s = np.array(sws).dot(np.array(return_vec).T)

    return_vec = [None]*len(longs)
    for i, v in enumerate(longs):
        if v[1] == v[0].split('_')[1]:
            return_vec[i] = expected_returns[v[0]][0]
        else:
            return_vec[i] = expected_returns[v[0]][1]

    # XXX: Make the return part of the optimisation objective
    max_ret_l = np.array(lws).dot(np.array(return_vec).T)
    max_ret = max_ret_s + max_ret_l
    # print('max_ret: ', max_ret)  # correct!

    # XXX: Make the risk part of the objective
    ws = [None]*len(risk_vec)
    # XXX: This one makes 2 vectors in the same order of weight
    # variables as the expected_risk matrix
    for j, (f, s) in enumerate(zip(sws, lws)):
        assert(f.VarName.split('!')[1] == s.VarName.split('!')[1])
        if f.VarName.split('!')[2] < s.VarName.split('!')[2]:
            ws[j] = np.array((f, s))
        else:
            ws[j] = np.array((s, f))
    assert(len(ws) == len(risk_vec))

    # XXX: Make the covariance part of the matrix -ve, because of shorts
    # print(risk_vec)
    for k in risk_vec:
        risk_vec[k].iat[0, 1] = -risk_vec[k].iat[0, 1]
        risk_vec[k].iat[1, 0] = -risk_vec[k].iat[1, 0]
    # print('after change: ', risk_vec)
    min_risk = np.sum([(ws[i].dot(r).dot(ws[i].T))
                       for i, r in enumerate(risk_vec.values())])
    # print('min_risk: ', min_risk)

    # XXX: The overall optimisation objective
    obj = max_ret - (LAMBDA*min_risk)
    # print('obj: ', obj)
    model.setObjective(obj, gp.GRB.MAXIMIZE)
    model.update()

    # XXX: Overall cash available constraint (robust, no bools)
    currs = [ll.VarName.split('!')[2] for ll in lws] + [ss.VarName.split('!')[2] for ss in sws]
    currs = sorted(set(currs))

    for c in currs:
        # Vars that correspond to this currency index
        ws_vars = [w for w in lws if w.VarName.split('!')[2] == c] + \
                  [w for w in sws if w.VarName.split('!')[2] == c]

        # Already-used weights for this currency (floats)
        used = float(np.sum(trading_weights[c])) if len(trading_weights[c]) > 0 else 0.0

        # Constraint: 1 - (used + sum(new weights)) >= 0
        model.addConstr(1 - (used + gp.quicksum(ws_vars)) >= 0, name=f"cash_{c}")
    
    # print('Cash constraint:')
    # [print(c) for c in c_cons]
    # XXX: Price constraint
    p_cons = [None]*len(lws)
    for j, (ll, ss) in enumerate(zip(lws, sws)):
        lindex = int(ll.VarName.split('!')[2])
        sindex = int(ss.VarName.split('!')[2])
        # print('long: ', lindex, 'long price: ', prices[lindex][index])
        # print('short: ', sindex, 'short price: ', prices[sindex][index])
        if tc:
            # when having transaction cost, ll need to be a bit more because the price for buying is higher than the price to short
            p_cons[j] = (-1*ss*((prices[lindex][index]*(1+TX_COST)))/(prices[sindex][index]*(1-TX_COST))== ll)
        else:
            p_cons[j] = (-1*ss*(prices[lindex][index]/prices[sindex][index])== ll)
        model.addConstr(p_cons[j])

    # print('Price constraint')
    # [print(p) for p in p_cons]

    return model, zip(lws, sws)

# XXX: Simulate trading. Variable spread_dates has the common dates for
# all spreads to be traded.

def simulate_trade(spreads, sigma3, lookup, spread_dates,
                   expected_returns, expected_risk, prices,
                   start, ed, zscores, tc):
    
    global LAMBDA, MAKE_SLICE, SLICE_START, SLICE_END

    # XXX: The required prices
    prices = [price[price['Date'].isin(spread_dates)]['Close'].to_numpy()
              for price in prices]

    # XXX: 7235--24186 is for CAD-GBP-USD (5 minutes)
    # max_p_index = np.min([np.argmax(p[0:24186]) for p in prices])
    max_p_index = 0 if start is None else start
    end_p_index = spread_dates.shape[0] if ed is None else ed
    # XXX: Dates:
    start_d = pd.to_datetime(spread_dates[max_p_index])
    end_d = pd.to_datetime(spread_dates[end_p_index-1])

    start_date = start_d.strftime('%Y-%m-%d')
    end_date = end_d.strftime('%Y-%m-%d')

    print('START DATE: ', start_date)
    print('END DATE: ', end_date)

    trading_days = (end_d - start_d).days
    trading_days = 1 if trading_days <= 0 else trading_days
    print('# of days: ', trading_days)
    print('# of days: ', trading_days)

    # raise Exception
    for p in prices:
        assert(spread_dates.shape[0] == len(p))

    trading_weights = {k: [] for k in lookup.keys()}
    trading_indices = {k: [max_p_index] for k in lookup.keys()}

    # XXX: Only one open position/spread at any given time
    open_positions = {k: None for k in spreads.columns}
    # open_trades = {k: None for k in spreads.columns}

    # XXX: Using X units of currency for each trade max
    orig_amount = {k: ORIG_AMOUNT for k in lookup.keys()}

    TOTAL = {k: 0 for k in spreads.columns}
    # XXX: Go through dates one by one (general case)
    # for i in range(max_p_index, spread_dates.shape[0]):

    # XXX: Best case
    for i in range(max_p_index+1, end_p_index):

   # print(open_positions)
        # XXX: Close any open position
        for k in open_positions.keys():
            dev = spreads[k].std(axis=0)
            if ((open_positions[k] is not None) and abs(zscores[k][i]) < CROSSING_MEAN*dev):
                TOTAL[k] += 1     
                sindex = open_positions[k][1]
                lindex = open_positions[k][2]
                Eth = open_positions[k][0]
                ps = prices[int(sindex)][i]*(1-TX_COST if tc else 1)
                pl = prices[int(lindex)][i]*(1+TX_COST if tc else 1)
                ws = -1*(Eth*ps)/orig_amount[sindex]
                wl = (Eth*pl)/orig_amount[lindex]
                trading_weights[sindex].append(ws)
                trading_weights[lindex].append(wl)

                # XXX: Append long and short too
                # trading_sws[sindex].append(ws)
                # trading_lws[lindex].append(wl)

                # XXX: For plotting
                trading_indices[sindex].append(i)
                trading_indices[lindex].append(i)

                # Fill closing details into trades_total
                # assert(len(trades_total[(trades_total['ct'].isna()) & (trades_total['pair']==k)]['ot']) <= 1)
                # trades_total.loc[(trades_total['ct'].isna()) & (trades_total['pair']==k), ['ct', 'pcl', 'pcs']] = [spread_dates[i-1], pl, ps]

                #assert(len(trades_total2[(trades_total2['ct'].isna()) & (trades_total2['pair']==k)]['ot']) <= 1)
                #trades_total2.loc[(trades_total2['ct'].isna()) & (trades_total2['pair']==k), ['ct', 'wcl', 'wcs']] = [spread_dates[i-1], wl, ws]

                # XXX: Position closed
                open_positions[k] = None
                # print('ps: %f, ws: %f, pl: %f, wl: %f, Eth: %f' %
                #       (ps, ws, pl, wl, Eth))

        # XXX: Get the longs and the shorts
        longs = list()
        shorts = list()
        risk_vec = dict()
        for s in spreads.columns:
            if (abs(zscores[s][i]) > sigma3[s] and (open_positions[s] is None)):
                risk_vec[s] = expected_risk[s]
                skeys = s.split('_')
                if zscores[s][i] > 0:
                    # if spreads[s][i] > 0:
                    # XXX: Short first
                    shorts.append((s, skeys[1]))
                    # XXX: Long second
                    longs.append((s, skeys[2]))
                else:
                    # XXX: Short second
                    shorts.append((s, skeys[2]))
                    # XXX: Long first
                    longs.append((s, skeys[1]))

        # XXX: DEBUG
        # if(len(longs) >= 2):
        #     print('GREAT! i, longs, shorts: ', i, longs, shorts)
        assert(len(longs) == len(shorts))
        if (len(longs) > 0):
            # print('longs: ', longs, 'shorts:', shorts)
            # LAMBDA = 1       # 1.0 is good enough
            problem, lsws = build_prob_cons(longs, shorts,
                                            prices, expected_returns,
                                            expected_risk,
                                            trading_weights, risk_vec, i,
                                            LAMBDA, tc)
            problem.Params.OutputFlag = 0
            # problem.write('problem_%d.lp' % i)

            problem.optimize()

            if problem.status == gp.GRB.INFEASIBLE:
                continue

            # XXX: Open trade positions and weights
            for ll, s in lsws:
                # XXX: ETH to buy (long)
                lindex = ll.VarName.split('!')[2]
                sindex = s.VarName.split('!')[2]
                ps = prices[int(sindex)][i]*(1-TX_COST if tc else 1)
                pl = prices[int(lindex)][i]*(1+TX_COST if tc else 1)
                Ethl = (orig_amount[lindex]*ll.X)/pl   # Eth to buy
                Eths = (orig_amount[lindex]*s.X)/ps    # Eth to sell
                # XXX: Handling precision
                Ethl = float('%0.6f' % Ethl)
                Eths = float('%0.6f' % Eths)
                # print('pl: %f, wl: %f, ps: %f, ws: %f, Ethl: %f, Eths: %f' %
                #       (pl, ll.X, ps, s.X, Ethl, Eths))
                assert Ethl == abs(Eths), ('%f != %f' % (Ethl, abs(Eths)))

                # XXX: Add the computed weights to the weight dictionary
                trading_weights[lindex].append(ll.X)
                trading_weights[sindex].append(s.X)

                # XXX: Trade (Open position)
                assert(ll.VarName.split('!')[1] == s.VarName.split('!')[1])
                open_positions[ll.VarName.split('!')[1]] = (Ethl, lindex, sindex)

                # XXX: For plotting
                trading_indices[sindex].append(i)
                trading_indices[lindex].append(i)

    # print('TOTAL trades: ', TOTAL)
    # print('open positions: ', open_positions)
    # print('max index: ', max_p_index)
    # [print(trading_indices[k][-1]) for k in trading_weights.keys()]

    for k in open_positions.keys():
        if open_positions[k] is not None:
            lkey = open_positions[k][1]
            skey = open_positions[k][2]
            # XXX: Drop the last weights from lkey and skey
            trading_weights[lkey] = trading_weights[lkey][:-1]
            trading_weights[skey] = trading_weights[skey][:-1]
            trading_indices[lkey] = trading_indices[lkey][:-1]
            trading_indices[skey] = trading_indices[skey][:-1]
            # XXX: Close the position
            open_positions[k] = None

    # print('open positions closed: ', open_positions)
    # assert len(trading_lws) == len(trading_sws)
                     
    # XXX: Total profit/loss
    fig, ax = plt.subplots(nrows=2, sharex=True)

    df_vals = pd.DataFrame()
    for k in lookup.keys():
        assert(len(trading_weights[k]) % 2 == 0)
        # print(k, ":", trading_weights[k])
        pl = (1-np.sum(trading_weights[k]))*orig_amount[k]-orig_amount[k]
        print('Profit/Loss %s: %f' % (lookup[k], pl))
        # Consider to include transaction cost
        buyeth = orig_amount[k]/prices[int(k)][max_p_index] * (1-TX_COST if tc else 1)
        sellprice = buyeth * prices[int(k)][end_p_index-1] * (1-TX_COST if tc else 1)
        # print('Buy-Hold Profit/Loss: %s: %f' %
        #       (lookup[k], (sellprice-orig_amount[k])))

        toplot = (prices[int(k)][max_p_index:end_p_index])*buyeth/orig_amount[k]
        xlabels = [start_date, end_date]
      
        xticks = [0, len(toplot)-1]

        ax[0].plot(toplot, label=lookup[k])
        ax[0].set_xticks(xticks, xlabels)
        ax[0].set_ylabel('Price Index')
        ax[0].legend()

        # ax[0].grid()
        # for i, txt in enumerate(toplot):
        #     ax[0].annotate(int(txt), xy=(i, toplot[i]))

        # XXX: Computing the calmar ratio with with buy-hold
        # pp = pd.Series((prices[int(k)][max_p_index:end_p_index])*buyeth)
        # avgpp = (pp[len(pp)-1]-orig_amount[k])/orig_amount[k]
        # ann_avgpp = ((1 + avgpp)**(365/trading_days)) - 1
        # minj = np.argmax(np.maximum.accumulate(pp)-pp)
        # maxi = np.argmax(pp[:minj])
        # maxp = pp[maxi]
        # minp = pp[minj]
        # drawdown = (maxp - minp)/maxp
        # # ppstd = ret.std()
        # rfrate = RISK_FREE_RATE
        # calmar_ratio_bh = (ann_avgpp-rfrate)/drawdown
        # # print('MAX: %f, MIN: %f, MAXI: %d, MINI: %d' %
        # #       (maxp, minp, maxi, minj))
        # print('Calmar ratio %s: %f, cum return: %f, annual return: %f,\
        # drawdown: %f' %
        #       (lookup[k], calmar_ratio_bh, (avgpp*100), (ann_avgpp*100),
        #        drawdown*100))

        # XXX: Plotting growing profits
        pl = pd.Series(trading_weights[k]).cumsum()
        ones = np.ones(len(pl))
        val = (ones-pl)*orig_amount[k]
        val = list(val)
        val.insert(0, orig_amount[k])
        # print(lookup[k], ': ', len(val))
        # print(lookup[k], 'indices : ', len(trading_indices[k]))
        assert(len(trading_indices[k]) == len(val))

        # XXX: Get the trading dates
        t_ds = pd.DataFrame(columns=['Date', ('P/L_%s' % lookup[k])])
        t_ds['Date'] = spread_dates
        for j, i in enumerate(trading_indices[k]):
            t_ds.loc[i, ('P/L_%s' % lookup[k])] = val[j]
        val = t_ds.fillna(method='ffill')
        # the amount of currency holding
        val = list(val[('P/L_%s' % lookup[k])][max_p_index:end_p_index])

        # XXX: Calmar ratio for technique
        pp = pd.Series(val)
        # ret = (pp - orig_amount[k])/orig_amount[k]  # return
        # avgpp = ret.mean()                          # average return
        # The amount of return at the last day
        avgpp = (pp[len(pp)-1]-orig_amount[k])/orig_amount[k]
        ann_avgpp = ((1 + avgpp)**(365/trading_days)) - 1
        rfrate = RISK_FREE_RATE
        # XXX: Drawdown calculation
        # XXX: Drawdown calculation (robust)
        dd = np.maximum.accumulate(pp) - pp
        minj = int(np.argmax(dd))

        if minj == 0:
            # no meaningful drawdown window
            drawdown = 0.0
            calmar_ratio_t = np.nan
        else:
            maxi = int(np.argmax(pp.iloc[:minj]))
            maxp = float(pp[maxi])
            minp = float(pp[minj])
            drawdown = (maxp - minp) / maxp if maxp != 0 else 0.0
            calmar_ratio_t = (ann_avgpp - rfrate) / drawdown if drawdown > 0 else np.nan
        # print('MAX: %f, MIN: %f, MAXI: %d, MINI: %d' %
        #       (maxp, minp, maxi, minj))
        print('Our Calmar ratio %s: %f, cum return: %f, \
            annual return: %f,drawdown: %f' %
              (lookup[k], calmar_ratio_t, (avgpp*100), (ann_avgpp*100),
               drawdown*100))

        # print('Better?: ', (calmar_ratio_t - calmar_ratio_bh)/2)

        xlabels = [start_date, end_date]

        xticks = [0, len(val)-1]

        df_vals[k] = val
        ax[1].plot(val, label=lookup[k])
        # ax[1].set_ylabel('Profit/Loss proposed tech')
        ax[1].set_ylabel('Position Movement')
        ax[1].set_xticks(xticks, xlabels)
        ax[1].legend()

        # ax[1].grid()
        # for i, txt in enumerate(val):
        #     ax[1].annotate(int(txt), xy=(i, val[i]))

    plt.tight_layout()
    # plt.savefig(SAVE_FILE, bbox_inches='tight')
    plt.show(block=True)

def keytoCUR(key, lookup):
    keys = key.split('_')
    return '_'.join([keys[0], lookup[keys[1]], lookup[keys[2]]] + keys)


def get_expected_returns(dfs):
    # First compute the log returns for the dfs
    rets = [df['Close'].transform(np.log).diff().dropna()
            for df in dfs]
    return rets


def pairwise_cov_matrices(dfs):
    # XXX: First get the log returns for each day
    rets = [df['Close'].transform(np.log).diff() for df in dfs]
    rets = [pd.DataFrame({'Date': d['Date'], 'Close': r})
            for d, r in zip(dfs, rets)]
    rets = [r.dropna() for r in rets]

    # XXX: Now do pairwise convariane matrices
    covdf = dict()
    for i in range(len(rets)):
        idates = set(rets[i]['Date'])
        for j in range(i+1, len(rets)):
            cdf = pd.DataFrame()
            jdates = set(rets[j]['Date'])
            cdates = jdates.intersection(idates)

            # XXX: Take the common dates from rets
            cdfi = rets[i][rets[i].Date.isin(cdates)]

            cdfj = rets[j][rets[j].Date.isin(cdates)]
            cdf['Close_%d' % i] = cdfi['Close'].to_numpy()
            cdf['Close_%d' % j] = cdfj['Close'].to_numpy()

            covdf['s_%d_%d' % (j, i)] = cdf.cov()
    return covdf


# XXX: oc_dates is a dict of list of tuples
def get_expected_time_to_mean(oc_dates):
    """
    oc_dates: dict spread -> list of (open_date, close_date) timestamps
    returns: pd.Series of expected time-to-mean in DAYS (float)
    """
    diff_days = {}
    for k, v in oc_dates.items():
        if len(v) == 0:
            diff_days[k] = np.nan
            continue
        # convert Timedelta to number of days (float)
        days = [(pd.to_datetime(vv[1]) - pd.to_datetime(vv[0])) / np.timedelta64(1, "D") for vv in v]
        diff_days[k] = float(np.mean(days))
    return pd.Series(diff_days)

# get_close_date(index, pair, spreads, spread_dates, zscores)
def get_close_date(start, spread, spreads, spread_dates, zscores):
    # get the std
    dev = spreads[spread].std(axis=0)
    # get in timestamp
    toret = spread_dates[start]
    # maximum index
    index = spreads.shape[0]
    for i in range(start, spreads.shape[0]):
        # XXX: Falls to within 1/2 std-dev of the spread then close
        # From the current index loop to the end, wait when spreads become small
        if(abs(zscores[spread][i]) < CROSSING_MEAN*dev):
            # record closing timestamp
            toret = spread_dates[i]
            index = i
            break

    return index, toret

# XXX: spread_dates is a numpy array of common dates
def sigma3(dfs, spreads, lookup, spread_dates, start, end, zscores, tc):
    # XXX: Get everything ready to compute the open position

    # XXX: Vector of Expected returns for each currency anchored to ETH
    ret_vec = get_expected_returns(dfs)
    # var_vec = [v.var() for v in ret_vec]
    ret_vec = [v.mean() for v in ret_vec]
    # print('log returns: ', ret_vec)

    # XXX: Get the mean of the different spreads
    # means = spreads.mean(axis=0)
    stds = spreads.std(axis=0)
    sigma3 = stds*CROSSING_MAX

    open_close_dates = {s: list() for s in spreads.columns}
    c_index = {s: -1 for s in spreads.columns}
    for i in range(spreads.shape[0]):
        for s in spreads.columns:
            # XXX: For each spread do the following
            # if((abs(spreads[s][i]) > sigma3[s]) and (i > c_index[s])):
            if((abs(zscores[s][i]) > sigma3[s]) and (i > c_index[s])):
                ci, cdate = get_close_date(i, s, spreads, spread_dates,
                                           zscores)
                c_index[s] = ci  # setting the close index
                # XXX: It is possible that it never closes at all!
                # We only consider for orders that closed in the future, otherwise not even record it.
                if cdate > spread_dates[i]:
                    open_close_dates[s].append((spread_dates[i], cdate))

    # XXX: Expected time to revert to mean for any spread
    time_to_mean_vec = get_expected_time_to_mean(open_close_dates)
    time_to_mean_vec = time_to_mean_vec.fillna(1.0).clip(lower=1.0, upper=60.0)
    # print('Time to mean:', time_to_mean_vec)

    # XXX: Now get the covariance matrices
    cov_matrices = pairwise_cov_matrices(dfs)
    # print('Pairwise covariance matrices')
    # for k, v in cov_matrices.items():
    #     print(k, ":", v)
    # XXX: DEBUG -- not exactly the same, but close enough
    # print('var_vec:', var_vec)

    # XXX: The total expected return =
    # Expected[ticks]*Expected[change/tick] for each spread to be
    # traded.
    expected_returns = pd.DataFrame()
    for s in time_to_mean_vec.index:
        keys = s.split('_')
        if not tc:
            # without transaction cost
            expected_returns[s] = [
                ret_vec[int(keys[1])]*time_to_mean_vec[s], 
                ret_vec[int(keys[2])]*time_to_mean_vec[s]
            ]
        else:
            # have to consider the return can be positive and negative
            expected_returns[s] = [
                ret_vec[int(keys[1])]*time_to_mean_vec[s] - TX_COST*abs(ret_vec[int(keys[1])]*time_to_mean_vec[s]), 
                ret_vec[int(keys[2])]*time_to_mean_vec[s] - TX_COST*abs(ret_vec[int(keys[2])]*time_to_mean_vec[s])
            ]
    # print(expected_returns)

    # XXX: Expected risk = Covariance matrix * Expected time to mean
    expected_risk = dict()
    for s in time_to_mean_vec.index:
        expected_risk[s] = cov_matrices[s]*time_to_mean_vec[s]
    # print('Expected Risk')
    # for k, v in expected_risk.items():
    #     print(k, ":", v)

    # XXX: Simulate trading
    simulate_trade(spreads, sigma3, lookup, spread_dates,
                   expected_returns, expected_risk, dfs, start, end,
                   zscores, tc)


# XXX: Main function for trading
def arbitrage_trade(fileNames, lookup, start, end, dofuller=True, tc=False):
    """
    fileNames: list of csv paths (one per asset) with columns:
              Date, Open, High, Low, Close, Volume (Yahoo format OK)
    lookup: dict mapping string indices -> ticker names
            e.g. {"0":"ALI=F","1":"XLB",...}
    start/end: index bounds (None to use all)
    dofuller: whether to filter spreads by ADF p-value
    tc: include transaction costs
    """

    # -----------------------------
    # 1) READ CSVs (robust)
    # -----------------------------
    dfs = []
    odfs = []

    for f in fileNames:
        df = pd.read_csv(f)

        # Parse Date robustly (drop bad rows)
        if "Date" not in df.columns:
            raise ValueError(f"Missing 'Date' column in: {f}")

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).copy()

        # Keep required columns (must exist)
        required = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns {missing} in: {f} | columns={df.columns.tolist()}")

        df = df[["Date"] + required].copy()

        # Force numeric
        for c in required:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # Drop rows where Close is NaN
        df = df.dropna(subset=["Close"]).sort_values("Date").reset_index(drop=True)

        dfs.append(df)
        odfs.append(df[["Date", "Close"]].copy())

    # -----------------------------
    # 2) ALIGN DATES (intersection)
    # -----------------------------
    # repeat until all have same number of dates
    while True:
        base = min([(i, len(dfs[i])) for i in range(len(dfs))], key=lambda x: x[1])[0]
        base_dates = set(dfs[base]["Date"])
        dfs = [df[df["Date"].isin(base_dates)].copy() for df in dfs]

        lengths = [df.shape[0] for df in dfs]
        if len(set(lengths)) == 1:
            break

    # reset indices after filtering
    dfs = [df.sort_values("Date").reset_index(drop=True) for df in dfs]
    common_dates = dfs[0]["Date"]
    odfs = [df[df["Date"].isin(common_dates)].sort_values("Date").reset_index(drop=True) for df in odfs]

    # -----------------------------
    # 3) BUILD CLOSE MATRIX
    # -----------------------------
    close_dict = {f"Close_{i}": dfs[i]["Close"].to_numpy() for i in range(len(dfs))}
    spread_dates = dfs[0]["Date"].to_numpy()
    close_df = pd.DataFrame(close_dict)

    # -----------------------------
    # 4) CORRELATION (no crash)
    # -----------------------------
    corrdf = close_df.corr()
    print("Correlation summary (min/median/max):",
          corrdf.values.min(),
          pd.Series(corrdf.values.ravel()).median(),
          corrdf.values.max())
    corrdf.to_csv("corr_matrix.csv")

    # -----------------------------
    # 5) SPREADS: ANCHOR vs ALL (cointegration filter)
    # -----------------------------
    close_log = close_df.transform(np.log)

    spreads = pd.DataFrame()
    anchor_idx = 0  # IMPORTANT: ALI=F must be fileNames[0] / lookup["0"]

    for j in range(close_log.shape[1]):
        if j == anchor_idx:
            continue

        x = close_log.iloc[:, anchor_idx]   # anchor
        y = close_log.iloc[:, j]            # asset j

        res = linregress(x, y)
        X = y - (res.slope * x + res.intercept)  # spread / residual

        if dofuller:
            # ADF test on residual
            _, pvalue, _, _, _, _ = adfuller(X, regression="ct")
            if pvalue < 0.1:
                spreads[f"s_{j}_{anchor_idx}"] = X
        else:
            spreads[f"s_{j}_{anchor_idx}"] = X

    if spreads.shape[1] == 0:
        raise RuntimeError("No spreads passed the ADF filter. Try dofuller=False or adjust p-value threshold.")

    # -----------------------------
    # 6) Z-SCORES
    # -----------------------------
    zscores = pd.DataFrame(index=spreads.index, columns=spreads.columns)
    for k in spreads.columns:
        zscores[k] = (spreads[k] - spreads[k].mean()) / spreads[k].std(ddof=0)

    # -----------------------------
    # 7) PLOT SPREADS
    # -----------------------------
    for s in spreads.columns:
        plt.plot(spreads[s], label=keytoCUR(s, lookup))
    plt.legend()
    plt.show(block=True)

    # -----------------------------
    # 8) RUN TRADING (sigma3 -> simulate_trade)
    # -----------------------------
    sigma3(odfs, spreads, lookup, spread_dates, start, end, zscores, tc)
    # XXX: Will most likely need both Wiener process + Poisson Jump

def do_aluminium():
    arbitrage_trade(
        [
            "./data_ohlcv/ALI_F.csv",
            "./data_ohlcv/XLB.csv",
            "./data_ohlcv/PICK.csv",
            "./data_ohlcv/DBB.csv",
            "./data_ohlcv/AA.csv",
            "./data_ohlcv/CENX.csv",
            "./data_ohlcv/KALU.csv",
            "./data_ohlcv/RIO.csv",
            "./data_ohlcv/NHYDY.csv",
            "./data_ohlcv/ACH.csv",
            "./data_ohlcv/CSTM.csv",
            "./data_ohlcv/S32.AX.csv",
            "./data_ohlcv/HINDALCO.NS.csv",
            "./data_ohlcv/1211.HK.csv",
        ],
        {
            "0": "ALI=F",
            "1": "XLB",
            "2": "PICK",
            "3": "DBB",
            "4": "AA",
            "5": "CENX",
            "6": "KALU",
            "7": "RIO",
            "8": "NHYDY",
            "9": "ACH",
            "10": "CSTM",
            "11": "S32.AX",
            "12": "HINDALCO.NS",
            "13": "1211.HK",
        },
        start=None,     # on utilisera tout l'historique
        end=None,
        dofuller=True,  # cointegration screening
        tc=True         # transaction costs 10bps
    )

def main():
    global SAVE_FILE
    SAVE_FILE = None
    print('---------- Aluminium ----------')
    do_aluminium()

if __name__ == '__main__':
    # matplotlib.use('Agg', force=True)
    plt.style.use('seaborn-v0_8-deep')
    main()
