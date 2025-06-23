import pandas as pd
from scipy import signal
import tracker
from scipy.optimize import curve_fit
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import seaborn as sns
from cycler import cycle
from matplotlib.patches import Patch
from scipy.optimize import linear_sum_assignment

"""
Steps for Peak Extraction using Repetitive Filtering Method:
    1. Obtain all peaks using iterative filtering;
    2. Track peak clusters;
    3. Determine the resting value and filter out unnecessary peak clusters;

Obtain Cluster Intervals:
    1. Read the filtered clusters
    2. Cluster into primary and secondary clusters
    3. Derive pulse interval estimates from cluster intervals

Obtain Pulse Interval Estimates:
"""

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'axes.titlesize': 14,
    'savefig.dpi': 300,
    'font.weight': 'bold'
})


class GetPP:

    def __init__(self):
        self.sig = None
        self.pp = None  # peak pillars
        self.i = None  # id
        self.initial_guess = [300, 1., 0.5]  # for curve fit

    def _rfa(self, vis=False):
        """
        Repetitive filtering algorithm.
        :param vis: 是否可视化。
        :return: the sorted peak pillars,
        """
        tra = tracker.Tracker()  # use MOT(multi object tracking) methods
        n = 200  # number of iterations
        points = []  # peak points
        win = np.ones(3)  # window function
        f = self.sig.copy()
        for i in range(n):
            f = np.convolve(win, f, mode="same") / sum(win)
            peaks, _ = signal.find_peaks(f)
            values = f[peaks]
            p = np.c_[peaks, i * np.ones_like(peaks), values]
            tra.predict()
            tra.update(p[:, [0, 2]])
            points.append(p)
        points = np.concatenate(points)
        df_dict = {"id": [], "loc": [], "h": []}
        for item in tra.tracks:  # obtain tracked stripes
            tra.records[item.id] = item.history
        for key, values in tra.records.items():
            for value in values:
                df_dict["id"].append(key)
                df_dict["loc"].append(value[0])
                df_dict["h"].append(value[1])
        self.pp = pd.DataFrame(df_dict)  #

        if vis:
            images = []
            fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex="col")
            images.append(ax[0].plot(self.sig))
            ax[1].set_xlabel("position")
            ax[0].set_ylabel("iterations")
            ax[1].set_ylabel("iterations")
            for key in tra.records.keys():
                values = np.array(tra.records[key])
                ax[1].scatter(values[:, 0], np.arange(len(values[:, 0])), s=2)
            plt.show()
        return self.pp  # dataframe, id, loc, h

    @staticmethod
    def _func(x, L, k, x0):
        return L / (1 + np.exp(-k * (x - x0)))

    def _tan_estimator(self, alpha):
        dp = "../DetStageI/npy/knee/"
        fns = os.listdir(dp)
        arr = np.load(dp + fns[self.i])
        initial_guess = [1., 1., 0.5]
        x, y = arr[:, 0], arr[:, 1]
        y = 0.5 * y / y.max()
        params = curve_fit(self._func, x, y, p0=initial_guess)
        params = params[0]
        L, k, x0 = params
        x_a = x0 - (1. / k) * np.log(1. / (2 * alpha) * (L * k - 2 * alpha - np.sqrt((L * k) ** 2 - 4 * alpha * L * k)))
        return np.round(x_a, 3)

    def _traverse(self, vis=False):  # 分析用
        # plot fitted logistic function
        # obtain height threshold to filter weak stripes
        h = np.arange(0.06, 0.5, 0.001)
        aver_h = self.pp.groupby("id")["h"].agg(["mean"]).sort_values(by="mean")
        result = [aver_h[aver_h['mean'] < threshold].shape[0] for threshold in h]

        params = curve_fit(self._func, h, np.array(result), p0=self.initial_guess)
        params = params[0]
        if vis:
            plt.plot(h, result, 'b-', label='data')
            plt.plot(h, self._func(h, *params), 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(params))
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend()
            plt.savefig(f"pic/fit/{self.i}.png")
            plt.close()
        return np.c_[h, np.array(result)]

    def _filter_pp(self, theta, alpha, vis=False):
        # the filter process of filtering weak stripes
        grouped1 = self.pp.groupby("id")["h"].agg(["mean", "min", "max", "std", "size"])
        f1 = grouped1[(grouped1["size"] >= 2) & (grouped1["mean"] >= theta)]  # 初步筛选结果
        f2 = self.pp[self.pp["id"].isin(f1.index)]
        reindex = f2.set_index("id")  # the filtered stripes.
        if not os.path.exists(f"pic/stripe{alpha}"):
            os.makedirs(f"pic/stripe{alpha}")
        if vis:
            fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex='col')
            ax[0].plot(self.sig)
            ax[0].set_ylim([0, 2.])
            ax[0].set_ylabel("Amplitude", fontsize=16)
            for i in f1.index:
                tmp = reindex.xs(i)
                ax[1].scatter(tmp["loc"], np.arange(len(tmp["loc"])), s=2)
            ax[1].set_xlabel("Position", fontsize=16)
            ax[1].set_ylabel("Iteration", fontsize=16)
            plt.savefig(f"pic/stripe{alpha}/{self.i}.png")
            plt.close()
        return f2  # 过滤后的Dataframe。

    def run(self, sig, i):
        """
        execute the whole framework of stage I
        :param sig: Input signal，one dimensional vector.
        :param i: unique number of the signal
        :return: None
        """
        self.sig = sig
        self.i = i
        pp = self._rfa(vis=False)  # obtain peak pillar stripes
        alpha = 1. / 16
        theta = self._tan_estimator(alpha=alpha)
        # print(i, "height threshold: ", theta)
        # self._traverse(vis=True)
        # # self._estimator()
        self.pp.to_csv(f"dataframe/fstripesRaw/{self.i}.csv", index=False)
        self._traverse(vis=True)
        filtered_pp = self._filter_pp(theta=theta, alpha=alpha, vis=True)  # dataframe: id, loc, h
        filtered_pp.to_csv(f"dataframe/fstripes/{self.i}.csv", index=False)


class StripeManager:
    # cluster stripes, obtain pulse intervals from stripe intervals
    def __init__(self, i, sig, csv_dp, k=4):
        self.i = i
        self.pairs = None
        self.stripeInterval = None
        self.pulseInterval = None
        self.sig = sig
        med = np.median(sig)
        mad = np.median(np.abs(sig - med))
        self.tau = med + mad
        self.csv_path = os.path.join(csv_dp, f"{i}.csv")
        self.outlier = med + mad * k

    def _assign_stripes(self, stripes, M=200):  # 条纹归属
        """
        cluster stripes into main stripe and secondary stripe

        :param stripes: List[dict] 输入条纹列表，每个条纹包含'x'坐标及'L'长度字段
        :param M: int 主条纹长度阈值
        :return: List[tuple] 返回元组列表，每个元组包含(主条纹, 附属次条纹列表)
        """
        # step 1：divide main stripe and secondary stripe.

        masters = [s for s in stripes if s['L'] == M]
        secondary = [s for s in stripes if s['L'] < M]

        # sort according to x coordinate.
        masters.sort(key=lambda x: x['x'])

        # step 2：build the map. Each secondary stripe belongs to a main stripe
        mapping = {id(m): (m, []) for m in masters}

        for s in secondary:
            min_dist = float('inf')
            closest_master = None

            # find the closest main stripe
            for m in masters:
                dist = abs(m['x'] - s['x'])
                if dist < min_dist:
                    min_dist = dist
                    closest_master = m
                elif dist == min_dist and m['x'] < closest_master['x']:
                    closest_master = m

            # decide the relationship
            if closest_master:
                mapping[id(closest_master)][1].append(s)

        self.pairs = list(mapping.values())

    def _convert_stripe_csv_to_list(self):  # load stripes（filtered）
        # use default dict
        stripe_data = defaultdict(list)

        with open(self.csv_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                stripe_id = row['id']
                x = float(row['loc'])
                h = float(row['h'])
                stripe_data[stripe_id].append((x, h))
        # result list
        result = []
        for stripe_id, tmp in stripe_data.items():
            length = len(tmp)
            tmp = np.array(tmp)
            avg_x = np.round(sum(tmp[:, 0]) / length, 3)
            h = np.round(sum(tmp[:, 1]) / length, 3)
            result.append({
                'stripe_id': stripe_id,
                'x': avg_x,
                'L': length,
                'h': h
            })

        return result

    def _get_stripe_interval(self):
        intervals = []
        for pair in self.pairs:
            loc = list()
            loc.append(pair[0]['x'])
            for s in pair[1]:
                loc.append(s['x'])
            loc = np.array(loc).astype(int)
            interval = (np.min(loc), np.max(loc))
            intervals.append(interval)
        self.stripeInterval = intervals

    def _get_pulse_interval(self):
        """
        Process the list of intervals, split and merge the sub-intervals that meet the conditions.

        parameter:
            sig (np.ndarray): one dimensional
            t (float): threshold
            intervals (list of tuple): initialize intervals，In each interval, sig < t

        :return:
            list of tuple: The processed new interval list: each interval satisfies sig >= t, with adjacent qualifying intervals merged.
        """
        # Step 1: Split each initial interval into subintervals where sig >= t
        sub_intervals = []
        for (start, end) in self.stripeInterval:
            current_start = None
            for i in range(start, end + 1):
                if self.sig[i] >= self.tau:
                    if current_start is None:
                        current_start = i  # Start new subinterval
                else:
                    if current_start is not None:
                        sub_intervals.append((current_start, i - 1))  # End current subinterval
                        current_start = None
            # Process the last unclosed subinterval
            if current_start is not None:
                sub_intervals.append((current_start, end))

        if not sub_intervals:
            return []

        # Step 2: Sort sub-intervals by start position
        sub_intervals.sort(key=lambda x: x[0])

        # Step 3: Merge adjacent sub-intervals
        merged = [sub_intervals[0]]  # Initialize merged list

        for current in sub_intervals[1:]:
            last = merged[-1]
            a1, a2 = last
            b1, b2 = current

            # Calculate gap range
            gap_start = a2 + 1
            gap_end = b1 - 1

            # Determine if mergeable
            can_merge = True
            if gap_start <= gap_end:
                # Verify all values in gap are >= threshold
                for i in range(gap_start, gap_end + 1):
                    if self.sig[i] < self.tau:
                        can_merge = False
                        break

            if can_merge:
                # Merge intervals: take min start and max end
                merged[-1] = (min(a1, b1), max(a2, b2))
            else:
                merged.append(current)
        self.pulseInterval = []
        for s, e in merged:
            # Expand left boundary leftward
            left = s
            while left > 0 and self.sig[left - 1] >= self.tau:
                left -= 1

            # Expand right boundary rightward
            right = e
            while right < len(self.sig) - 1 and self.sig[right + 1] >= self.tau:
                right += 1

            # filter
            if left != right:
                if right - left > 20:
                    if np.max(self.sig[left:right]) >= self.outlier:
                        self.pulseInterval.append((left, right))

    def pipeline1(self, y, pic_save=False):
        """
        Obtain classified stripes (final form of stripes)
        Processes individual signals

        :param y: label
        :param pic_save: whether to save visualization images
        :return:
        """
        # 1. Get stripes from CSV file
        stripes = self._convert_stripe_csv_to_list()
        self._assign_stripes(stripes)
        self._get_stripe_interval()  # Obtain stripe intervals
        self.vis_stripe_interval(pic_save)
        self._get_pulse_interval()  # Obtain pulse intervals
        self.vis_pulse_interval(y, pic_save)

    def vis_stripe_interval(self, pic_save=False):
        if not pic_save:
            return
        palette = sns.color_palette("tab20", n_colors=20)
        colors = cycle(palette.as_hex())
        fig, ax1 = plt.subplots(figsize=(14, 5))
        # Signal plot
        ax1.plot(self.sig, linewidth=1.5, color='#1f77b4', linestyle='-', label="Signal", zorder=3)
        ax1.hlines(self.tau, 0, len(self.sig), colors='#d62728', linestyles='-.',
                   linewidth=1.8, zorder=2,
                   label='Resting State Estimation')
        ax1.fill_between(np.arange(len(self.sig)), self.sig, 0., alpha=0.1, zorder=2)
        ax1.set_xlabel("Position", fontsize=26)
        ax1.set_ylabel("Amplitude", fontsize=26)
        ax1.tick_params(axis='y')
        # Interval visualization
        for interval in self.stripeInterval:
            ax1.axvspan(interval[0], interval[1], color='red', alpha=0.1)

        # Draw stripes
        ax2 = ax1.twinx()
        for master, secondaries in self.pairs:
            color = next(colors)
            # Master stripe
            ax2.plot([master['x'], master['x']], [0, master['L']],
                     color=color, linewidth=2)
            # Secondary stripes
            for secondary in secondaries:
                ax2.plot([secondary['x'], secondary['x']], [0, secondary['L']],
                         color=color, linewidth=1, linestyle='--')
        ax2.set_ylabel("Iterations", fontsize=26)
        ax2.tick_params(axis='y')
        plt.savefig(f"pic/stripeInterval/{self.i}.png")
        plt.close()

    def vis_pulse_interval(self, y, pic_save=False):
        """
        Compare estimated and labeled intervals

        :param y: labeled intervals
        :param pic_save: whether to save visualization images
        :return:
        """
        if not pic_save:
            return
        mark = y[y[:, 0] == self.i]
        plt.figure(figsize=(15, 4))
        plt.plot(self.sig, label='Signal', color='gray', alpha=0.8)

        # Draw predicted intervals
        if self.pulseInterval is not None:
            for interval in self.pulseInterval:
                plt.axvspan(interval[0], interval[1], color='red', alpha=0.3)

        # Draw labeled intervals
        if len(mark) != 0:
            for m in mark:
                plt.axvspan(m[1], m[2], color='green', alpha=0.3)

        # Custom legend (avoid duplicate labels)
        legend_elements = [
            Patch(facecolor='red', alpha=0.3, label='Predicted'),
            Patch(facecolor='green', alpha=0.3, label='Label')
        ]
        plt.legend(handles=legend_elements)

        plt.xlabel('Position', fontsize=18)
        plt.ylabel('Amplitude', fontsize=18)
        plt.savefig(f"pic/pulseInterval/{self.i}.png")
        plt.close()


if __name__ == '__main__':

    save_figure = True  # visualization or not
    csv_dp = "dataframe/fstripes/"

    sol = GetPP()  # Stage I, obtaining stripes, track and cluster stripes, filter stripes
    x = np.load("example_signal.npy")  # load signal data
    y = np.load("label.npy")  # load labels for comparison
    """make files to save results"""
    if not os.path.exists("pic/fit"):
        os.makedirs("pic/fit")
    if not os.path.exists("pic/stripeInterval"):
        os.makedirs("pic/stripeInterval")
    if not os.path.exists("dataframe/fstripesRaw"):
        os.makedirs("dataframe/fstripesRaw")
    if not os.path.exists("dataframe/fstripes"):
        os.makedirs("dataframe/fstripes/")  # filtered stripes
    if not os.path.exists("pic/pulseInterval"):
        os.makedirs("pic/pulseInterval")

    for i in tqdm(range(x.shape[0])):
        sol.run(x[i], i)
        # break
        get_pulse_range = StripeManager(i=i, sig=x[i], csv_dp=csv_dp)
        print("outlier: ", get_pulse_range.outlier)
        get_pulse_range.pipeline1(y, save_figure)
