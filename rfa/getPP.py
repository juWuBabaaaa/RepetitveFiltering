from .tracker import tracker
import numpy as np
from scipy import signal
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import itertools


class GetPP:
    """
    use function run, generate peak pillars.
    The only postprocessing is in the tangent part.
    """

    def __init__(self, dataframe_p):
        self.sig = None
        self.pp = None   # peak pillars
        self.i = None  # id
        self.initial_guess = [300, 1., 0.5]  # for curve fit
        self.dataframe_p = dataframe_p

    def _rfa(self, vis=False, save_stripe_raw=True):
        """
        Repetitive filtering algorithm.
        :param vis: 是否可视化。
        :param save_stripe_raw: whether save dataframe of raw stripes
        :return: the sorted peak pillars,
        """
        tra = tracker.Tracker()  # 调用MOT模型
        n = 200  # 重复滤波的次数
        points = []  # 波峰点
        win = np.ones(3)  # 窗口函数
        f = self.sig.copy()
        for i in range(n):
            f = np.convolve(win, f, mode="same") / sum(win)
            peaks, _ = signal.find_peaks(f)
            values = f[peaks]
            p = np.c_[peaks, i * np.ones_like(peaks), values]   # for plot, (peak, iter, f[peak])
            tra.predict()
            tra.update(p[:, [0, 2]])
            points.append(p)
        points = np.concatenate(points)
        df_dict = {"id": [], "loc": [], "h": []}
        for item in tra.tracks:  # 取出一直没被删除的目标的历史记录
            tra.records[item.id] = item.history
        for key, values in tra.records.items():
            for value in values:
                df_dict["id"].append(key)
                df_dict["loc"].append(value[0])
                df_dict["h"].append(value[1])
        self.pp = pd.DataFrame(df_dict)

        if save_stripe_raw:
            dp = "dataframe/raw"
            if not os.path.exists(dp):
                os.makedirs(dp)
            fp = os.path.join(dp, f"{self.i}.csv")
            self.pp.to_csv(fp, index=False)
            knee_dp = "knee/npy"
            if not os.path.exists(knee_dp):
                os.makedirs(knee_dp)
            knee_fp = os.path.join(knee_dp, f"{self.i}.npy")
            h = self.pp.groupby("id")["h"].mean()
            sorted_h = np.sort(h)
            H = np.arange(0.06, 0.5, 0.001)
            x = np.searchsorted(sorted_h, H, side='left')
            knee = np.c_[H, x]  # shape m*2
            np.save(knee_fp, knee)

        if vis:
            colors = ["#C52A20", "#508AB2", "#D5BA82", "#B36A6F", "#A1D0C7", "#508AB2"]
            cycle = itertools.cycle(colors)

            # scatter peaks
            fig, ax = plt.subplots(1, 1, figsize=(12, 5))
            ax.scatter(
                points[:, 0], points[:, 1],
                s=2,
                c="#508AB2",
            )
            ax.set_xlabel("Position", fontsize=24)
            ax.set_ylabel("Iteration", fontsize=24)
            plt.savefig(f"pic/ScatterPeaks/{self.i}.png", bbox_inches='tight', transparent=True)
            plt.close()

            # plot stripe
            images = []
            fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex="col")
            images.append(ax[0].plot(self.sig))
            ax[1].set_xlabel("Position", fontsize=24)
            ax[0].set_ylabel("Iteration", fontsize=24)
            ax[1].set_ylabel("Iteration", fontsize=24)
            for key in tra.records.keys():
                c = next(cycle)
                values = np.array(tra.records[key])
                ax[1].scatter(values[:, 0], np.arange(len(values[:, 0])), s=2, c=c)
            plt.savefig(f"pic/TrackedStripes/{self.i}.png", bbox_inches='tight', transparent=True)
            plt.close()
        return self.pp  # dataframe, id, loc, h

    @staticmethod
    def _func(x, L, k, x0):
        return L / (1 + np.exp(-k * (x - x0)))

    def _tan_estimator(self, alpha):
        # dp = "D:/code/fiber/DetStageI/npy/knee/"
        dp = "knee/npy"
        # fns = os.listdir(dp)
        arr = np.load(os.path.join(dp, f"{self.i}.npy"))
        initial_guess = [1., 1., 0.5]
        x, y = arr[:, 0], arr[:, 1]
        y = 0.5 * y / y.max()
        params = curve_fit(self._func, x, y, p0=initial_guess)
        params = params[0]
        L, k, x0 = params
        x_a = x0 - (1. / k) * np.log(1. / (2 * alpha) * (L * k - 2 * alpha - np.sqrt((L * k) ** 2 - 4 * alpha * L * k)))
        return np.round(x_a, 3)

    def _traverse(self, vis=False):  # 分析用
        # 绘制logistic切线图，查看拟合效果
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
        # 跟据切线切点坐标，滤除多余柱状条纹
        grouped1 = self.pp.groupby("id")["h"].agg(["mean", "min", "max", "std", "size"])
        f1 = grouped1[(grouped1["size"] >= 2) & (grouped1["mean"] >= theta)]  # 初步筛选结果
        f2 = self.pp[self.pp["id"].isin(f1.index)]
        reindex = f2.set_index("id")  # the filtered stripes.
        if not os.path.exists(f"pic/stripe{alpha}"):
            os.makedirs(f"pic/stripe{alpha}")
        if vis:
            fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex='col')
            colors = ["#C52A20", "#508AB2", "#D5BA82", "#B36A6F", "#A1D0C7", "#508AB2"]
            cycle = itertools.cycle(colors)
            ax[0].plot(self.sig)
            ax[0].set_ylim([0, 2.])
            ax[0].set_ylabel("Amplitude", fontsize=16)
            for i in f1.index:
                c = next(cycle)
                tmp = reindex.xs(i)
                ax[1].scatter(tmp["loc"], np.arange(len(tmp["loc"])), s=2, c=c)
            ax[1].set_xlabel("Position", fontsize=24)
            ax[1].set_ylabel("Iteration", fontsize=24)
            plt.savefig(f"pic/stripe{alpha}/{self.i}.png")
            plt.close()
        return f2  # 过滤后的Dataframe。

    def run(self, sig, i, vis=False, save_stripe_raw=True):
        """
        执行整个流程
        :param sig: 输入信号，一维向量
        :param i: 编号（第几个信号）
        :param vis: visualize?
        :param save_stripe_raw: save raw stripes, dataframe
        :return: None
        """
        self.sig = sig
        self.i = i
        pp = self._rfa(vis=vis, save_stripe_raw=save_stripe_raw)  # 获得柱状条纹 散点图

        alpha = 1./16
        theta = self._tan_estimator(alpha=alpha)

        # self.pp.to_csv(f"dataframe/fstripesFilter{alpha}/{self.i}.csv", index=False)
        self._traverse(vis=vis)
        filtered_pp = self._filter_pp(theta=theta, alpha=alpha, vis=True)   # dataframe: id, loc, h
        filtered_pp.to_csv(os.path.join(self.dataframe_p, f"{self.i}.csv"), index=False)
