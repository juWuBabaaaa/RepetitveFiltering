from filterpy.kalman import KalmanFilter
from numpy import *
import numpy as np
from filterpy.common import Q_discrete_white_noise
from scipy.optimize import linear_sum_assignment
import pandas as pd


class Track:
    def __init__(self, iid):
        self.track = KalmanFilter(dim_x=2, dim_z=1)
        self.track.x = array([0., 0])
        self.track.F = array([[1., 1.], [0., 1.]])
        self.track.H = array([[1., 0.]])
        self.track.P *= array([[5000., 0.], [0., 5000.]])    # P 很有用啊, covariance matrix
        self.track.R = array([[5.]])
        self.track.Q = array([[0., 0.], [0., 0.01]])
        # self.track.Q = Q_discrete_white_noise(dim=1, dt=0.1, var=0.13)
        self.age = 0
        self.height = 0
        self.id = iid
        self.state = 1
        self.history = []

    def predict(self):
        self.age += 1
        self.track.predict()

    def update(self, m):
        self.age = 0
        self.track.update(m)

    def get(self):
        return self.track.x[0]

    def record(self):
        self.history.append([self.track.x[0], self.height])


class Tracker:
    def __init__(self, max_age=0, max_distance=20, n_init=2):
        self.max_age = max_age
        self.max_distance = max_distance
        self.n_init = n_init
        self._next_id = 0
        self.tracks = []
        self.records = {}

    def predict(self):  # 不需要在外部调用
        for track in self.tracks:
            track.predict()

    def update(self, measurements):  # mx2 shape, (loc, height)
        if len(self.tracks) == 0:
            self._initiate(measurements)
            return
        else:
            predictions = array([i.get() for i in self.tracks])
        cost = abs(predictions[:, None] - measurements[:, 0])
        row_ind, col_ind = linear_sum_assignment(cost)
        for i, j in zip(row_ind, col_ind):
            self.tracks[i].update(measurements[j, 0])
            self.tracks[i].height = measurements[j, 1]
            self.tracks[i].record()
        mask1 = in1d(arange(predictions.size), row_ind)
        mask2 = in1d(arange(measurements[:, 0].size), col_ind)
        unmatched_pre_id = np.array([i.id for i in array(self.tracks)[~mask1]])
        unmatched_m = measurements[~mask2, :]
        self._initiate(unmatched_m)
        for item in self.tracks:  # 没匹配到的移除
            if item.id in unmatched_pre_id:
                if item.age <= self.max_age:
                    item.predict()
                    item.record()
                else:
                    self.records[item.id] = item.history
                    self.tracks.remove(item)     # 不能只在这里移除，如果一直都匹配到，就不移除了吗？

    def _initiate(self, measurements):
        for i in arange(measurements.shape[0]):
            self.tracks.append((Track(self._next_id)))
            self.tracks[-1].update(measurements[i, 0])
            self.tracks[-1].height = measurements[i, 1]
            self.tracks[-1].record()
            self._next_id += 1


if __name__ == "__main__":
    from tqdm import tqdm
    from scipy import signal
    from matplotlib.pyplot import *
    import os

    tracker = Tracker()
    arr = load("/data/npy/4.npy")

    # fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(20, 6), sharex=True)
    # sig = arr[127, 3000:3600]
    sig1 = arr[80, 100:3000]
    sig2 = arr[245, 100:3000]
    sig3 = arr[210, 3100:5901]
    sig = sig3[:2000]

    images = []

    fig, ax = subplots(2, 1, figsize=(12, 6), sharex=True)
    images.append(ax[0].plot(sig))
    points = []
    M = 3
    win = np.ones(M)
    f = sig
    ax[1].set_xlabel("position")
    ax[0].set_ylabel("iterations")
    ax[1].set_ylabel("iterations")
    for k in tqdm(range(100)):
        print("--------k:", k)
        f = convolve(win, f, mode='same') / sum(win)
        peaks, _ = signal.find_peaks(f)
        values = f[peaks]
        P = c_[peaks, k * ones_like(peaks), values]  # (loc, y, height)
        tracker.predict()
        tracker.update(P[:, [0, 2]])
        print("len tracker:", len(tracker.tracks))
        points.append(P)
    df_dict = {"id": [], "loc": [], "h": []}
    for item in tracker.tracks:  # 取出一直没被删除的目标的历史记录
        tracker.records[item.id] = item.history
    for key, values in tracker.records.items():
        for value in values:
            df_dict["id"].append(key)
            df_dict["loc"].append(value[0])
            df_dict["h"].append(value[1])
    df = pd.DataFrame(df_dict)
    df.to_csv("mot_re.csv", index=False)
    for key in tracker.records.keys():
        values = np.array(tracker.records[key])
        ax[1].scatter(values[:, 0], arange(len(values[:, 0])), s=2)
        # ax[2].text(values[-1, 0], arange(len(values[:, 0]))[-1], f"{key}")
    points = np.concatenate(points)
    # ax[0].plot(f)
    ax[0].scatter(points[:, 0], points[:, 1], s=2)
    # images.append(ax[1].scatter(points[:, 0], points[:, 1], c=points[:, 2], s=2, cmap='viridis'))
    # fig.colorbar(images[1], orientation='horizontal')
    savefig("peakSort.png")
    show()


