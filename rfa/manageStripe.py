import numpy as np
import os
import csv
from collections import defaultdict
import seaborn as sns
from cycler import cycle
from matplotlib.patches import Patch
import matplotlib.pyplot as plt


class StripeManager:
    # 条纹聚类，针对单一信号，给出脉冲区间
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
        对平行于y轴的条纹进行主次分类和归属计算

        :param stripes: List[dict] 输入条纹列表，每个条纹包含'x'坐标及'L'长度字段
        :param M: int 主条纹长度阈值
        :return: List[tuple] 返回元组列表，每个元组包含(主条纹, 附属次条纹列表)
        """
        # 阶段1：分离主次条纹

        masters = [s for s in stripes if s['L'] == M]
        secondary = [s for s in stripes if s['L'] < M]

        # 按x坐标对主条纹排序（解决距离相同时选较小x的需求）
        masters.sort(key=lambda x: x['x'])

        # 阶段2：建立主-次映射关系
        mapping = {id(m): (m, []) for m in masters}  # 使用对象id处理可能重复的字典

        for s in secondary:
            min_dist = float('inf')
            closest_master = None

            # 寻找最近主条纹
            for m in masters:
                dist = abs(m['x'] - s['x'])
                if dist < min_dist:
                    min_dist = dist
                    closest_master = m
                elif dist == min_dist and m['x'] < closest_master['x']:
                    closest_master = m

            # 绑定归属关系
            if closest_master:
                mapping[id(closest_master)][1].append(s)

        self.pairs = list(mapping.values())

    def _convert_stripe_csv_to_list(self):  # 条纹读取（已用切线滤除）
        # 使用 default dict 按照 stripe_id 聚合数据
        stripe_data = defaultdict(list)

        with open(self.csv_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                stripe_id = row['id']
                x = float(row['loc'])
                h = float(row['h'])
                stripe_data[stripe_id].append((x, h))
        # 构建结果列表
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
        处理区间列表，拆分并合并满足条件的子区间。

        参数:
            sig (np.ndarray): 一维信号数组
            t (float): 阈值
            intervals (list of tuple): 初始区间列表，每个区间内 sig < t

        返回:
            list of tuple: 处理后的新区间列表，每个区间内 sig >= t，并合并了相邻符合条件的区间
        """
        # Step 1: 拆分每个初始区间为 sig >= t 的子区间
        sub_intervals = []
        for (start, end) in self.stripeInterval:
            current_start = None
            for i in range(start, end + 1):
                if self.sig[i] >= self.tau:
                    if current_start is None:
                        current_start = i  # 开始新的子区间
                else:
                    if current_start is not None:
                        sub_intervals.append((current_start, i - 1))  # 结束当前子区间
                        current_start = None
            # 处理最后一个未闭合的子区间
            if current_start is not None:
                sub_intervals.append((current_start, end))

        # 如果没有子区间，直接返回空列表
        if not sub_intervals:
            return []

        # Step 2: 按起始位置排序子区间
        sub_intervals.sort(key=lambda x: x[0])

        # Step 3: 合并相邻子区间
        merged = [sub_intervals[0]]  # 初始化合并列表

        for current in sub_intervals[1:]:
            last = merged[-1]
            a1, a2 = last
            b1, b2 = current

            # 计算间隙范围
            gap_start = a2 + 1
            gap_end = b1 - 1

            # 判断是否可以合并
            can_merge = True
            if gap_start <= gap_end:
                # 检查间隙内的所有值是否 >= t
                for i in range(gap_start, gap_end + 1):
                    if self.sig[i] < self.tau:
                        can_merge = False
                        break

            if can_merge:
                # 合并区间：取最小起始和最大结束
                merged[-1] = (min(a1, b1), max(a2, b2))
            else:
                merged.append(current)
        self.pulseInterval = []
        for s, e in merged:
            # 向左扩展左边界
            left = s
            while left > 0 and self.sig[left - 1] >= self.tau:
                left -= 1

            # 向右扩展右边界
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
        得到归类后的条纹，也就是最终形式的条纹
        针对的是单独的信号
        :param y: label
        :param pic_save: 是否保存可视化图片
        :return:
        """
        # 1. 得到csv文件得到stripes
        stripes = self._convert_stripe_csv_to_list()
        self._assign_stripes(stripes)
        self._get_stripe_interval()   # 得到条纹区间
        self.vis_stripe_interval(pic_save)
        self._get_pulse_interval()    # 得到脉冲区间
        self.vis_pulse_interval(y, pic_save)

    def vis_stripe_interval(self, pic_save=False):
        if not pic_save:
            return
        palette = sns.color_palette("tab20", n_colors=20)
        colors = cycle(palette.as_hex())
        fig, ax1 = plt.subplots(figsize=(14, 5))
        # 信号部分
        ax1.plot(self.sig, linewidth=1.5, color='#1f77b4', linestyle='-', label="Signal", zorder=3)
        ax1.hlines(self.tau, 0, len(self.sig), colors='#d62728', linestyles='-.',
                   linewidth=1.8, zorder=2,
                   label='Resting State Estimation')
        ax1.fill_between(np.arange(len(self.sig)), self.sig, 0., alpha=0.1, zorder=2)
        ax1.set_xlabel("Position", fontsize=26)
        ax1.set_ylabel("Amplitude", fontsize=26)
        ax1.tick_params(axis='y')
        # 区间部分
        for interval in self.stripeInterval:
            ax1.axvspan(interval[0], interval[1], color='red', alpha=0.1)

        # 绘制条纹
        ax2 = ax1.twinx()
        for master, secondaries in self.pairs:
            color = next(colors)
            # 主条纹
            ax2.plot([master['x'], master['x']], [0, master['L']],
                     color=color, linewidth=2)
            # 次条纹
            for secondary in secondaries:
                ax2.plot([secondary['x'], secondary['x']], [0, secondary['L']],
                         color=color, linewidth=1, linestyle='--')
        ax2.set_ylabel("Iterations", fontsize=26)
        ax2.tick_params(axis='y')
        plt.savefig(f"pic/stripeInterval/{self.i}.png")
        plt.close()

    def vis_pulse_interval(self, y, pic_save=False):
        """
        比较估计和标注区间
        :param y: 标注区间
        :param pic_save: 是否保存可视化图片
        :return:
        """
        if not pic_save:
            return
        mark = y[y[:, 0] == self.i]
        plt.figure(figsize=(15, 4))
        plt.plot(self.sig, label='Signal', color='gray', alpha=0.8)

        # 绘制预测区间
        if self.pulseInterval is not None:
            for interval in self.pulseInterval:
                plt.axvspan(interval[0], interval[1], color='red', alpha=0.3)

        # 绘制标注区间
        if len(mark) != 0:
            for m in mark:
                plt.axvspan(m[1], m[2], color='green', alpha=0.3)

        # 自定义图例（避免重复标签）
        legend_elements = [
            Patch(facecolor='red', alpha=0.3, label='Predicted'),
            Patch(facecolor='green', alpha=0.3, label='Label')
        ]
        plt.legend(handles=legend_elements)

        plt.xlabel('Position', fontsize=18)
        plt.ylabel('Amplitude', fontsize=18)
        plt.savefig(f"pic/pulseInterval/{self.i}.png")
        plt.close()
