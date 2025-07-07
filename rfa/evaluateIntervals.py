import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm


class EvaluateIntervals:

    def __init__(self, pulses, marks, m):
        """
        load estimate pulse intervals and labeled intervals
        :param pulses: pulse intervals of a group of signals, list
        :param marks: labeled intervals of a group of signals, list
        :param m: the number of samples
        """
        self.a = pulses
        self.b = marks
        self.m = m
        print(f"The number of samples is {m}")
        # assert len(self.b) != m, "Static signals exist!"
        # assert len(self.a) != m, "Empty estimation exists!"
        # assert len(self.a) != len(self.b), "Interval lists' length unequal!"

    @staticmethod
    def calculate_iou_matrix(a, b):
        """
        计算两组区间之间的IoU矩阵
        参数：
            a : np.ndarray, 形状为(m, 2)的数组，表示m个区间[开始, 结束]
            b : np.ndarray, 形状为(n, 2)的数组，表示n个区间[开始, 结束]
        返回：
            iou_matrix : np.ndarray, 形状为(m, n)的IoU矩阵
        """
        # 分解区间起点和终点
        if min(len(a), len(b)) == 0:
            return np.nan
        a, b = np.array(a), np.array(b)
        a_start = a[:, 0]  # 形状(m,)
        a_end = a[:, 1]
        b_start = b[:, 0]  # 形状(n,)
        b_end = b[:, 1]

        # 扩展维度以便广播计算
        a_start = a_start[:, np.newaxis]  # 形状(m, 1)
        a_end = a_end[:, np.newaxis]  # 形状(m, 1)
        b_start = b_start[np.newaxis, :]  # 形状(1, n)
        b_end = b_end[np.newaxis, :]  # 形状(1, n)

        # 计算交叠区域的起止点
        overlap_start = np.maximum(a_start, b_start)  # 形状(m, n)
        overlap_end = np.minimum(a_end, b_end)  # 形状(m, n)

        # 计算交叠长度（处理无交叠情况）
        overlap_length = np.clip(overlap_end - overlap_start, a_min=0, a_max=None)

        # 计算各区间长度
        a_lengths = a_end - a_start  # 形状(m, 1)
        b_lengths = b_end - b_start  # 形状(1, n)

        # 计算合并区域长度
        union_length = a_lengths + b_lengths - overlap_length

        # 计算IoU（处理除零情况）
        iou_matrix = np.divide(
            overlap_length,
            union_length,
            out=np.zeros_like(overlap_length, dtype=np.float32),
            where=(union_length != 0)
        )

        return np.round(iou_matrix, 2)

    def match_intervals(self, a, b, iou_threshold=0.5):
        """
        基于匈牙利算法的区间匹配
        参数：
            a : np.ndarray, 形状(m,2) 预测区间数组
            b : np.ndarray, 形状(n,2) 真实区间数组
            iou_threshold : float IoU阈值，低于此值的匹配视为无效
        返回：
            matches : list 有效匹配列表 [(pred_idx, true_idx, iou), ...]
            fp : list 未匹配的预测区间索引
            fn : list 未匹配的真实区间索引
        """
        # 计算IoU矩阵
        iou_matrix = self.calculate_iou_matrix(a, b)  # 使用之前实现的函数
        if np.isnan(iou_matrix).any():
            return [], 0, 0
        # 转换为成本矩阵
        cost_matrix = 1 - iou_matrix

        # 使用匈牙利算法找到最小成本匹配
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # 筛选有效匹配（IoU >= 阈值）
        valid_matches = []
        for r, c in zip(row_ind, col_ind):
            iou = iou_matrix[r, c]
            if iou >= iou_threshold:
                valid_matches.append((r, c, iou))

        # 统计未匹配项
        matched_pred = set(r for r, _, _ in valid_matches)
        matched_true = set(c for _, c, _ in valid_matches)

        fp = [i for i in range(len(self.a)) if i not in matched_pred]
        fn = [j for j in range(len(self.b)) if j not in matched_true]

        return valid_matches, fp, fn

    def cal_map(self, alpha):
        """
        calculate the P and R of a series of intervals and marks, given alpha.
        :param alpha: iou threshold, 0.5:0.05:0.95
        :return:
        """
        P, R = [], []
        total_match = 0
        total_fp = 0
        total_fn = 0
        # TP, FP, FN = 0, 0, 0

        for i in tqdm(range(self.m)):
            pulse_intervals = self.a[i]
            mark_ = self.b[i]
            if pulse_intervals is None:
                if len(mark_) == 0:
                    TP += 1
                else:
                    FN += 1
                continue
            mark = []
            for item in mark_:
                left, right = item
                if left != right:
                    if right - left > 30:  # filter
                        mark.append([left, right])

            matches, fp, fn = self.match_intervals(pulse_intervals, mark, alpha)
            if not matches:
                continue
            TP = len(matches)
            FP = len(fp)
            FN = len(fn)
            total_match += TP
            total_fn += FN
            total_fp += FP
            precision = round(TP / (TP + FP), 2)
            recall = round(TP / (TP + FN), 2)
            P.append(precision)  # for radar char
            R.append(recall)
        Precision = round(total_match / (total_match + total_fp), 2)
        Recall = round(total_match / (total_match + total_fn), 2)
        return Precision, Recall

    def map(self):
        """
        cal mAP
        :return: precision, tmp1; recall, tmp2.
        """
        tmp1 = []
        tmp2 = []
        for a in np.arange(0.5, 1, 0.05):
            p, r = self.cal_map(alpha=a)
            tmp1.append(p)
            tmp2.append(r)
        return tmp1, tmp2
