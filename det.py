from rfa import GetPP, StripeManager, EvaluateIntervals
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


# plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'axes.titlesize': 14,
    'savefig.dpi': 300,
    'font.weight': 'bold'
})

# # example input
# x = np.load("example_signal.npy")
# y = np.load("label.npy")

# # whole data
x = np.load("D:/code/fiber/label/x.npy")
y = np.load("D:/code/fiber/label/stageI.npy")

csv_dp = os.path.abspath("dataframe/fstripesFilter/")
vis = False
predictions = []  # prediction intervals
marks = []  # labeled intervals
m = x.shape[0]
gp = GetPP(csv_dp)

if not os.path.exists("pic/fit"):
    os.makedirs("pic/fit")

if not os.path.exists("pic/ScatterPeaks"):
    os.makedirs("pic/ScatterPeaks")

if not os.path.exists("pic/TrackedStripes"):
    os.makedirs("pic/TrackedStripes")

if not os.path.exists("pic/ClusteredStripes"):
    os.makedirs("pic/ClusteredStripes")

if not os.path.exists("pic/stripeInterval"):
    os.makedirs("pic/stripeInterval")

if not os.path.exists("dataframe/fstripesFilter"):
    os.makedirs("dataframe/fstripesFilter")

if not os.path.exists("dataframe/fstripes"):
    os.makedirs("dataframe/fstripes/")  # filtered stripes

if not os.path.exists("pic/pulseInterval"):
    os.makedirs("pic/pulseInterval")

for i in tqdm(range(m)):
    gp.run(sig=x[i], i=i, vis=vis)  # get tracked stripes dataframe
    manager = StripeManager(i=i, sig=x[i], csv_dp=csv_dp, )
    manager.pipeline1(y, vis=vis)
    mark = y[y[:, 0] == i]
    marks.append(mark[:, 1:].tolist())
    predictions.append(manager.pulseInterval)

ei = EvaluateIntervals(predictions, marks, m)
p, r = ei.map()
print("Precisions: ", p)
print("Recall: ", r)
