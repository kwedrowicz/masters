import pandas
from numpy import array, inf
from scipy.cluster.vq import kmeans, whiten

df = pandas.read_csv('resources/out.csv', sep=';', usecols=['runId', 'instanceId', 'score'])
runs = {}
for index, row in df.iterrows():
    if not row['runId'] in runs.keys():
        runs[row['runId']] = {}
    runs[row['runId']][row['instanceId']] = row['score']

for key in runs.keys():
    if len(runs[key]) != 31:
        del runs[key]

list = []
sorted_keys = sorted(runs.keys())
for key in sorted_keys:
    list.append(runs[key].values())

features = array(list)
whitened = whiten(features)
k = 10
means = kmeans(whitened, k)
for mean in means[0]:
    minDiff = inf
    minIndex = -1
    for i in range(0, len(whitened)-1):
        curDiff = 0
        for j in range(0, 30):
            curDiff += abs(mean[j]-whitened[i][j])
        if curDiff < minDiff:
            minDiff = curDiff
            minIndex = i
    print sorted_keys[minIndex]

