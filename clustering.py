import pandas
from numpy import array
from scipy.cluster.vq import kmeans, whiten

from functions import map_runs, get_closest_run_to_mean

df = pandas.read_csv('resources/out.csv', sep=';', usecols=['runId', 'instanceId', 'score'])
runs = map_runs(df)

dictListed = []
sorted_keys = sorted(runs.keys())
for key in sorted_keys:
    dictListed.append(runs[key].values())

features = array(dictListed)
whitened = whiten(features)
k = 10
means = kmeans(whitened, k)
for mean in means[0]:
    print get_closest_run_to_mean(mean, whitened, sorted_keys)
