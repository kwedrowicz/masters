import pandas

from lumberjack import parameters

df = pandas.read_csv('resources/out.csv', sep=';', usecols=parameters.cols, dtype='int')
runs = df.pivot(index='runId', columns='instanceId', values='score')
runs = runs.fillna(0).astype(int)

runs.to_csv('resources/lumberjack.csv')

