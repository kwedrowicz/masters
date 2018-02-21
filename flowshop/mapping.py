import pandas

df = pandas.read_csv('../resources/flowshop_raw.csv', sep=';', usecols=['run_id', 'instance_id', 'score'], dtype='int')
runs = df.pivot(index='run_id', columns='instance_id', values='score')
runs = runs.fillna(0).astype(int)

all_rows = runs.shape[0]

deduplicated_runs = runs.drop_duplicates()
deduplicated_rows = deduplicated_runs.shape[0]
print("Removed {} from {} rows".format(all_rows-deduplicated_rows, all_rows))

deduplicated_runs = deduplicated_runs.assign(total=deduplicated_runs.sum(axis=1).values)
print("Added total row")

quantile7 = deduplicated_runs['total'].quantile(.7)

mask = deduplicated_runs['total'] >= quantile7
best_runs = deduplicated_runs[mask]
worst_runs = deduplicated_runs[~mask]

best_runs.to_csv('../resources/flowshop_best.csv')
worst_runs.to_csv('../resources/flowshop_worst.csv')
