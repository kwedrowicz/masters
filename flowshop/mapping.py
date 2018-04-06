import pandas
from flowshop.conditional_print import print_if


def prepare_runs(source_path='../resources/csv/flowshop_raw.csv', quantile=.7, printable=False, flattened=False):
    if flattened:
        runs = pandas.read_csv(source_path, sep=',', index_col='run_id', dtype='float')
        print(runs)
    else:
        df = pandas.read_csv(source_path, sep=';', usecols=['run_id', 'instance_id', 'score'],
                             dtype='int')
        runs = df.pivot(index='run_id', columns='instance_id', values='score')
    runs = runs.fillna(0).astype(float)
    print_if("Generated pivoted table", boolean=printable)

    zero_rows_count = runs.shape[0]
    runs = runs[(runs.T != 0).any()]
    non_zero_rows_count = runs.shape[0]

    print_if("Removed {} zero rows from {} rows".format(zero_rows_count-non_zero_rows_count, zero_rows_count), boolean=printable)

    all_rows = runs.shape[0]
    deduplicated_runs = runs.drop_duplicates()
    deduplicated_rows = deduplicated_runs.shape[0]
    print_if("Removed {} from {} rows".format(all_rows - deduplicated_rows, all_rows), boolean=printable)

    deduplicated_runs = deduplicated_runs.assign(total=deduplicated_runs.sum(axis=1).values)
    print_if("Added total row", boolean=printable)

    quantile = deduplicated_runs['total'].quantile(1 - quantile)
    mask = deduplicated_runs['total'] <= quantile
    best_runs = deduplicated_runs[mask]
    worst_runs = deduplicated_runs[~mask]
    print_if("Divided into {} best and {} worst runs".format(best_runs.shape[0], worst_runs.shape[0]), boolean=printable)

    best_runs.to_csv('../resources/csv/flowshop_best.csv')
    worst_runs.to_csv('../resources/csv/flowshop_worst.csv')
    print_if("Saved runs to CSV", boolean=printable)
