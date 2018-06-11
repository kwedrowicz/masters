import pandas

from flowshop.functions.conditional_print import print_if
from flowshop.functions.data_preparations import remove_zero_rows, remove_duplicates


def prepare_base_data():
    df = pandas.read_csv('resources/csv/flowshop_raw.csv', sep=';', usecols=['run_id', 'instance_id', 'score'],
                         dtype='float')
    runs = df.pivot(index='run_id', columns='instance_id', values='score')
    runs = runs.fillna(0).astype(float)
    print_if("Generated pivoted table", boolean=True)

    runs = remove_zero_rows(runs, True)
    runs = remove_duplicates(runs, True)

    return runs


def prepare_belgian_data():
    runs = pandas.read_csv('resources/csv/space.csv', sep=',', index_col='run_id', dtype='float')
    runs = runs.fillna(0).astype(float)
    print_if("Loaded data", boolean=True)

    runs = remove_zero_rows(runs, True)
    runs = remove_duplicates(runs, True)

    return runs


baseRuns = prepare_base_data()
totalRuns = baseRuns.assign(total=baseRuns.sum(axis=1).values)[['total']]
belgianRuns = prepare_belgian_data()

joined = belgianRuns.join(totalRuns).dropna()
joined.index = joined.index.map(int)

joined.to_csv('resources/csv/space_total.csv')
