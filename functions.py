from numpy import inf


def map_runs(df):
    runs = {}
    for index, row in df.iterrows():
        if not row['runId'] in runs.keys():
            runs[row['runId']] = {}
        runs[row['runId']][row['instanceId']] = row['score']

    for key in list(runs):
        if len(runs[key]) != 31:
            del runs[key]
    return runs


def map_dict_to_list(runs):
    dict_listed = []
    sorted_keys = sorted(runs.keys())
    for key in sorted_keys:
        dict_listed.append(list(runs[key].values()))
    return dict_listed


def get_closest_run_to_mean(mean, whitened, sorted_keys):
    min_diff = inf
    min_index = -1
    for i in range(0, len(whitened) - 1):
        cur_diff = 0
        for j in range(0, 31):
            cur_diff += abs(mean[j] - whitened[i][j])
        if cur_diff < min_diff:
            min_diff = cur_diff
            min_index = i
    return sorted_keys[min_index]
