from flowshop.functions.conditional_print import print_if


def remove_zero_rows(data, printable):
    zero_rows_count = data.shape[0]
    data = data[(data.T != 0).any()]
    non_zero_rows_count = data.shape[0]

    print_if("Removed {} zero rows from {} rows".format(zero_rows_count - non_zero_rows_count, zero_rows_count),
             boolean=printable)

    return data


def remove_duplicates(data, printable):
    count_before = data.shape[0]
    deduplicated_data = data.drop_duplicates()
    count_after = deduplicated_data.shape[0]
    print_if("Removed {} duplicates from {} rows".format(count_before - count_after, count_before),
             boolean=printable)

    return deduplicated_data
