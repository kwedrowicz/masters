from flowshop.functions.conditional_print import print_if


def set_divide(data, quantile, printable):

    data = data.assign(total=data.sum(axis=1).values)
    print_if("Added total row", boolean=printable)

    quantile = data['total'].quantile(quantile)
    mask = data['total'] <= quantile
    bests = data[mask]
    worsts = data[~mask]
    print_if("Divided into {} best and {} worst runs".format(bests.shape[0], worsts.shape[0]), boolean=printable)

    return bests, worsts
