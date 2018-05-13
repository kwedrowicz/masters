from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from flowshop.functions.conditional_print import print_if


def pca_reduce(data, variance):
    pca = PCA(variance)
    x = StandardScaler().fit_transform(data.values.astype(float))
    pca.fit(x)
    print_if("PCA reduced to {} dimensions".format(len(pca.explained_variance_ratio_)), boolean=True)
    print_if("PCA variance ratio {}".format(pca.explained_variance_ratio_), boolean=True)

    return pca.fit_transform(x)
