import matplotlib.pyplot as plt
from stringcase import titlecase, snakecase
import os


def extract_base_name(file_path):
    base = os.path.basename(file_path)
    return os.path.splitext(base)[0]


def draw_plots(features, estimator, file_path, labels):

    print("Plot")
    print(features)

    plt_x = features[:, 0]
    if features.shape[1] > 1:
        plt_y = features[:, 1]
    else:
        plt_y = [1] * features.shape[0]

    base_name = extract_base_name(file_path)
    estimator_name = estimator.__class__.__name__

    plt.scatter(plt_x, plt_y)
    plt.title(titlecase(base_name) + " PCA")
    plt.savefig("../resources/plots/" + base_name + "_pca.png")
    plt.close()

    plt.scatter(plt_x, plt_y, c=labels)
    plt.title(titlecase(base_name) + " Clustered(" + estimator_name + ")")
    plt.savefig("../resources/plots/" + base_name + "_" + snakecase(estimator_name) + "_clustered.png")
    plt.close()
