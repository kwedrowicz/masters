import matplotlib.pyplot as plt
from stringcase import snakecase
import os


def extract_base_name(file_path):
    base = os.path.basename(file_path)
    return os.path.splitext(base)[0]


def draw_plots(features, estimator, labels, title, total, representatives, value_tags=True):
    plt_x, plt_y = get_coordinates(features)

    draw_pca(plt_x, plt_y, title)
    draw_clustered(plt_x, plt_y, title, estimator, labels, total, representatives, value_tags)


def get_coordinates(features):
    plt_x = features[:, 0]
    if features.shape[1] > 1:
        plt_y = features[:, 1]
    else:
        plt_y = [1] * features.shape[0]
    return plt_x, plt_y


def draw_pca(x, y, title):
    plt.scatter(x, y)
    plt.title(title + " PCA")
    plt.savefig("resources/plots/" + snakecase(title) + "_pca.png")
    plt.close()


def draw_clustered(x, y, title, estimator, labels, total, representatives, value_tags):
    estimator_name = estimator.__class__.__name__

    plt.scatter(x, y, c=labels)
    plt.title(title + " Clustered(" + estimator_name + ")")

    for representative_id in representatives:
        rep = total.loc[representative_id]
        i = total.index.tolist().index(representative_id)
        plt.annotate(
            # "{} -> {} ({})".format(rep.name, rep.total, rep.cluster),
            # rep.total,
            rep.total if value_tags else int(rep.name),
            xy=(x[i], y[i]),
            xytext=(-10, 20),
            textcoords='offset points',
            arrowprops=dict(arrowstyle='->')
        )

    plt.savefig("resources/plots/" + snakecase(title) + "_" + snakecase(estimator_name) + "_clustered.png")
    plt.close()
