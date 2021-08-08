from pandas.plotting import scatter_matrix
from matplotlib import pyplot


def box(dataset):
    dataset.plot(kind='box', subplots=True, layout=(
        2, 2), sharex=False, sharey=False)
    pyplot.show()


def historgram(dataset):
    dataset.hist()
    pyplot.show()


def scatter_matrix(dataset):
    scatter_matrix(dataset)
    pyplot.show()


def box_results(results, names):
    pyplot.boxplot(results, labels=names)
    pyplot.title('Algorithm Comparison')
    pyplot.show()
