# import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import FigureCanvas


def make_figure():

    fig = Figure()
    canvas = FigureCanvas(fig)

    return fig


def subplots(*args, **kwargs):

    fig = make_figure()
    axs = fig.subplots(*args, **kwargs)

    return fig, axs
