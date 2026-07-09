"""Plotting utilities for gmm-mi-torch.

Forked from GMM-MI (https://github.com/dpiras/GMM-MI) by Davide Piras: the
panel helpers and the GMM error-ellipse machinery are his, imported with
their upstream commit history (see the repository log) and pruned to the
subset used by the walkthrough notebook. `plot_contours` is reimplemented
here with matplotlib/scipy so the package needs no chainconsumer install.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def choose_ax(ax, figsize=(15, 10)):
    """Select panel where to plot. Normally returns the input panel;
    if that is None, create a new panel with the specified figsize.

    Parameters
    ----------
    ax : instance of the axes.Axes class from pyplot
        The input panel.
    figsize : tuple of two integers, default=(15,10)
        The width and height of a newly created panel, in inches.

    Returns
    -------
    fig: instance of the figure.Figure class from pyplot
        The output figure.
    ax : instance of the axes.Axes class from pyplot
        The output panel.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = None
    return fig, ax


def set_ticksize(ax, axis='both', which='major', labelsize=20, size=6):
    """Set the size of the ticks in a panel.

    Parameters
    ----------
    ax : instance of the axes.Axes class from pyplot
        The panel whose ticks need to be set.
    axis : {'both', 'x', 'y'}, default='both'
        The axis (or axes) to which the parameters are applied.
    which : {'major', 'minor', 'both'}, default='major'
        The group of ticks to which the parameters are applied.
    labelsize : int, default=20
        Font size of the tick label.
    size : int, default=6
        Size of the ticks.

    Returns
    -------
    None
    """
    ax.tick_params(axis=axis, which=which, labelsize=labelsize, size=size)


def set_titles(ax, xlabel='', ylabel='', title='', fontsize=30, color='black'):
    """Set the titles in a panel.

    Parameters
    ----------
    ax : instance of the axes.Axes class from pyplot
        The panel whose titles need to be set.
    xlabel : string, default=''
        The x-label.
    ylabel : string, default=''
        The y-label.
    title : string, default=''
        The title of the panel.
    fontsize : int, default=30
        Font size of all titles and labels.
    color : string, default='black'
        Color of all titles and labels.

    Returns
    -------
    None
    """
    ax.set_xlabel(xlabel, fontsize=fontsize, color=color)
    ax.set_ylabel(ylabel, fontsize=fontsize, color=color)
    ax.set_title(title, fontsize=fontsize, color=color)


def set_legend(ax, fontsize=25, frameon=False, loc='best', handles=None, labels=None):
    """Set the legend in a panel.

    Parameters
    ----------
    ax : instance of the axes.Axes class from pyplot
        The panel whose legend need to be set.
    fontsize : int, default=25
        Font size of the legend.
    frameon : bool, default=False
        Whether to frame the legend or not.
    loc : string, default='best'
        Position of the legend.
    handles : sequence of `.Artist` (see pyplot docs), default=None
        The handles, only used when playing around with the order of the legend (quite rarely),
        and together with labels.
    labels : list of str, default=None
        The labels, only used when playing around with the order of the legend (quite rarely),
        and together with handles.

    Returns
    -------
    None
    """
    if handles is not None:
        ax.legend(handles, labels, fontsize=fontsize, frameon=frameon, loc=loc)
    else:
        ax.legend(fontsize=fontsize, frameon=frameon, loc=loc)


def calculate_ellipse_params(covariance):
    """Given a covariance matrix in 2D, calculate the orientation (angle), height and width of
    the error ellipse representing an iso-contour of the corresponding Gaussian distribution.
    Based on https://www.visiondummy.com/2014/04/draw-error-ellipse-representing-covariance-matrix/.

    Parameters
    ----------
    covariance : array-like of shape (2, 2)
        The covariance matrix of the Gaussian distribution.

    Returns
    -------
    angle : float
        Orientation of the error ellipse.
    width : float
        Width of the error ellipse.
    height : float
        Height of the error ellipse.
    """
    U, s, Vt = np.linalg.svd(covariance)
    # using arctan2 instead of arctan needed to take into account sign
    angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
    width, height = 2*np.sqrt(s)
    return angle, width, height


def find_contours(contour_levels):
    """Associate the user-required sigma levels to the actual size of the error ellipse.
    The 1-sigma, 2-sigma, and/or 3-sigma contours are obtained from a chi2 table, and based on
    https://www.visiondummy.com/2014/04/draw-error-ellipse-representing-covariance-matrix/.

    Parameters
    ----------
    contour_levels : list
        The user-required contours to display, in sigmas.
        Only 1, 2 and 3 are accepted in this list.

    Returns
    -------
    ranges : list
        The size of the error ellipse corresponding to the input contour levels.
    """
    if any(cl not in [1,2,3] for cl in contour_levels):
        raise ValueError(f'Contour levels must be either 1, 2 and 3; current contour levels are {contour_levels}.')
    assert len(contour_levels) <= 3, 'Length of contour levels cannot exceed 3'
    ranges_dict = {1: np.sqrt(0.77), 2: np.sqrt(5.991), 3: np.sqrt(10.597)}
    ranges = []
    for cl in contour_levels:
        ranges.append(ranges_dict[cl])
    return ranges


def draw_ellipse(mean, covariance, weight, ax=None, fill=False, marker='X',
                 contour_levels=[2], weight_size=1000, label=None, component_count=0, **kwargs):
    """Draw the error ellipse corresponding to a given mean vector and Gaussian covariance matrix.
    Since this is done in the context of Gaussian mixture models (GMMs), the weight is also indicated
    with a marker whose size is proportional to the weight value. Based on
    https://www.visiondummy.com/2014/04/draw-error-ellipse-representing-covariance-matrix/.

    Parameters
    ----------
    mean : array-like of shape (2)
        The mean of the Gaussian distribution.
    covariance : array-like of shape (2, 2)
        The covariance matrix of the Gaussian distribution.
    weight : float
        The weight associated with this GMM component.
    ax : instance of the axes.Axes class from pyplot, default='None'
        The panel where to draw the ellipse.
    fill : bool, default=False
        Whether to fill the contours or not.
    marker : string, default='X'
        Marker type for the centroid of the ellipse.
    contour_levels : list of integers, default=[2]
        The sigma levels whose contour are going to be drawn.
        The integers can be only 1, 2 and 3.
    weight_size : float, default=1000
        The multiplier for the marker size (also proportional to the weight).
    label : string, default=None
        Label associated to the ellipse.
    component_count : int, default=0
        A counter so that the label is printed only for the first component.
    kwargs : dictionary
        The extra keyword arguments to pass to the plotting function.

    Returns
    -------
    None
    """
    assert covariance.shape == (2, 2), 'Each covariance matrix must be of shape (2, 2)!'
    # convert covariance to the parameters of the corresponding error ellipse
    angle, width, height = calculate_ellipse_params(covariance)
    # find the contour sigma levels
    contours = find_contours(contour_levels)
    # draw the ellipse
    for contour in contours:
        ax.add_patch(Ellipse(mean, contour*width, contour*height, angle=angle, fill=fill,
                             label=label if component_count==0 else "", **kwargs))
        ax.scatter(mean[0], mean[1], s=weight_size*weight, marker=marker, **kwargs)


def _kde_contours_2d(samples, x_range, y_range, ax, color, mass_levels,
                     gridsize, shade_alpha, linewidths, seed=0):
    """Draw KDE contours of 2-D samples at the given probability-mass levels."""
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(samples.T)
    xx, yy = np.meshgrid(np.linspace(*x_range, gridsize),
                         np.linspace(*y_range, gridsize))
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    # density thresholds enclosing each mass level, estimated from the KDE
    # evaluated at (a subsample of) the samples themselves
    pts = samples
    if len(pts) > 5000:
        keep = np.random.default_rng(seed).choice(len(pts), 5000, replace=False)
        pts = pts[keep]
    dens = kde(pts.T)
    thresholds = sorted(np.quantile(dens, 1.0 - np.asarray(mass_levels)))
    if shade_alpha:
        ax.contourf(xx, yy, zz, levels=[*thresholds, zz.max() * 1.01],
                    colors=[color], alpha=shade_alpha)
    ax.contour(xx, yy, zz, levels=thresholds, colors=[color],
               linewidths=linewidths)


def plot_contours(input_samples, fitted_samples, parameters=None, extents=None,
                  mass_levels=(0.68, 0.95), gridsize=80, shade_alpha=0.4,
                  linewidths=2, colors=('C0', 'salmon'),
                  labels=('Input data', 'Fitted model'), figsize=None):
    """Corner-style comparison of the input data with samples from the fitted model.

    Every pair of variables gets a panel with KDE contours enclosing the
    requested probability-mass levels for both sample sets; the diagonal
    shows the 1-D KDE marginals. Reimplemented with matplotlib/scipy for
    gmm-mi-torch (upstream GMM-MI drew these contours with chainconsumer).

    Parameters
    ----------
    input_samples : array-like of shape (n_samples, n_features)
        The input samples that were used to fit the GMM.
    fitted_samples : array-like of shape (n_samples, n_features)
        The samples obtained from the fitted GMM model.
    parameters : list of str, default=None
        Variable names for the axis labels; defaults to X1, X2, ...
    extents : dictionary, default=None
        Ranges of each variable, keyed by its name in `parameters`.
    mass_levels : tuple of floats, default=(0.68, 0.95)
        Probability-mass levels enclosed by the contours.
    gridsize : int, default=80
        KDE evaluation grid points per axis.
    shade_alpha : float, default=0.4
        Alpha for the filled contour regions; 0 disables filling.
    linewidths : float, default=2
        Width of the contour lines.
    colors : tuple of two colors, default=('C0', 'salmon')
        Colors for the input data and the fitted model.
    labels : tuple of two str, default=('Input data', 'Fitted model')
        Legend labels for the two sample sets.
    figsize : tuple, default=None
        Figure size; defaults to 3.5 inches per variable.

    Returns
    -------
    fig: instance of the figure.Figure class from pyplot
        The output figure.
    """
    input_samples = np.asarray(input_samples)
    fitted_samples = np.asarray(fitted_samples)
    d = input_samples.shape[1]
    if parameters is None:
        parameters = [f'X{i + 1}' for i in range(d)]
    ranges = []
    for j, name in enumerate(parameters):
        if extents is not None and name in extents:
            ranges.append(tuple(extents[name]))
        else:
            lo = min(input_samples[:, j].min(), fitted_samples[:, j].min())
            hi = max(input_samples[:, j].max(), fitted_samples[:, j].max())
            pad = 0.05 * (hi - lo)
            ranges.append((lo - pad, hi + pad))
    if figsize is None:
        figsize = (3.5 * d, 3.5 * d)
    fig, axes = plt.subplots(d, d, figsize=figsize, squeeze=False)
    for i in range(d):
        for j in range(d):
            ax = axes[i][j]
            if j > i:
                ax.set_visible(False)
                continue
            for samples, color, label in zip((input_samples, fitted_samples),
                                             colors, labels):
                if i == j:
                    from scipy.stats import gaussian_kde
                    grid = np.linspace(*ranges[i], 512)
                    ax.plot(grid, gaussian_kde(samples[:, i])(grid),
                            color=color, lw=linewidths, label=label)
                else:
                    _kde_contours_2d(samples[:, [j, i]], ranges[j], ranges[i],
                                     ax, color, mass_levels, gridsize,
                                     shade_alpha, linewidths)
            ax.set_xlim(ranges[j])
            if i == j:
                ax.set_yticks([])
            else:
                ax.set_ylim(ranges[i])
            if i == d - 1:
                ax.set_xlabel(parameters[j], fontsize=14)
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(parameters[i], fontsize=14)
            elif i != j:
                ax.set_yticklabels([])
    handles, leg_labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, fontsize=14, frameon=False,
               loc='upper right')
    fig.tight_layout()
    return fig


def plot_gmm_contours(gmm, ax=None, fill=False, label='', xlabel='X1', ylabel='X2', **kwargs):
    """Draw the contour ellipses corresponding to a given Gaussian mixture model (GMM).

    Parameters
    ----------
    gmm : object with sklearn-style GMM attributes
        The GMM whose contours are going to be displayed; needs `means_`,
        `covariances_` and `weights_` (e.g. the `FittedGMM` returned by
        `estimate(..., return_fit=True)`).
    ax : instance of the axes.Axes class from pyplot
        The panel where to plot the samples.
    fill : bool, default=False
        Whether to fill the contours or not.
    label : string, default=''
        The legend label to associate to the contours.
    xlabel : string, default='X1'
        Label of the x-axis.
    ylabel : string, default='X2'
        Label of the y-axis.
    kwargs : dictionary
        The extra keyword arguments to pass to the plotting functions.

    Returns
    -------
    ax : instance of the axes.Axes class from pyplot
        The output panel.
    """
    means, covariances, weights = gmm.means_, gmm.covariances_, gmm.weights_
    for component_count, (mean, covariance, weight) in enumerate(zip(means, covariances, weights)):
        draw_ellipse(mean, covariance, weight, ax=ax, fill=fill,
                     component_count=component_count, label=label, **kwargs)
    return ax
