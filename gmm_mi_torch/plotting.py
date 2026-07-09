"""Plotting utilities for gmm-mi-torch.

Forked from GMM-MI (https://github.com/dpiras/GMM-MI) by Davide Piras: the
panel helpers and the GMM error-ellipse machinery are his, imported with
their upstream commit history (see the repository log) and pruned to the
subset used by the walkthrough notebook. `plot_contours` is reimplemented
here with matplotlib/scipy so the package needs no chainconsumer install.
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable


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
def calculate_summary_and_significance(estimates):
    """Given an array of estimates, calculate their sample mean, the error on the mean,
    and the significance with respect to 0.
    
    Parameters
    ----------
    estimates : 1D array-like
        The value of all estimates for a certain number of trials. 
        
    Returns
    -------
    mean_value : float
        The mean of the estimates.
    mean_error : float
        The error on the mean of the estimates.
    significance : float
        The distance of the mean with respect to 0.
    """
    trials_number = len(estimates)
    mean_value = np.mean(estimates)
    mean_error = np.sqrt(np.var(estimates, ddof=1))/np.sqrt(trials_number)
    # significance is with respect to 0
    significance = np.mean(estimates) / mean_error
    return mean_value, mean_error, significance


def histogram_estimates(estimates, ax=None, bins=20, alpha=1, title='',
                   legendsize=25, color='salmon', lw=3, histtype='step'):
    """Plot the histogram of an array of estimates, centred on 0.
    
    Parameters
    ----------
    estimates : 1D array-like
        The estimates whose histogram we plot. 
    ax : instance of the axes.Axes class from pyplot, default=None
        The panel where to draw the histogram. 
    bins : int, default=20
        Number of histogram bins.
    alpha : float, default=1
        Transparency of the histogram line. Must be between 0 and 1.
    title : string, default=''
        Title of the panel.
    legendsize : int, default=25
         Font size of the legend.
    color : string, default='salmon'
        Histogram color.
    lw : int, default=3
        Line width of the histogram line.  
    histtype : {'bar', 'barstacked', 'step', 'stepfilled'}, default='step'
        The type of histogram to draw.
        
    Returns
    -------
    None
    """    
    fig, ax = choose_ax(ax)
    mean_value, mean_error, significance = calculate_summary_and_significance(estimates)
    ax.hist(estimates.flatten(), alpha=alpha, bins=bins, color=color, histtype=histtype, lw=lw,
            label=(f'MI values, {significance:.1f}' + '$\\sigma$,\n'
                   + f'{mean_value:.4f}' + '$\\pm$' + f'{mean_error:.4f}')
            )
    ax.axvline(0, ls='--', lw=lw, c='black')
    set_ticksize(ax)
    set_titles(ax, xlabel='MI [nat]', ylabel='Counts', title=title)
    set_legend(ax, fontsize=legendsize)


def create_heatmap(data, row_labels, col_labels, fsize=30, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data : array-like of shape (n_rows, n_columns)
        A 2D numpy array containing the data values that are reported.
    row_labels : list
        A list with the labels for the rows.
    col_labels : list
        A list with the labels for the columns.
    ax : `matplotlib.axes.Axes` instance, default=None
        The panel in which the heatmap is plotted. 
        If not provided, use current axes or create a new one.
    cbar_kw : dict, default={}
        Contains the arguments to `matplotlib.Figure.colorbar`.
    cbarlabel : string, default=""
        The label for the colorbar.
    **kwargs
        All other arguments are forwarded to `imshow`.
        
    Returns
    -------
    im : AxesImage object
        The output of `imshow`.
    cbar : `colorbar` object
        The colorbar.
    """

    fig, ax = choose_ax(ax, figsize=(24, 8))
       
    # plot the heatmap
    im = ax.imshow(data, **kwargs)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.5)
    # create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, cax=cax, **cbar_kw)
    cbar.ax.tick_params(labelsize=20, length=10, width=2) 
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", fontsize=30)

    # show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels(labels=col_labels, fontsize=fsize)
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels(labels=row_labels, fontsize=fsize)

    # set the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False, length=10, width=2)

    # Turn spines off and create white grid.
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)  
    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     save_fig=False, save_path="", 
                     threshold=0.01, **textkw):
    """Used to annotate a previously-created heatmap.

    Parameters
    ----------
    im : AxesImage
        The image to be annotated.
    data : array-like of shape (n_rows, n_columns)
        Data used to annotate.  If None, the image's data is used.
    valfmt : either str, or `matplotlib.ticker.Formatter`, default="{x:.2f}"
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.
    save_fig : bool, default=False
        Whether to save the annotated heatmap or not.
    save_path : string, default=""
        If save_fig is True, the path where to save the figure.
    threshold : float, default=0.01
        All values of MI below this are displayed as 0.
    **textkw
        All other arguments are forwarded to each call to `text` used to create the text labels.
        
    Returns
    -------
    texts : list
        Contains all the text that is annotated onto the heatmap.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # set default alignment to center, but allow it to be overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # loop over the data and create a `Text` for each "pixel".
    # change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            # ignore if value is too small
            if data[i, j] < threshold:
                text = im.axes.text(j, i, matplotlib.ticker.StrMethodFormatter("{x:.0f}")(data[i, j], None), **kw) 
            else:
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)
            
    if save_fig:
        plt.savefig(save_path, bbox_inches='tight')
    return texts
