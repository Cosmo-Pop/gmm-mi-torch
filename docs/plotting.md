<!-- markdownlint-disable -->

<a href="../gmm_mi_torch/plotting.py"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `plotting.py`

Plotting utilities for gmm-mi-torch.

Forked from GMM-MI (https://github.com/dpiras/GMM-MI) by Davide Piras: the
panel helpers and the GMM error-ellipse machinery are his, imported with
their upstream commit history (see the repository log) and pruned to the
subset used by the walkthrough notebook. `plot_contours` is reimplemented
here with matplotlib/scipy so the package needs no chainconsumer install.

**On this page:** [`choose_ax`](#function-choose_ax) · [`set_ticksize`](#function-set_ticksize) · [`set_titles`](#function-set_titles) · [`set_legend`](#function-set_legend) · [`calculate_ellipse_params`](#function-calculate_ellipse_params) · [`find_contours`](#function-find_contours) · [`draw_ellipse`](#function-draw_ellipse) · [`plot_contours`](#function-plot_contours) · [`plot_gmm_contours`](#function-plot_gmm_contours) · [`calculate_summary_and_significance`](#function-calculate_summary_and_significance) · [`histogram_estimates`](#function-histogram_estimates) · [`swap_leg_handles`](#function-swap_leg_handles) · [`plot_multivariate_normal`](#function-plot_multivariate_normal) · [`plot_bias_chi2_histogram`](#function-plot_bias_chi2_histogram) · [`create_heatmap`](#function-create_heatmap) · [`annotate_heatmap`](#function-annotate_heatmap)

---

<a href="../gmm_mi_torch/plotting.py#L16"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `choose_ax`

```python
choose_ax(ax, figsize=(15, 10))
```

Select panel where to plot. Normally returns the input panel; if that is None, create a new panel with the specified figsize.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ax` | `instance of the axes.Axes class from pyplot` | - | The input panel. |
| `figsize` | `tuple of two integers` | `(15, 10)` | The width and height of a newly created panel, in inches. |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| - | `fig: instance of the figure.Figure class from pyplot` | The output figure. |
| `ax` | `instance of the axes.Axes class from pyplot` | The output panel. |

---

<a href="../gmm_mi_torch/plotting.py#L41"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `set_ticksize`

```python
set_ticksize(ax, axis='both', which='major', labelsize=20, size=6)
```

Set the size of the ticks in a panel.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ax` | `instance of the axes.Axes class from pyplot` | - | The panel whose ticks need to be set. |
| `axis` | `{'both', 'x', 'y'}` | `'both'` | The axis (or axes) to which the parameters are applied. |
| `which` | `{'major', 'minor', 'both'}` | `'major'` | The group of ticks to which the parameters are applied. |
| `labelsize` | `int` | `20` | Font size of the tick label. |
| `size` | `int` | `6` | Size of the ticks. |

---

<a href="../gmm_mi_torch/plotting.py#L60"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `set_titles`

```python
set_titles(ax, xlabel='', ylabel='', title='', fontsize=30, color='black')
```

Set the titles in a panel.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ax` | `instance of the axes.Axes class from pyplot` | - | The panel whose titles need to be set. |
| `xlabel` | `str` | `''` | The x-label. |
| `ylabel` | `str` | `''` | The y-label. |
| `title` | `str` | `''` | The title of the panel. |
| `fontsize` | `int` | `30` | Font size of all titles and labels. |
| `color` | `str` | `'black'` | Color of all titles and labels. |

---

<a href="../gmm_mi_torch/plotting.py#L83"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `set_legend`

```python
set_legend(
    ax,
    fontsize=25,
    frameon=False,
    loc='best',
    handles=None,
    labels=None,
)
```

Set the legend in a panel.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ax` | `instance of the axes.Axes class from pyplot` | - | The panel whose legend need to be set. |
| `fontsize` | `int` | `25` | Font size of the legend. |
| `frameon` | `bool` | `False` | Whether to frame the legend or not. |
| `loc` | `str` | `'best'` | Position of the legend. |
| `handles` | sequence of `.Artist` (see pyplot docs) | `None` | The handles, only used when playing around with the order of the legend (quite rarely), and together with labels. |
| `labels` | `list of str` | `None` | The labels, only used when playing around with the order of the legend (quite rarely), and together with handles. |

---

<a href="../gmm_mi_torch/plotting.py#L109"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `calculate_ellipse_params`

```python
calculate_ellipse_params(covariance)
```

Given a covariance matrix in 2D, calculate the orientation (angle), height and width of the error ellipse representing an iso-contour of the corresponding Gaussian distribution. Based on https://www.visiondummy.com/2014/04/draw-error-ellipse-representing-covariance-matrix/.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `covariance` | `array-like of shape (2, 2)` | The covariance matrix of the Gaussian distribution. |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `angle` | `float` | Orientation of the error ellipse. |
| `width` | `float` | Width of the error ellipse. |
| `height` | `float` | Height of the error ellipse. |

---

<a href="../gmm_mi_torch/plotting.py#L135"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `find_contours`

```python
find_contours(contour_levels)
```

Associate the user-required sigma levels to the actual size of the error ellipse. The 1-sigma, 2-sigma, and/or 3-sigma contours are obtained from a chi2 table, and based on https://www.visiondummy.com/2014/04/draw-error-ellipse-representing-covariance-matrix/.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `contour_levels` | `list` | The user-required contours to display, in sigmas. Only 1, 2 and 3 are accepted in this list. |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `ranges` | `list` | The size of the error ellipse corresponding to the input contour levels. |

---

<a href="../gmm_mi_torch/plotting.py#L165"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `draw_ellipse`

```python
draw_ellipse(
    mean,
    covariance,
    weight,
    ax=None,
    fill=False,
    marker='X',
    contour_levels=[2],
    weight_size=1000,
    label=None,
    component_count=0,
    **kwargs,
)
```

Draw the error ellipse corresponding to a given mean vector and Gaussian covariance matrix. Since this is done in the context of Gaussian mixture models (GMMs), the weight is also indicated with a marker whose size is proportional to the weight value. Based on https://www.visiondummy.com/2014/04/draw-error-ellipse-representing-covariance-matrix/.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mean` | `array-like of shape (2)` | - | The mean of the Gaussian distribution. |
| `covariance` | `array-like of shape (2, 2)` | - | The covariance matrix of the Gaussian distribution. |
| `weight` | `float` | - | The weight associated with this GMM component. |
| `ax` | `instance of the axes.Axes class from pyplot` | `None` | The panel where to draw the ellipse. |
| `fill` | `bool` | `False` | Whether to fill the contours or not. |
| `marker` | `str` | `'X'` | Marker type for the centroid of the ellipse. |
| `contour_levels` | `list of integers` | `[2]` | The sigma levels whose contour are going to be drawn. The integers can be only 1, 2 and 3. |
| `weight_size` | `float` | `1000` | The multiplier for the marker size (also proportional to the weight). |
| `label` | `str` | `None` | Label associated to the ellipse. |
| `component_count` | `int` | `0` | A counter so that the label is printed only for the first component. |
| `**kwargs` | - | - | The extra keyword arguments to pass to the plotting function. |

---

<a href="../gmm_mi_torch/plotting.py#L254"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `plot_contours`

```python
plot_contours(
    input_samples,
    fitted_samples,
    parameters=None,
    extents=None,
    mass_levels=(0.68, 0.95),
    gridsize=80,
    shade_alpha=0.4,
    linewidths=2,
    colors=('C0', 'salmon'),
    labels=('Input data', 'Fitted model'),
    figsize=None,
)
```

Corner-style comparison of the input data with samples from the fitted model.

Every pair of variables gets a panel with KDE contours enclosing the
requested probability-mass levels for both sample sets; the diagonal
shows the 1-D KDE marginals. Reimplemented with matplotlib/scipy for
gmm-mi-torch (upstream GMM-MI drew these contours with chainconsumer).

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_samples` | `array-like of shape (n_samples, n_features)` | - | The input samples that were used to fit the GMM. |
| `fitted_samples` | `array-like of shape (n_samples, n_features)` | - | The samples obtained from the fitted GMM model. |
| `parameters` | `list of str` | `None` | Variable names for the axis labels; defaults to X1, X2, ... |
| `extents` | `dictionary` | `None` | Ranges of each variable, keyed by its name in `parameters`. |
| `mass_levels` | `tuple of floats` | `(0.68, 0.95)` | Probability-mass levels enclosed by the contours. |
| `gridsize` | `int` | `80` | KDE evaluation grid points per axis. |
| `shade_alpha` | `float` | `0.4` | Alpha for the filled contour regions; 0 disables filling. |
| `linewidths` | `float` | `2` | Width of the contour lines. |
| `colors` | `tuple of two colors` | `('C0', 'salmon')` | Colors for the input data and the fitted model. |
| `labels` | `tuple of two str` | `('Input data', 'Fitted model')` | Legend labels for the two sample sets. |
| `figsize` | `tuple` | `None` | Figure size; defaults to 3.5 inches per variable. |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| - | `fig: instance of the figure.Figure class from pyplot` | The output figure. |

---

<a href="../gmm_mi_torch/plotting.py#L348"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `plot_gmm_contours`

```python
plot_gmm_contours(
    gmm,
    ax=None,
    fill=False,
    label='',
    xlabel='X1',
    ylabel='X2',
    **kwargs,
)
```

Draw the contour ellipses corresponding to a given Gaussian mixture model (GMM).

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gmm` | `object with sklearn-style GMM attributes` | - | The GMM whose contours are going to be displayed; needs `means_`, `covariances_` and `weights_` (e.g. the `FittedGMM` returned by `estimate(..., return_fit=True)`). |
| `ax` | `instance of the axes.Axes class from pyplot` | `None` | The panel where to plot the samples. |
| `fill` | `bool` | `False` | Whether to fill the contours or not. |
| `label` | `str` | `''` | The legend label to associate to the contours. |
| `xlabel` | `str` | `'X1'` | Label of the x-axis. |
| `ylabel` | `str` | `'X2'` | Label of the y-axis. |
| `**kwargs` | - | - | The extra keyword arguments to pass to the plotting functions. |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `ax` | `instance of the axes.Axes class from pyplot` | The output panel. |

---

<a href="../gmm_mi_torch/plotting.py#L380"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `calculate_summary_and_significance`

```python
calculate_summary_and_significance(estimates)
```

Given an array of estimates, calculate their sample mean, the error on the mean, and the significance with respect to 0.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `estimates` | `1D array-like` | The value of all estimates for a certain number of trials. |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `mean_value` | `float` | The mean of the estimates. |
| `mean_error` | `float` | The error on the mean of the estimates. |
| `significance` | `float` | The distance of the mean with respect to 0. |

---

<a href="../gmm_mi_torch/plotting.py#L406"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `histogram_estimates`

```python
histogram_estimates(
    estimates,
    ax=None,
    bins=20,
    alpha=1,
    title='',
    legendsize=25,
    color='salmon',
    lw=3,
    histtype='step',
)
```

Plot the histogram of an array of estimates, centred on 0.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `estimates` | `1D array-like` | - | The estimates whose histogram we plot. |
| `ax` | `instance of the axes.Axes class from pyplot` | `None` | The panel where to draw the histogram. |
| `bins` | `int` | `20` | Number of histogram bins. |
| `alpha` | `float` | `1` | Transparency of the histogram line. Must be between 0 and 1. |
| `title` | `str` | `''` | Title of the panel. |
| `legendsize` | `int` | `25` | Font size of the legend. |
| `color` | `str` | `'salmon'` | Histogram color. |
| `lw` | `int` | `3` | Line width of the histogram line. |
| `histtype` | `{'bar', 'barstacked', 'step', 'stepfilled'}` | `'step'` | The type of histogram to draw. |

---

<a href="../gmm_mi_torch/plotting.py#L444"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `swap_leg_handles`

```python
swap_leg_handles(ax, i1=-1, i2=-2)
```

Swap legend handles and labels.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ax` | `instance of the axes.Axes class from pyplot` | - | The panel whose legend is going to be affected |
| `i1` | `int` | `-1` | The first index to swap. |
| `i2` | `int` | `-2` | The first index to swap. |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `handles` | sequence of `.Artist` (see pyplot docs) | The swapped handles. |
| `labels` | `list of str` | The swapped labels. |

---

<a href="../gmm_mi_torch/plotting.py#L468"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `plot_multivariate_normal`

```python
plot_multivariate_normal(
    ax=None,
    mean=0,
    cov=1,
    alpha=1,
    color='black',
    lw=5,
    ls='--',
)
```

Plot Gaussian distribution in 1D.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ax` | `instance of the axes.Axes class from pyplot` | `None` | The panel where to draw the Gaussian. |
| `mean` | `float` | `0` | The mean of the Gaussian distribution. |
| `cov` | `float` | `1` | The covariance of the Gaussian distribution. |
| `alpha` | `float` | `1` | Transparency of the line. Must be between 0 and 1. |
| `color` | `str` | `'black'` | Line color. |
| `lw` | `int` | `5` | Line width. |
| `ls` | `str` | `'--'` | Line style. Check pyplot for possible values. |

---

<a href="../gmm_mi_torch/plotting.py#L496"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `plot_bias_chi2_histogram`

```python
plot_bias_chi2_histogram(
    estimates,
    analytic,
    ax=None,
    bins=20,
    alpha=0.5,
    show_title=True,
    show_legend=True,
    legendsize=44,
    color='grey',
    lw=3,
    histtype='bar',
    savefig=False,
    savepath='./',
    labelfontsize=60,
    labelcolor='black',
    hide_ylabel=False,
    ds='',
)
```

Plot the histogram of the bias of estimates, centred on 0. Also calculate reduced chi2.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `estimates` | `array-like of shape (n_estimates, 2)` | - | The estimates whose bias we histogram. First column is the MI estimate, second column is the MI error. |
| `analytic` | `float` | - | The analytic MI value. |
| `ax` | `instance of the axes.Axes class from pyplot` | `None` | The panel where to draw the histogram. |
| `bins` | `int` | `20` | Number of histogram bins. |
| `alpha` | `float` | `0.5` | Transparency of the histogram line. Must be between 0 and 1. |
| `show_title` | `bool` | `True` | Whether to display the title with bias and chi2 results. |
| `show_legend` | `bool` | `True` | Whether to display the legend. |
| `legendsize` | `int` | `44` | Font size of the legend. |
| `color` | `str` | `'grey'` | Histogram color. |
| `lw` | `int` | `3` | Line width of the histogram line. |
| `histtype` | `{'bar', 'barstacked', 'step', 'stepfilled'}` | `'bar'` | The type of histogram to draw. |
| `savefig` | `bool` | `False` | Whether to save or not the figure as a pdf file (for the paper). |
| `savepath` | `str` | `'./'` | The path where to save the figure. Only used if savefig is True. |
| `labelfontsize` | `int` | `60` | Fontsize of labels and title. |
| `labelcolor` | `str` | `'black'` | Color of labels and title. |
| `hide_ylabel` | `bool` | `False` | Whether to paint the ylabel white, so that it's hidden. Only for the paper. |
| `ds` | `str` | `''` | The dataset which is being analysed. |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `biases` | `np.array of shape (n_estimates)` | The biases of the estimates. |
| `red_chi2` | `float` | The reduced chi2 of the estimates. |
| `df` | `int` | The number of degrees of freedom. |

---

<a href="../gmm_mi_torch/plotting.py#L579"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `create_heatmap`

```python
create_heatmap(
    data,
    row_labels,
    col_labels,
    fsize=30,
    ax=None,
    cbar_kw={},
    cbarlabel="",
    **kwargs,
)
```

Create a heatmap from a numpy array and two lists of labels.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `array-like of shape (n_rows, n_columns)` | - | A 2D numpy array containing the data values that are reported. |
| `row_labels` | `list` | - | A list with the labels for the rows. |
| `col_labels` | `list` | - | A list with the labels for the columns. |
| `fsize` | `int` | `30` | Font size of the row and column labels. |
| `ax` | `matplotlib.axes.Axes` instance | `None` | The panel in which the heatmap is plotted. If not provided, use current axes or create a new one. |
| `cbar_kw` | `dict` | `{}` | Contains the arguments to `matplotlib.Figure.colorbar`. |
| `cbarlabel` | `str` | `""` | The label for the colorbar. |
| `**kwargs` | - | - | All other arguments are forwarded to `imshow`. |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `im` | `AxesImage object` | The output of `imshow`. |
| `cbar` | `colorbar` object | The colorbar. |

---

<a href="../gmm_mi_torch/plotting.py#L645"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `annotate_heatmap`

```python
annotate_heatmap(
    im,
    data=None,
    valfmt="{x:.2f}",
    save_fig=False,
    save_path="",
    threshold=0.01,
    **textkw,
)
```

Used to annotate a previously-created heatmap.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `im` | `AxesImage` | - | The image to be annotated. |
| `data` | `array-like of shape (n_rows, n_columns)` | `None` | Data used to annotate.  If None, the image's data is used. |
| `valfmt` | either str, or `matplotlib.ticker.Formatter` | `"{x:.2f}"` | The format of the annotations inside the heatmap.  This should either use the string format method, e.g. "$ {x:.2f}", or be a `matplotlib.ticker.Formatter`. |
| `save_fig` | `bool` | `False` | Whether to save the annotated heatmap or not. |
| `save_path` | `str` | `""` | If save_fig is True, the path where to save the figure. |
| `threshold` | `float` | `0.01` | All values of MI below this are displayed as 0. |
| `**textkw` | - | - | All other arguments are forwarded to each call to `text` used to create the text labels. |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `texts` | `list` | Contains all the text that is annotated onto the heatmap. |

---

_This file was generated by `docs/make_api_docs.py` from the docstrings; edit the docstrings, not this file._
