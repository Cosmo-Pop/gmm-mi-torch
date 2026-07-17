"""Generate docs/validation_summary.pdf: a distilled summary of the
gmm-mi-torch validation and benchmark campaign (full logs: validation/).

Run from the repository root:  python docs/make_validation_summary.py
Requires reportlab.
"""
import os

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, KeepTogether)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   'validation_summary.pdf')

styles = getSampleStyleSheet()
H1 = ParagraphStyle('H1x', parent=styles['Heading1'], fontSize=15,
                    spaceBefore=14, spaceAfter=6)
H2 = ParagraphStyle('H2x', parent=styles['Heading2'], fontSize=12,
                    spaceBefore=10, spaceAfter=4)
BODY = ParagraphStyle('Bodyx', parent=styles['Normal'], fontSize=9.5,
                      leading=13, spaceAfter=5)
BULL = ParagraphStyle('Bullx', parent=BODY, leftIndent=14, bulletIndent=4)
CAP = ParagraphStyle('Capx', parent=BODY, fontSize=8.5, leading=11,
                     textColor=colors.HexColor('#444444'), spaceBefore=2,
                     spaceAfter=8)
CELL = ParagraphStyle('Cellx', parent=BODY, fontSize=8.5, leading=10.5,
                      spaceAfter=0)
TITLE = ParagraphStyle('Titlex', parent=styles['Title'], fontSize=19,
                       spaceAfter=2)
SUB = ParagraphStyle('Subx', parent=BODY, fontSize=10.5,
                     textColor=colors.HexColor('#333333'), spaceAfter=12)


def tbl(header, rows, widths=None, bold_row=None):
    data = [[Paragraph(f'<b>{h}</b>', CELL) for h in header]]
    for r in rows:
        data.append([Paragraph(str(c), CELL) for c in r])
    t = Table(data, colWidths=widths, hAlign='LEFT')
    style = [
        ('LINEBELOW', (0, 0), (-1, 0), 0.8, colors.black),
        ('LINEBELOW', (0, -1), (-1, -1), 0.4, colors.grey),
        ('LINEABOVE', (0, 0), (-1, 0), 0.8, colors.black),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]
    if bold_row is not None:
        style.append(('BACKGROUND', (0, bold_row + 1), (-1, bold_row + 1),
                      colors.HexColor('#f0ede8')))
    t.setStyle(TableStyle(style))
    return t


story = []
A = story.append

A(Paragraph('gmm-mi-torch: validation and benchmark summary', TITLE))
A(Paragraph('Distilled from the GB10 acceptance campaign, July 2026. '
            'Full working logs, task-scoped diffs and bench scripts: '
            '<font face="Courier">validation/</font> in this repository.',
            SUB))

A(Paragraph('1. What was validated', H1))
A(Paragraph(
    'gmm-mi-torch is a pure-PyTorch reimplementation of GMM-MI '
    '(Piras et al. 2023): mutual information and related quantities (MI, '
    'conditional MI, total correlation, marginal KL, continuous-categorical '
    'MI) with bootstrap errorbars, from Gaussian mixture fits. The port '
    'replaces GMM-MI\'s sklearn/loop machinery with a single weighted EM '
    'batched over a replicate dimension: cross-validation folds are 0/1 '
    'sample-weight masks, bootstrap realisations are multinomial-count '
    'weights, and initialisations are additional replicates, so the whole '
    'procedure (initialisations x folds, or all bootstrap realisations, or '
    'all candidate component counts) runs as one tensor computation on CPU '
    'or GPU, at float64 or float32. Marginalisation is analytic index '
    'masking of the joint fit, exactly as in GMM-MI.', BODY))
A(Paragraph(
    'The campaign validated: (i) numerical correctness against analytic, '
    'quadrature and Monte-Carlo ground truths; (ii) float32 viability on '
    'GPU; (iii) statistical calibration of the bootstrap errorbars; '
    '(iv) performance against the sklearn-based reference; and (v) scaling '
    'to N = 10<super>8</super> samples. Hardware: NVIDIA GB10 (DGX Spark), '
    '128 GB unified memory, CUDA 13.0, torch 2.11.0+cu128.', BODY))

A(Paragraph('2. Correctness: test suite', H1))
A(Paragraph(
    'The suite holds 25 tests (24 in CI plus one offline-gated statistical '
    'test) in GMM-MI\'s own test style: ground-truth targets asserted within '
    '3-sigma bootstrap bands. A conftest shim routes the unchanged suite to '
    'any device/dtype via environment variables. Every release gate in the '
    'campaign required the full suite green on both production configs:', BODY))
A(tbl(['Config', 'Result', 'Wall time'],
      [['CPU float64 (parity baseline)', '24 passed, 1 skipped', '51 s'],
       ['CUDA float32, em_chunk=1000 (full batch)', '24 passed, 1 skipped',
        '7 s']], widths=[210, 130, 80]))
A(Paragraph(
    'Selected targets: analytic Gaussian MI under both selection protocols; '
    'pinned conditional-MI values for a dependent conditioner '
    '(I(X;Y|Z) = 0.0671910473 nat); total correlation of an asymmetric '
    '3-component trivariate mixture against a Monte-Carlo ground truth of '
    '1.092080 nat (standard error 1.4e-4); forward/reverse KL between '
    'genuinely trimodal marginals against 1-D quadrature (0.1725434 / '
    '0.1309184 nat); continuous-categorical MI (two and three categories, '
    'empirical and uniform category weights) against quadrature; '
    'chunked-vs-unchunked EM parity at float64 (agreement at machine-epsilon '
    'level, |delta| < 1e-9); bootstrap-fallback distributional consistency; '
    'and the documented error paths. All estimator tests land well inside '
    'their 3-sigma bands on both configs.', BODY))
A(Paragraph(
    'Two design points worth recording: the original 12 tests reproduce '
    'their historical values bit-for-bit at CPU float64 across all module '
    'evolution; and the multimodal tests deliberately use the default '
    'initialisation with BIC selection rather than k-means initialisation '
    '(a first draft that only passed under k-means was rejected as brittle: '
    'a test that requires a non-default initialiser does not exercise the '
    'recommended path).', BODY))

A(Paragraph('3. Float32 on GPU', H1))
A(Paragraph(
    'The full suite passes at float32 with the default regularisation '
    '(reg_covar = 1e-6); the Cholesky-conditioning fallback was never '
    'needed. On the benchmark workload the float32 and float64 MI estimates '
    'agree to about 0.1 bootstrap sigma. One numerical design choice '
    'matters at float32: the streaming (n_chunk) M-step accumulates '
    'second moments shifted by the incoming means rather than raw moments, '
    'avoiding catastrophic cancellation when |mu|<super>2</super> is much '
    'larger than the variance.', BODY))

A(Paragraph('4. Errorbar calibration', H1))
A(Paragraph(
    'Two independent checks confirm the bootstrap errorbars mean what they '
    'claim. The offline suite test measures coverage over 40 independent '
    'datasets (rho = 0.6 Gaussian):', BODY))
A(tbl(['Config', '+-1 sigma coverage (nominal 0.68)',
       '+-2 sigma coverage (nominal 0.95)'],
      [['CPU float64', '0.675', '0.925'],
       ['CUDA float32', '0.575', '0.925']], widths=[120, 160, 160]))
A(Paragraph(
    'Independently, the walkthrough notebook estimates MI on 501 fresh '
    'realisations (matching the sample size of the GMM-MI significance '
    'notebooks, computed live): the scatter of the estimates is 0.0091 nat '
    'against an average claimed bootstrap sigma of 0.0087 nat, and the '
    'standardized residuals against N(0,1) give 0.07 +- 1.08 with reduced '
    'chi-squared 1.16 (500 dof) - i.e. the errorbars are honest to within '
    'a few per cent (marginally tight), consistent with the behaviour of '
    'the reference implementation.', BODY))

A(Paragraph('5. Performance', H1))
A(Paragraph(
    'Benchmark workload: 3-component bivariate mixture, N = 20 000, '
    'selection over k = 1..8 (3 initialisations x 2 folds), 50 bootstrap '
    'realisations. Reference: sklearn-based GMM-MI on the same CPU.', BODY))
A(tbl(['Config (em_chunk/mc_chunk)', 'Model selection', 'Bootstrap phase'],
      [['GMM-MI reference (sklearn, CPU)', '15.4 s', '10.1 s'],
       ['gmm-mi-torch, CPU fp64 (8/8)', '11.5 s', '1.9 s'],
       ['gmm-mi-torch, CPU fp32 (8/8)', '4.6 s', '0.95 s'],
       ['gmm-mi-torch, CUDA fp32 (8/8)', '1.9 s', '0.24 s'],
       ['gmm-mi-torch, CUDA fp32 (48/48)', '0.90 s', '0.17 s']],
      widths=[210, 105, 105], bold_row=4))
A(Paragraph(
    'Best GPU config: 17x faster selection and 59x faster bootstrap than '
    'the reference. The torch port already beats sklearn on CPU at float64, '
    'because the batched-replicate EM does in one pass what sklearn does '
    'across loops. Setting em_chunk to the full replicate count (here 48) '
    'runs EM as a single batch and roughly halves GPU wall time. Hardware '
    'caveat: on GPUs with rate-limited float64 throughput (such as the '
    'GB10), CUDA float64 is slower than CPU float64 - use float32 on such '
    'devices (validated above).', BODY))

A(Paragraph('6. Large-N scaling (n_chunk)', H1))
A(Paragraph(
    'Sample-dimension chunking streams each EM iteration over sample blocks '
    '(accumulating sufficient statistics), making the EM and validation-'
    'scoring passes O(n_chunk) in memory. Chunked and unchunked paths agree '
    'at the ulp level at float64 (bitwise identity is impossible in '
    'principle: reductions are reordered). Scale demonstration - total '
    'correlation of an 8-dimensional correlated Gaussian, full pipeline '
    '(selection over k = 1..8, 50-replicate bootstrap, MC integration), '
    'CUDA float32, n_chunk = 2.5e5, em_chunk = 48, init_type = minmax:',
    BODY))
A(tbl(['N', 'Total wall', 'Peak GPU memory', 'TC vs sample-cov target'],
      [['10<super>7</super>', '123 s', '17.3 GiB', '-0.13 sigma'],
       ['4 x 10<super>7</super>', '498 s', '31.0 GiB', '-0.04 sigma'],
       ['6 x 10<super>7</super>', '756 s', '46.5 GiB', '-0.08 sigma'],
       ['10<super>8</super>', '23 min', '77.5 GiB', '+0.03 sigma']],
      widths=[80, 80, 110, 150], bold_row=3))
A(Paragraph(
    'Wall time and peak memory scale linearly in N; without n_chunk the '
    'unchunked E-step at N = 10<super>8</super> would need of order 10 TiB. '
    'Above torch.multinomial\'s 2<super>24</super>-category cap the uniform '
    'bootstrap falls back to an identically-distributed randint histogram '
    '(the fallback triggers only where the reference implementation would '
    'have raised). Remaining N-ceilings, documented but unhit at 128 GB: '
    'the CV-mask replicates, the bootstrap-weight transient, and the '
    'default initialisation\'s (R, N, K) responsibility draw (which binds '
    'near N = 2-3 x 10<super>7</super>; minmax initialisation avoids it).',
    BODY))

A(Paragraph('7. Feature parity with GMM-MI', H1))
A(Paragraph(
    'Ported with semantics taken from the GMM-MI source (not re-derived): '
    'initialisation types (random responsibilities, minmax, k-means, '
    'randomized k-means - including GMM-MI\'s covariance normalisation '
    'conventions); rank-probit marginal gaussianisation (preprocess = '
    '\'gaussianise\', rejected with an error for the non-invariant KL '
    'quantities); fixed_components_number; AIC/BIC component selection '
    '(computed on a full-data refit, with cross-validated parameters '
    'returned, exactly as GMM-MI does); marginal KL in both directions; '
    'and continuous-categorical MI with empirical or uniform category '
    'weights (Piras et al. 2023, App. B). Parameter names follow GMM-MI '
    'where a counterpart exists (reg_covar, threshold_fit, '
    'threshold_components, max_components). GMM-MI\'s sequential selection '
    'protocol (increment k, patience early-stop) is available alongside '
    'the batched protocol.', BODY))

A(Paragraph('8. Known limitations', H1))
for item in [
    'The categorical bootstrap is stratified per category and holds the '
    'empirical category weights fixed, so the reported sigma excludes '
    'category-weight sampling uncertainty (an underestimate only for small '
    'N or extreme imbalance).',
    'n_chunk makes the EM and scoring passes N-independent in memory, but '
    'the CV masks and bootstrap-weight construction still scale with N '
    '(see Section 6).',
    'The batched EM records no per-iteration loss curves, so GMM-MI\'s '
    'loss-curve diagnostics have no counterpart; the walkthrough documents '
    'this and routes users to the fitted-model and calibration plots.',
    'On GPUs whose torch build JIT-compiles jiterator kernels via nvrtc '
    '(observed on GB10 with torch 2.11), Tensor.prod, torch.special.ndtri '
    'and torch.erfinv can fail to compile; the module routes around all '
    'three (log-sum-exp product; probit on CPU).',
]:
    A(Paragraph(item, BULL, bulletText='-'))

A(Spacer(1, 6))
A(Paragraph(
    'Provenance: distilled from validation/gb10_report.md (Parts 1-6 of the '
    'GB10 acceptance campaign) and the executed walkthrough notebook; '
    'regenerate with docs/make_validation_summary.py.', CAP))

SimpleDocTemplate(OUT, pagesize=A4,
                  leftMargin=18 * mm, rightMargin=18 * mm,
                  topMargin=16 * mm, bottomMargin=16 * mm,
                  title='gmm-mi-torch: validation and benchmark summary',
                  author='Cosmo-pop / gmm-mi-torch').build(story)
print(f'wrote {OUT}')
