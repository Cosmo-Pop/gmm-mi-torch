"""Generate the API reference pages in docs/ from the package's docstrings.

One Markdown page per module with public names (estimator, em, selection,
gaussian, plotting), in the layout of the pop-cosmos documentation: for each
public function and class, in source order, a link to its source line, its
signature, its summary, and its Parameters, Returns, Raises, Warns and Notes
sections. Parameters and Returns are tables built from the numpydoc
sections, with a Default column taken from the signature.

Usage, from the repository root (needs numpydoc: pip install -e '.[docs]'):

    python docs/make_api_docs.py           # write the pages
    python docs/make_api_docs.py --check   # exit 1 if a page is out of date

The pages are committed; regenerate them whenever a public docstring
changes. The output depends only on the code, so --check is exact.
"""
import argparse
import importlib
import inspect
import os
import re
import sys

from numpydoc.docscrape import NumpyDocString

DOCS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(DOCS)
sys.path.insert(0, ROOT)
MODULES = ['estimator', 'em', 'selection', 'gaussian', 'plotting']
BADGE = ('<a href="../gmm_mi_torch/{module}.py{anchor}"><img align="right" '
         'style="float:right;" '
         'src="https://img.shields.io/badge/-source-cccccc?style=flat-square">'
         '</a>')


def def_line(obj):
    """Line of the `def` or `class` statement (after any decorators)."""
    lines, start = inspect.getsourcelines(inspect.unwrap(obj))
    for i, line in enumerate(lines):
        if line.lstrip().startswith(('def ', 'class ', 'async def ')):
            return start + i
    return start


def public_members(mod):
    """Public functions and classes defined in `mod`, in source order."""
    out = []
    for name, obj in vars(mod).items():
        if name.startswith('_') or getattr(obj, '__module__', None) != mod.__name__:
            continue
        if inspect.isfunction(obj) or inspect.isclass(obj):
            out.append((def_line(obj), name, obj))
    return [(line, name, obj) for line, name, obj in sorted(out, key=lambda t: t[0])]


def signature(name, obj):
    """The call signature, one parameter per line when it is long."""
    params = [str(p) for p in inspect.signature(obj).parameters.values()]
    if inspect.isclass(obj) and params and params[0] == 'self':
        params = params[1:]
    one = f'{name}({", ".join(params)})'
    if len(one) <= 79:
        return one
    return f'{name}(\n' + ''.join(f'    {p},\n' for p in params) + ')'


def cell(text):
    """Text for one table cell: one line, pipes escaped."""
    return text.replace('|', '\\|')


def description(lines):
    """A numpydoc description (list of lines) as one table cell."""
    paras, cur = [], []
    for line in lines:
        if line.strip():
            cur.append(line.strip())
        elif cur:
            paras.append(' '.join(cur))
            cur = []
    if cur:
        paras.append(' '.join(cur))
    return cell('<br><br>'.join(paras))


def type_cell(text):
    return f'`{text}`' if text and '`' not in text else cell(text or '-')


def names_cell(name):
    return ', '.join(f'`{n.strip()}`' for n in name.split(','))


def parameters_table(params, sig_params):
    """The Parameters section as a table; a Default column when any
    documented parameter has a default in the signature."""
    defaults = {}
    for p in params:
        vals = []
        for n in p.name.split(','):
            sp = sig_params.get(n.strip().lstrip('*'))
            if sp is not None and sp.default is not inspect.Parameter.empty:
                vals.append(f'`{sp.default!r}`')
        defaults[p.name] = ', '.join(vals)
    with_default = any(defaults.values())
    if with_default:
        rows = ['| Parameter | Type | Default | Description |',
                '|-----------|------|---------|-------------|']
    else:
        rows = ['| Parameter | Type | Description |',
                '|-----------|------|-------------|']
    for p in params:
        cols = [names_cell(p.name), type_cell(p.type)]
        if with_default:
            cols.append(defaults[p.name] or '-')
        cols.append(description(p.desc))
        rows.append('| ' + ' | '.join(cols) + ' |')
    return rows


def returns_table(returns):
    rows = ['| Return Value | Type | Description |',
            '|--------------|------|-------------|']
    for r in returns:
        name = names_cell(r.name) if r.name else '-'
        rows.append(f'| {name} | {type_cell(r.type)} | {description(r.desc)} |')
    return rows


def paragraphs(lines):
    """Free text (summary, extended summary, notes) with its paragraph
    breaks; lines are kept as written."""
    return [line.rstrip() for line in lines]


def render(name, obj, module, kind, level, sig_name=None):
    """One public symbol: badge, heading, signature and sections."""
    line = def_line(obj)
    doc = NumpyDocString(inspect.getdoc(obj) or '')
    out = [BADGE.format(module=module, anchor=f'#L{line}'), '',
           f'{"#" * level} <kbd>{kind}</kbd> `{name}`', '',
           '```python', signature(sig_name or name, obj), '```', '']
    if doc['Summary']:
        out += [' '.join(s.strip() for s in doc['Summary']), '']
    if doc['Extended Summary']:
        out += paragraphs(doc['Extended Summary']) + ['']
    sub = '#' * (level + 1)
    try:
        sig_params = dict(inspect.signature(obj).parameters)
    except (TypeError, ValueError):
        sig_params = {}
    if doc['Parameters']:
        out += [f'{sub} Parameters', ''] + parameters_table(doc['Parameters'], sig_params) + ['']
    if doc['Attributes']:
        out += [f'{sub} Attributes', ''] + parameters_table(doc['Attributes'], {}) + ['']
    if doc['Returns']:
        out += [f'{sub} Returns', ''] + returns_table(doc['Returns']) + ['']
    for section in ('Raises', 'Warns'):
        if doc[section]:
            out += [f'{sub} {section}', '']
            for e in doc[section]:
                exc = e.type or e.name
                out.append(f'- `{exc}`: {description(e.desc)}')
            out.append('')
    if doc['Notes']:
        out += [f'{sub} Notes', ''] + paragraphs(doc['Notes']) + ['']
    return out


def page(module):
    """The whole page for gmm_mi_torch.<module>."""
    mod = importlib.import_module(f'gmm_mi_torch.{module}')
    from gmm_mi_torch import __version__
    out = ['<!-- markdownlint-disable -->', '',
           BADGE.format(module=module, anchor=''), '',
           f'# <kbd>module</kbd> `{module}.py`', '']
    if mod.__doc__:
        out += [inspect.cleandoc(mod.__doc__), '']
    for _, name, obj in public_members(mod):
        out += ['---', '']
        if inspect.isclass(obj):
            out += render(name, obj, module, 'class', 2)
            for mname, mobj in vars(obj).items():
                if inspect.isfunction(mobj) and not mname.startswith('_') \
                        and (inspect.getdoc(mobj) or '').strip():
                    out += render(mname, mobj, module, 'method', 3)
        else:
            out += render(name, obj, module, 'function', 2)
    out += ['---', '',
            f'_This file was generated by `docs/make_api_docs.py` from the docstrings '
            f'of gmm-mi-torch {__version__}; edit the docstrings, not this file._', '']
    text = '\n'.join(out)
    while '\n\n\n' in text:
        text = text.replace('\n\n\n', '\n\n')
    return link_docs(text)


def link_docs(text):
    """Make a mention of another page in docs/ ("docs/design_notes.md")
    a link, outside code blocks and inline code."""
    parts = text.split('```')
    for i in range(0, len(parts), 2):          # even parts: outside fences
        parts[i] = re.sub(r'(?<![\[(/`\w])docs/([a-z_]+\.md)(?![`\w])', r'[docs/\1](\1)', parts[i])
    return '```'.join(parts)


def main():
    ap = argparse.ArgumentParser(description='Generate (or check) the API pages in docs/.')
    ap.add_argument('--check', action='store_true',
                    help='exit 1 if any page differs from what the docstrings give')
    args = ap.parse_args()
    stale = []
    for module in MODULES:
        path = os.path.join(DOCS, f'{module}.md')
        text = page(module)
        old = open(path).read() if os.path.exists(path) else None
        if args.check:
            if old != text:
                stale.append(path)
        elif old != text:
            with open(path, 'w') as f:
                f.write(text)
            print('wrote', os.path.relpath(path, ROOT))
    if args.check:
        for path in stale:
            print('out of date:', os.path.relpath(path, ROOT))
        print('all API pages up to date' if not stale else f'{len(stale)} page(s) out of date')
        sys.exit(1 if stale else 0)


if __name__ == '__main__':
    main()
