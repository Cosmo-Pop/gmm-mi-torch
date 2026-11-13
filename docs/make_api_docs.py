"""Generate the API reference pages in docs/api/ from the package's docstrings.

One Markdown page per module with public names (estimator, em, selection,
gaussian, plotting), in the layout of the pop-cosmos documentation: a list
of the page's contents, then for each public function and class, in source
order, a link to its source line, its signature, its summary, and its
Parameters, Returns, Raises, Warns and Notes sections. Parameters and
Returns are tables built from the numpydoc sections; the Default column
shows each default as it is written in the code.

Usage, from the repository root (needs numpydoc and the plotting
dependencies, which the `docs` extra installs: pip install -e '.[docs]'):

    python docs/make_api_docs.py           # write the pages
    python docs/make_api_docs.py --check   # exit 1 if a page is out of date

The pages are committed. They record source line numbers, so regenerate
them whenever the package changes, and run --check before a release. The
output depends only on the code, so --check is exact.
"""
import argparse
import ast
import importlib
import inspect
import os
import re
import sys

from numpydoc.docscrape import NumpyDocString

DOCS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(DOCS)
API = os.path.join(DOCS, 'api')                 # the pages; the notes stay in docs/
sys.path.insert(0, ROOT)
MODULES = ['estimator', 'em', 'selection', 'gaussian', 'plotting']
BADGE = ('<a href="../../gmm_mi_torch/{module}.py{anchor}"><img align="right" '
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
    return [(name, obj) for _, name, obj in sorted(out, key=lambda t: t[0])]


def source_defaults(obj):
    """Each parameter's default as written in the source (`100_000`, not
    `100000`); empty when the source cannot be read."""
    try:
        src = inspect.getsource(inspect.unwrap(obj))
    except (OSError, TypeError):
        return {}
    indent = len(src) - len(src.lstrip())
    src = '\n'.join(l[indent:] for l in src.split('\n'))
    try:
        fn = next(n for n in ast.walk(ast.parse(src))
                  if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
    except (SyntaxError, StopIteration):
        return {}
    args = fn.args.posonlyargs + fn.args.args
    out = {a.arg: ast.get_source_segment(src, d)
           for a, d in zip(args[len(args) - len(fn.args.defaults):], fn.args.defaults)}
    out.update({a.arg: ast.get_source_segment(src, d)
                for a, d in zip(fn.args.kwonlyargs, fn.args.kw_defaults) if d is not None})
    return out


def sig_params(obj):
    """The signature's parameters without `self`."""
    params = dict(inspect.signature(obj).parameters)
    params.pop('self', None)
    return params


def signature(name, obj):
    """The call signature, defaults as in the source, one parameter per
    line when it is long."""
    written = source_defaults(obj)
    parts = []
    for pname, p in sig_params(obj).items():
        if p.default is not inspect.Parameter.empty and pname in written:
            parts.append(f'{pname}={written[pname]}')
        else:
            parts.append(str(p))
    one = f'{name}({", ".join(parts)})'
    if len(one) <= 79:
        return one
    return f'{name}(\n' + ''.join(f'    {p},\n' for p in parts) + ')'


def escape(text):
    """Make prose safe in Markdown: pipes (table cells), and `*` and
    word-edge `_`, which GitHub would read as emphasis (2**20 ... 2**62
    renders bold); inline code is left as it is."""
    parts = text.split('`')
    for i in range(0, len(parts), 2):           # even parts: outside code
        p = parts[i].replace('|', '\\|').replace('*', '\\*')
        parts[i] = re.sub(r'(?<![A-Za-z0-9])_|_(?![A-Za-z0-9])', r'\\_', p)
    return '`'.join(parts)


def description(lines):
    """A numpydoc description (list of lines) as one table cell or bullet:
    paragraphs joined with line breaks."""
    paras, cur = [], []
    for line in lines:
        if line.strip():
            cur.append(line.strip())
        elif cur:
            paras.append(' '.join(cur))
            cur = []
    if cur:
        paras.append(' '.join(cur))
    return '<br><br>'.join(escape(p) for p in paras)


def type_cell(text, has_default_column):
    """The Type cell. With a Default column, a trailing `, default=...` is
    dropped: the column shows the value."""
    if has_default_column:
        text = re.sub(r',?\s*default\s*[=:].*$', '', text or '').strip()
    if not text:
        return '-'
    return f'`{text}`' if '`' not in text else escape(text)


def names_cell(name):
    return ', '.join(f'`{n.strip()}`' for n in name.split(','))


def parameters_table(params, obj):
    """The Parameters section as a table; a Default column when any
    documented parameter has a default in the signature."""
    try:
        sig = sig_params(obj)
    except (TypeError, ValueError):
        sig = {}
    written = source_defaults(obj) if sig else {}
    defaults = {}
    for p in params:
        vals = []
        for n in p.name.split(','):
            n = n.strip().lstrip('*')
            sp = sig.get(n)
            if sp is not None and sp.default is not inspect.Parameter.empty:
                vals.append(f'`{written.get(n, repr(sp.default))}`')
        defaults[p.name] = ', '.join(vals)
    with_default = any(defaults.values())
    if with_default:
        rows = ['| Parameter | Type | Default | Description |',
                '|-----------|------|---------|-------------|']
    else:
        rows = ['| Parameter | Type | Description |',
                '|-----------|------|-------------|']
    for p in params:
        cols = [names_cell(p.name), type_cell(p.type, with_default)]
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
        rows.append(f'| {name} | {type_cell(r.type, False)} | {description(r.desc)} |')
    return rows


def anchor(kind, name):
    """GitHub's anchor for a heading `<kbd>kind</kbd> \\`name\\``."""
    return f'{kind}-{name.lower()}'


def render(name, obj, module, kind, level):
    """One public symbol: badge, heading, signature and sections."""
    doc = NumpyDocString(inspect.getdoc(obj) or '')
    out = [BADGE.format(module=module, anchor=f'#L{def_line(obj)}'), '',
           f'{"#" * level} <kbd>{kind}</kbd> `{name}`', '',
           '```python', signature(name, obj), '```', '']
    if doc['Summary']:
        out += [' '.join(s.strip() for s in doc['Summary']), '']
    if doc['Extended Summary']:
        out += [line.rstrip() for line in doc['Extended Summary']] + ['']
    sub = '#' * (level + 1)
    if doc['Parameters']:
        out += [f'{sub} Parameters', ''] + parameters_table(doc['Parameters'], obj) + ['']
    if doc['Attributes']:
        out += [f'{sub} Attributes', ''] + parameters_table(doc['Attributes'], None) + ['']
    if doc['Returns']:
        out += [f'{sub} Returns', ''] + returns_table(doc['Returns']) + ['']
    for section in ('Raises', 'Warns'):
        if doc[section]:
            out += [f'{sub} {section}', '']
            for e in doc[section]:
                out.append(f'- `{e.type or e.name}`: {description(e.desc)}')
            out.append('')
    if doc['Notes']:
        out += [f'{sub} Notes', ''] + [line.rstrip() for line in doc['Notes']] + ['']
    return out


def page(module):
    """The whole page for gmm_mi_torch.<module>."""
    mod = importlib.import_module(f'gmm_mi_torch.{module}')
    members = public_members(mod)
    out = ['<!-- markdownlint-disable -->', '',
           BADGE.format(module=module, anchor=''), '',
           f'# <kbd>module</kbd> `{module}.py`', '']
    if mod.__doc__:
        out += [inspect.cleandoc(mod.__doc__), '']
    # contents: functions first, then classes, each in source order
    links = [f'[`{n}`](#{anchor("function", n)})' for n, o in members if not inspect.isclass(o)]
    links += [f'[`{n}`](#{anchor("class", n)})' for n, o in members if inspect.isclass(o)]
    out += ['**On this page:** ' + ' · '.join(links), '']
    for name, obj in members:
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
            '_This file was generated by `docs/make_api_docs.py` from the docstrings; '
            'edit the docstrings, not this file._', '']
    text = '\n'.join(out)
    while '\n\n\n' in text:
        text = text.replace('\n\n\n', '\n\n')
    return link_docs(text)


def link_docs(text):
    """Make a mention of another page in docs/ ("docs/design_notes.md")
    a link, outside code blocks and inline code: an API page is in the same
    folder, a notes page one level up."""
    def link(m):
        page = m.group(1)
        target = page if page[:-3] in MODULES else '../' + page
        return f'[docs/{page}]({target})'
    parts = text.split('```')
    for i in range(0, len(parts), 2):          # even parts: outside fences
        parts[i] = re.sub(r'(?<![\[(/`\w])docs/([a-z_]+\.md)(?![`\w])', link, parts[i])
    return '```'.join(parts)


def main():
    ap = argparse.ArgumentParser(description='Generate (or check) the API pages in docs/api/.')
    ap.add_argument('--check', action='store_true',
                    help='exit 1 if any page differs from what the docstrings give')
    args = ap.parse_args()
    stale = []
    os.makedirs(API, exist_ok=True)
    for module in MODULES:
        path = os.path.join(API, f'{module}.md')
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
