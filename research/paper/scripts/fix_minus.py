"""Replace hyphen-minus before digits in table rows with a typographic minus ($-$).

Only rows containing '&' are touched, and a hyphen preceded by a word character, '$', '-' or '['
(model names such as Qwen3.8-27B, ranges such as 2--3, already-converted values) is left alone."""
import re
import sys
from pathlib import Path

PAT = re.compile(r'(?<![\w$\-\[\\{])-(?=\d)')
ZERO = re.compile(r'\$-\$(0\.0+)(?![0-9])')  # no negative zero


def fix(path):
    s = path.read_text(encoding='utf-8')
    out = [ZERO.sub(r'\1', PAT.sub('$-$', ln)) if '&' in ln else ln for ln in s.split('\n')]
    t = '\n'.join(out)
    if t != s:
        path.write_text(t, encoding='utf-8')
        return True
    return False


# Appendix tables that may go on float pages (keeps the layout of the preprint formatting pass).
PLACEMENT = {'contrasts': 'tp', 'contrasts_rq1': 'tp', 'main_full': 't', 'objectives_full': 't', 'per_task': 't',
             'probe': 't', 'robustness': 'tbp'}


def place(tab):
    for name, spec in PLACEMENT.items():
        f = tab / f'{name}.tex'
        if f.exists():
            s = f.read_text(encoding='utf-8')
            t = s
            for old in ('[tp]', '[tbp]', '[t]'):
                if s.startswith('\\begin{table*}' + old):
                    t = s.replace('\\begin{table*}' + old, '\\begin{table*}[' + spec + ']', 1)
                    break
            if t != s:
                f.write_text(t, encoding='utf-8')


def main():
    tab = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / 'tables'
    changed = [p.name for p in sorted(tab.glob('*.tex')) if p.name != 'generated_numbers.tex' and fix(p)]
    place(tab)
    print('typographic minus:', ', '.join(changed) or 'nothing to change')


if __name__ == '__main__':
    main()
