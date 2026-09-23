"""Synchronize the seven corrected captions from the reviewed manuscript."""
from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[2]
paper = (ROOT/'results/v11_corrections_0923/V11_corrected_0923.md').read_text()
for figure, filename, variable in [
    ('2', 'fig02_doors', 'NOTE'), ('E', 'fig02_doors', 'NOTE_E'),
    ('9', 'fig08_response', 'NOTE'), ('A', 'figA_mnist_chain', 'NOTE'),
    ('B', 'figB_5p1_budget', 'NOTE'), ('C', 'figC_initgeom_rest', 'NOTE'),
    ('J', 'figJ_layer_chimera', 'NOTE'),
]:
    block = paper.split(f'<!-- figure:{figure} -->')[1].split('<!-- /figure -->')[0]
    caption = '\n'.join(line[2:] for line in block.splitlines() if line.startswith('> '))+'\n'
    path = ROOT/f'analysis/v11_figures_0921/{filename}.py'
    source = path.read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == variable for t in n.targets))
    lines = source.splitlines(keepends=True)
    literal = caption.replace('\\', '\\\\').replace('"""', '\\"\\"\\"')
    lines[node.lineno-1:node.end_lineno] = [f'{variable} = """{literal}"""\n']
    path.write_text(''.join(lines))
