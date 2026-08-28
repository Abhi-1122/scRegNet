"""Walks out/differential/*/*/*/metrics.json and prints a summary table of macro-F1 (and
per-class where available) across all Phase 2 configs/pairs."""
import os
import json
import glob

ROOT = "out/differential"


def main():
    rows = []
    for mpath in sorted(glob.glob(os.path.join(ROOT, "*", "*", "*", "metrics.json"))):
        parts = mpath.split(os.sep)
        cell_type, pair, ablation = parts[-4], parts[-3], parts[-2]
        with open(mpath) as f:
            d = json.load(f)
        row = {
            'cell_type': cell_type, 'pair': pair, 'config': ablation,
            'macro_f1': d.get('macro_f1'),
        }
        pc = d.get('per_class', {})
        for cls in ['conserved', 'A-specific', 'B-specific', 'neither']:
            entry = pc.get(cls, {})
            row[f'{cls}_n_test'] = entry.get('n_test')
            row[f'{cls}_auroc'] = entry.get('auroc')
        rows.append(row)

    header = ['cell_type', 'pair', 'config', 'macro_f1',
              'conserved_n_test', 'conserved_auroc',
              'A-specific_n_test', 'A-specific_auroc',
              'B-specific_n_test', 'B-specific_auroc',
              'neither_n_test', 'neither_auroc']
    print("\t".join(header))
    for row in rows:
        print("\t".join(str(row.get(h, '')) for h in header))
    print(f"\n{len(rows)} runs found under {ROOT}/")


if __name__ == "__main__":
    main()
