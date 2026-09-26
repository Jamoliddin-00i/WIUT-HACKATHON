"""Validate and score local proposals using the unchanged organizer evaluator."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate import evaluate, print_report, validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pred', type=Path, default=ROOT / 'auto_proposals.json')
    parser.add_argument('--labels', type=Path,
                        default=ROOT / 'labels/user_truth_2026-09-26.json')
    parser.add_argument('--out', type=Path, help='optional metric JSON output')
    args = parser.parse_args()
    truth = json.loads(args.labels.read_text(encoding='utf-8'))
    raw = json.loads(args.pred.read_text(encoding='utf-8'))
    prediction = raw if 'videos' in raw else {'team': 'Salen local proposals', 'videos': raw}
    for title, data in [('labels', {'team': 'manual', 'videos': truth}),
                        ('proposals', prediction)]:
        errors, _ = validate(data, truth)
        if errors:
            parser.error(title + ': ' + '; '.join(errors))
    report = evaluate(truth, prediction, per_video=True)
    print('Development comparison against manual labels; not held-out accuracy.')
    print_report(report)
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
