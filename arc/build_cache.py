#!/usr/bin/env python
"""Build the prefix-tree caches (validation + test) of trained SeMA backbones.

    python arc/build_cache.py [--folds 0 1 2]

IEMOCAP has 6 modalities, i.e. 1,957 prefixes per sample; a fold takes a few minutes on a GPU.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from arc.prefix_cache import build_caches     # noqa: E402
from common.paths import config               # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--folds', type=int, nargs='+', default=None, help='default: configs/sema.json')
    ap.add_argument('--sema_dir', default=os.path.join(ROOT, 'checkpoints', 'sema'))
    ap.add_argument('--cache_dir', default=os.path.join(ROOT, 'cache'))
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    for fold in a.folds or config('sema')['folds']:
        build_caches(a.sema_dir, a.cache_dir, fold, device=a.device)


if __name__ == '__main__':
    main()
