"""Locations: the IEMOCAP data root (env ``IEMOCAP_ROOT``) and the config files."""
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IEMOCAP_ROOT = os.environ.get('IEMOCAP_ROOT', os.path.join(REPO, 'datasets', 'IEMOCAP'))


def config(name: str) -> dict:
    """configs/<name>.json"""
    return json.load(open(os.path.join(REPO, 'configs', f'{name}.json')))
