"""Utterance labels and transcripts from the IEMOCAP release.

    Session<k>/dialog/EmoEvaluation/*.txt   [start - end]  <utt>  <emo>  [V, A, D]
    Session<k>/dialog/transcriptions/*.txt  <utt> [start-end]: text

Every feature file written by preprocess/ carries these fields, and the 4-class label
(neu=0, ang=1, hap=2, sad=3; -1 for every other emotion).
"""
import re
from pathlib import Path

EMOTION_NAMES = {'neu': 'neutral', 'ang': 'angry', 'hap': 'happy', 'sad': 'sad', 'fru': 'frustrated',
                 'exc': 'excited', 'fea': 'fearful', 'sur': 'surprised', 'dis': 'disgusted',
                 'oth': 'other', 'xxx': 'undefined'}
FOUR_CLASS = {'neu': 0, 'ang': 1, 'hap': 2, 'sad': 3}

_EVAL_RE = re.compile(r'^\[(\d+\.\d+)\s*-\s*(\d+\.\d+)\]\s+(\w+)\s+(\w+)\s+\[([0-9.,\s]+)\]')
_TRANS_RE = re.compile(r'^(\S+)\s+\[[^\]]+\]:\s*(.*)')


def _parse_eval(path):
    labels = {}
    with open(path, encoding='latin-1') as f:
        for line in f:
            m = _EVAL_RE.match(line.strip())
            if not m:
                continue
            start, end, uid, emo, vad = m.groups()
            v, a, d = [float(x.strip()) for x in vad.split(',')]
            labels[uid] = {'start': float(start), 'end': float(end), 'emotion': emo,
                           'emotion_name': EMOTION_NAMES.get(emo, emo),
                           'emotion_label': FOUR_CLASS.get(emo, -1), 'v': v, 'a': a, 'd': d}
    return labels


def _parse_trans(path):
    trans = {}
    with open(path, encoding='latin-1') as f:
        for line in f:
            m = _TRANS_RE.match(line.strip())
            if m:
                trans[m.group(1)] = m.group(2)
    return trans


def session_labels(session_dir):
    """-> (labels {utt: {...}}, transcripts {utt: text}) for one Session<k> directory."""
    session_dir = Path(session_dir)
    labels, trans = {}, {}
    for p in (session_dir / 'dialog' / 'EmoEvaluation').glob('*.txt'):
        labels.update(_parse_eval(p))
    for p in (session_dir / 'dialog' / 'transcriptions').glob('*.txt'):
        trans.update(_parse_trans(p))
    return labels, trans


def utterance_wavs(session_dir):
    """Sorted utterance WAVs of one session (macOS '._' files skipped)."""
    return sorted(p for p in (Path(session_dir) / 'sentences' / 'wav').rglob('*.wav') if not p.name.startswith('._'))
