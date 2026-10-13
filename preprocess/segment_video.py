#!/usr/bin/env python
"""Cut the IEMOCAP dialog videos into one clip per utterance.

    python preprocess/segment_video.py --raw_root /path/to/IEMOCAP_full_release

For every dialog video dialog/avi/DivX/<dialog>.avi the utterance times are read from
dialog/lab/Ses0<k>_F/<dialog>.lab and .../Ses0<k>_M/<dialog>.lab. Each utterance is
cropped to its speaker's half of the frame (female speaker, _F###: left half; male, _M###:
right half) and written as XVID to sentences/avi/<dialog>/<utt>.avi.

The frames are read sequentially (no seeking in the DivX source). Each dialog is
processed in subprocesses of at most 15 clips, retried on crashes, because some
OpenCV/XVID builds crash when many writers are opened in one process.
"""
import argparse
import glob
import os
import subprocess
import sys
import tempfile

CODEC, CHUNK_SIZE, MAX_RETRIES = 'XVID', 15, 5


def parse_lab(path):
    """[(start_sec, end_sec, utterance)] of one .lab file (silence rows skipped)."""
    segs = []
    with open(path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) == 3 and p[2].startswith('Ses'):
                segs.append((float(p[0]), float(p[1]), p[2]))
    return segs


def _segments(f_lab, m_lab):
    """All utterances of a dialog, sorted by start time, with the speaker side."""
    segs = []
    for lab, side in ((f_lab, 'F'), (m_lab, 'M')):
        if lab and os.path.exists(lab):
            segs += [(s, e, n, side) for s, e, n in parse_lab(lab)]
    segs.sort(key=lambda x: x[0])
    return segs


def worker(avi_path, f_lab, m_lab, out_dir, result_file, seg_start, seg_end):
    """Write clips segs[seg_start:seg_end] of one dialog (runs in its own process)."""
    import cv2
    segs_all = _segments(f_lab, m_lab)[seg_start:seg_end]
    cap = cv2.VideoCapture(avi_path)
    if not cap.isOpened():
        open(result_file, 'w').write(f'0 {len(segs_all)}')
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    half_w = width // 2
    fourcc = cv2.VideoWriter_fourcc(*CODEC)
    os.makedirs(out_dir, exist_ok=True)
    segs, skipped = [], 0
    for start, end, name, side in segs_all:
        sf, ef = int(start * fps), int(end * fps)
        if ef <= sf:
            skipped += 1
            continue
        segs.append({'start': sf, 'end': ef, 'x': 0 if side == 'F' else half_w,
                     'path': os.path.join(out_dir, f'{name}.avi'), 'writer': None})
    nxt, active, frame_idx, written = 0, [], 0, 0
    while nxt < len(segs) or active:
        ret, frame = cap.read()
        if not ret:
            break
        while nxt < len(segs) and segs[nxt]['start'] <= frame_idx:
            seg = segs[nxt]
            seg['writer'] = cv2.VideoWriter(seg['path'], fourcc, fps, (half_w, height))
            if not seg['writer'].isOpened():
                seg['writer'] = None
            active.append(seg); nxt += 1
        still = []
        for seg in active:
            if frame_idx < seg['end']:
                if seg['writer'] is not None:
                    seg['writer'].write(frame[:, seg['x']:seg['x'] + half_w])
                still.append(seg)
            else:
                if seg['writer'] is not None:
                    seg['writer'].release()
                written += 1
        active = still
        frame_idx += 1
    for seg in active:
        if seg['writer'] is not None:
            seg['writer'].release()
        written += 1
    cap.release()
    open(result_file, 'w').write(f'{written} {skipped}')


def run_chunk(avi_path, f_lab, m_lab, out_dir, seg_start, seg_end):
    """Run one worker subprocess, retrying crashes -> (written, skipped) or (0, -1)."""
    for _ in range(MAX_RETRIES):
        fd, result_file = tempfile.mkstemp(suffix='.txt'); os.close(fd)
        rc = subprocess.run([sys.executable, __file__, '--worker', avi_path, f_lab or '', m_lab or '', out_dir,
                             result_file, str(seg_start), str(seg_end)]).returncode
        try:
            if rc == 0:
                return tuple(map(int, open(result_file).read().split()))
        finally:
            os.remove(result_file)
    return 0, -1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--raw_root', help='IEMOCAP_full_release directory')
    ap.add_argument('--sessions', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument('--resume', action='store_true', help='skip dialogs that already have clips')
    ap.add_argument('--worker', nargs='+', help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.worker:
        avi, f_lab, m_lab, out_dir, result_file, s0, s1 = a.worker
        worker(avi, f_lab or None, m_lab or None, out_dir, result_file, int(s0), int(s1))
        return
    for session in a.sessions:
        sdir = os.path.join(a.raw_root, f'Session{session}')
        total = 0
        for avi in sorted(glob.glob(os.path.join(sdir, 'dialog', 'avi', 'DivX', '*.avi'))):
            dialog = os.path.splitext(os.path.basename(avi))[0]
            f_lab = os.path.join(sdir, 'dialog', 'lab', f'Ses0{session}_F', f'{dialog}.lab')
            m_lab = os.path.join(sdir, 'dialog', 'lab', f'Ses0{session}_M', f'{dialog}.lab')
            out_dir = os.path.join(sdir, 'sentences', 'avi', dialog)
            if a.resume and os.path.isdir(out_dir) and any(f.endswith('.avi') for f in os.listdir(out_dir)):
                continue
            n = len(_segments(f_lab, m_lab))
            for s0 in range(0, n, CHUNK_SIZE):
                w, s = run_chunk(avi, f_lab, m_lab, out_dir, s0, min(s0 + CHUNK_SIZE, n))
                if s < 0:
                    print(f'  {dialog}: clips {s0}-{s0 + CHUNK_SIZE} failed after {MAX_RETRIES} attempts', flush=True)
                total += w
        print(f'Session {session}: {total} clips', flush=True)


if __name__ == '__main__':
    main()
