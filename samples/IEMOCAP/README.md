# IEMOCAP sample set

Eight utterances from IEMOCAP, each with its raw data and the features our pipeline consumes.
Two per emotion class (neutral, angry, happy, sad); one female and one male speaker per class;
sessions 1–5, improvised and scripted dialogs. `manifest.csv` lists them with labels,
valence/activation/dominance, timing and transcript.

```
<utterance_id>/
  raw/
    <utt>.wav              utterance audio (16 kHz mono)
    <utt>.avi              utterance video clip, cropped to the speaker's half of the dialog video
    MOCAP_hand.txt         raw motion capture: hand markers (T x 18)
    MOCAP_head.txt         head pitch/roll/yaw + translation (T x 6)
    MOCAP_rotated.txt      face markers (T x 165)
    transcription.txt      the utterance's line from dialog/transcriptions/<dialog>.txt
    label_EmoEvaluation.txt its block from dialog/EmoEvaluation/<dialog>.txt (consensus label, V/A/D, annotators)
    timing.lab             start/end seconds from dialog/lab/
  features/
    video_features.pkl     DINOv3 ViT-L/16 CLS token per frame, 100 x 1024
    audio_features.pkl     wav2vec2 XLS-R 1B last layer on the first 12 s, 1 x 599 x 1280
    text_features.pkl      bert-base-uncased last hidden state per token, T x 768 (incl. [CLS], [SEP])
    MOCAP_*.txt            identical to the raw files: motion capture is used as is
  feature_summary.json     shapes, dtypes and the metadata stored in each pickle
```

Each `.pkl` is a dict with `features` plus the utterance's metadata (id, session, dialog,
transcript, emotion, 4-class label, V/A/D, start/end). Labels: neu 0, ang 1, hap 2, sad 3.
The raw utterance clips were cut from the release's dialog videos `Session<k>/dialog/avi/DivX/<dialog>.avi`
(not included, ~20 MB each) by `preprocess/segment_video.py`; the features were produced by the
`preprocess/extract_*.py` scripts of this repository. See the top-level README for how to run
SeMARC on these samples with and without raw-data extraction.
