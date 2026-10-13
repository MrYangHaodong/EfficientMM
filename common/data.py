"""IEMOCAP data for SeMA and the prefix cache.

``build_data`` loads train/val/test with the IEMOCAP loader and preloads every batch ONCE as
``(high, y)``: ``high = {modality: FloatTensor[B, T, feat_dim]}`` with every modality resampled
or zero-padded to ``T = max_seq_len``, ``y = LongTensor[B]``. The training split is shuffled
once, when it is preloaded, so a run sees the same batch order in every epoch.
"""
import torch

from common.paths import IEMOCAP_ROOT

MODALITIES = ['video', 'audio', 'text', 'mocap_hand', 'mocap_head', 'mocap_rotated']
NUM_CLASSES = 4


def build_data(fold, batch_size=32, num_workers=4, max_seq_len=100):
    """-> (train_batches, val_batches, test_batches, modalities, num_classes).

    Loader batches are ``[high_0, low_0, high_1, low_1, ..., label]``; only the
    full-resolution ``high`` views are kept."""
    from data.IEMOCAP.get_data import get_dataloader
    tl, vl, el = get_dataloader(data_root=IEMOCAP_ROOT, batch_size=batch_size, num_workers=num_workers,
                                modalities=MODALITIES, split_id=fold, train_shuffle=True,
                                max_seq_len=max_seq_len, time_compression_ratio=4,
                                use_batched_fusion=True, split_mode='cross_subject')

    def preload(loader):
        out = []
        for batch in loader:
            *feats, label = batch
            high = {m: feats[2 * j].float().contiguous() for j, m in enumerate(MODALITIES)}
            out.append((high, label.view(-1).long()))
        return out

    train_b, val_b, test_b = preload(tl), preload(vl), preload(el)
    print(f'[data] iemocap fold{fold}: T={max_seq_len}, train/val/test = '
          f'{sum(len(b[1]) for b in train_b)}/{sum(len(b[1]) for b in val_b)}/{sum(len(b[1]) for b in test_b)}',
          flush=True)
    return train_b, val_b, test_b, list(MODALITIES), NUM_CLASSES


def rebatch(batches, size=256):
    """Concatenate preloaded batches and re-split them into chunks of ``size`` (eval only)."""
    high = {m: torch.cat([b[0][m] for b in batches]) for m in MODALITIES}
    y = torch.cat([b[1] for b in batches])
    return [({m: high[m][s:s + size] for m in MODALITIES}, y[s:s + size])
            for s in range(0, len(y), size)], y.numpy()
