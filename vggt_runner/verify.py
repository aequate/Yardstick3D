"""Standalone input/output validation for the cross-backbone VGGT pack.

Reads no dataset, no evaluation GT channel. Window count / frame count are read
from input_hashes.json so the pack generalises beyond the legacy ADVIO pack.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def contained(root, relative):
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError('Path escapes pack')
    return path


def verify_inputs(root):
    root = Path(root)
    manifest = json.loads((root / 'input_hashes.json').read_text())
    assert manifest['contains_ground_truth'] is False
    n_frames = int(manifest['n_frames'])
    n_windows = int(manifest['n_windows'])
    windows_list = manifest['windows']
    assert len(windows_list) == n_windows and len(set(windows_list)) == n_windows
    required = set()
    for entry in windows_list:
        required.update(
            f'{entry}/{name}'
            for name in ['window.json', 'timestamps.npy'] + [f'f{i:02d}.png' for i in range(n_frames)]
        )
    for seq in manifest['sequences']:
        required.add(f'{seq}/manifest.json')
    assert set(manifest['files']) == required, 'Incomplete or extra input inventory'
    for path, expected in manifest['files'].items():
        assert digest(contained(root, path)) == expected, f'Hash mismatch: {path}'
    windows = []
    for entry in windows_list:
        folder = contained(root, entry)
        rec = json.loads((folder / 'window.json').read_text())
        allowed = {'t0', 't1', 'timestamps', 'frame_indices', 'images', 'pose_convention', 'is_metric'}
        assert set(rec) <= allowed, 'Unexpected metadata (possible GT contamination)'
        assert rec['is_metric'] is False
        ts = np.load(folder / 'timestamps.npy', allow_pickle=False)
        assert ts.shape == (n_frames,) and np.isfinite(ts).all() and np.all(np.diff(ts) > 0)
        np.testing.assert_array_equal(ts, np.array(rec['timestamps']))
        assert rec['images'] == [f'f{i:02d}.png' for i in range(n_frames)]
        assert len(rec['frame_indices']) == n_frames
        assert np.all(np.diff(rec['frame_indices']) > 0)
        windows.append((entry, rec))
    for seq in manifest['sequences']:
        seq_manifest = json.loads((root / seq / 'manifest.json').read_text())
        seq_entries = [e for e, _ in windows if e.startswith(seq + '/')]
        assert seq_manifest['n_windows'] == len(seq_entries)
        assert len(seq_manifest['windows']) == len(seq_entries)
        by_folder = {e.split('/')[1]: rec for e, rec in windows if e.startswith(seq + '/')}
        for mrec in seq_manifest['windows']:
            folder = f"t{int(mrec['t0'])}_{int(mrec['t1'])}"
            assert folder in by_folder, f'Manifest entry without frozen window: {seq}/{folder}'
            wrec = by_folder[folder]
            for key in ('t0', 't1', 'timestamps', 'frame_indices', 'images', 'pose_convention', 'is_metric'):
                assert mrec[key] == wrec[key], f'Manifest/window mismatch: {seq}/{folder} key {key}'
    return windows


def verify_output(npz, timestamps):
    n_frames = int(len(timestamps))
    with np.load(npz, allow_pickle=False) as z:
        required = {'timestamps', 'T_w2c', 'K', 'depth_z', 'depth_conf', 'is_metric', 'model_id'}
        assert set(z.files) == required, 'Unexpected or missing prediction fields'
        np.testing.assert_array_equal(z['timestamps'], timestamps)
        assert z['T_w2c'].shape == (n_frames, 3, 4) and z['K'].shape == (n_frames, 3, 3)
        assert z['depth_z'].ndim == 3 and z['depth_z'].shape[0] == n_frames
        assert z['depth_conf'].shape == z['depth_z'].shape
        assert not bool(z['is_metric'].item())
        assert z['model_id'].item() == 'vggt-1b/facebook/VGGT-1B'
        for key in ('T_w2c', 'K', 'depth_z', 'depth_conf'):
            assert np.isfinite(z[key]).all()
        assert np.all(z['depth_z'] > 0) and np.all(z['depth_conf'] >= 0)
        np.testing.assert_allclose(z['K'][:, 2, :], np.tile([0., 0., 1.], (n_frames, 1)), atol=1e-6)
        assert np.all(z['K'][:, 0, 0] > 0) and np.all(z['K'][:, 1, 1] > 0)
        rot = z['T_w2c'][:, :, :3]
        np.testing.assert_allclose(
            rot @ rot.transpose(0, 2, 1),
            np.broadcast_to(np.eye(3), (n_frames, 3, 3)), atol=2e-3,
        )
        np.testing.assert_allclose(np.linalg.det(rot), np.ones(n_frames), atol=2e-3)


def verify_metadata(npz, root):
    npz, root = Path(npz), Path(root)
    manifest = json.loads((root / 'input_hashes.json').read_text())
    meta = json.loads(npz.with_suffix('.json').read_text())
    lock = json.loads((root / 'model_lock.json').read_text())
    assert meta['contains_ground_truth'] is False and meta['is_metric'] is False
    assert meta['npz_sha256'] == digest(npz)
    assert meta['input_manifest_sha256'] == digest(root / 'input_hashes.json')
    assert meta['source_revision'] == lock['source_revision']
    assert meta['model_revision'] == lock['model_revision']
    assert meta.get('model_id') == 'vggt-1b/facebook/VGGT-1B'
    assert int(meta['n_frames']) == int(manifest['n_frames'])
    return meta


if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    windows = verify_inputs(root)
    if '--outputs' in sys.argv:
        for entry, rec in windows:
            path = root / 'outputs' / f'{entry}.npz'
            verify_output(path, np.asarray(rec['timestamps']))
            verify_metadata(path, root)
    print(f'PASS: {len(windows)} frozen windows; outputs checked={"--outputs" in sys.argv}')
