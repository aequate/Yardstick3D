"""Pinned official VGGT inference for the cross-backbone pack.

Only frozen RGB images + timestamps enter the model. No evaluation GT exists in
the pack. Run on a CUDA GPU host via bootstrap.py.
"""
from pathlib import Path
import json
import subprocess
import sys
import time

import numpy as np

from verify import verify_inputs, verify_output, verify_metadata, digest

ROOT = Path(__file__).resolve().parent


def main():
    windows = verify_inputs(ROOT)
    manifest = json.loads((ROOT / 'input_hashes.json').read_text())
    n_frames = int(manifest['n_frames'])
    lock = json.loads((ROOT / 'model_lock.json').read_text())
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required; no CPU fallback')
    source = ROOT / 'source'
    if not source.exists():
        subprocess.run(['git', 'clone', lock['source_url'], str(source)], check=True)
        subprocess.run(['git', '-C', str(source), 'checkout', '--detach', lock['source_revision']], check=True)
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    assert revision == lock['source_revision'], 'Wrong source revision'
    assert not subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip(), 'Modified source'
    sys.path.insert(0, str(source))
    from huggingface_hub import hf_hub_download
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    checkpoint = hf_hub_download(repo_id=lock['model_id'], filename=lock['checkpoint'], revision=lock['model_revision'])
    assert Path(checkpoint).stat().st_size == lock['checkpoint_bytes']
    assert digest(checkpoint) == lock['checkpoint_sha256'], 'Checkpoint integrity failure'
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model = VGGT()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    model = model.eval().cuda()
    run_info = dict(
        lock=lock, pack_id=lock.get('pack_id'), sequence=lock.get('sequence'),
        torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
        contains_ground_truth=False, deterministic_bitwise_guarantee=False,
        dependencies=subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True),
    )
    out = ROOT / 'outputs'
    out.mkdir(exist_ok=True)
    (out / 'run_environment.json').write_text(json.dumps(run_info, indent=2))
    for entry, rec in windows:
        dest = out / f'{entry}.npz'
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            sidecar = dest.with_suffix('.json')
            if not sidecar.exists():
                raise RuntimeError(f'Incomplete output without sidecar: {dest}; move it (and any .partial.npz) out of outputs before retrying')
            verify_output(dest, np.array(rec['timestamps']))
            verify_metadata(dest, ROOT)
            continue
        images = load_and_preprocess_images([str(ROOT / entry / name) for name in rec['images']], mode='crop').cuda()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
            prediction = model(images)
        extrinsics, intrinsics = pose_encoding_to_extri_intri(prediction['pose_enc'], images.shape[-2:])
        asarray = lambda tensor: tensor.detach().float().cpu().numpy()
        payload = dict(
            timestamps=np.array(rec['timestamps']), T_w2c=asarray(extrinsics[0]),
            K=asarray(intrinsics[0]), depth_z=asarray(prediction['depth'][0, ..., 0]),
            depth_conf=asarray(prediction['depth_conf'][0]), is_metric=np.array(False),
            model_id=np.array('vggt-1b/facebook/VGGT-1B'),
        )
        temp = dest.with_suffix('.partial.npz')
        np.savez_compressed(temp, **payload)
        verify_output(temp, payload['timestamps'])
        temp.replace(dest)
        meta = dict(
            contains_ground_truth=False, source_revision=revision, model_revision=lock['model_revision'],
            input_manifest_sha256=digest(ROOT / 'input_hashes.json'), npz_sha256=digest(dest),
            n_frames=n_frames, is_metric=False, model_id='vggt-1b/facebook/VGGT-1B',
            license='CC-BY-NC-4.0', preprocess=lock['preprocess'], compute_dtype=lock['compute_dtype'],
            pose_convention='OpenCV world-to-camera 3x4; camera center = -R.T @ t',
            intrinsics_coordinates='official preprocessed/cropped image grid',
            depth_convention='camera Z, unknown global scale',
            runtime_seconds=time.perf_counter() - start, peak_vram_bytes=torch.cuda.max_memory_allocated(),
        )
        dest.with_suffix('.json').write_text(json.dumps(meta, indent=2))
        print('Exported', entry, flush=True)
        del prediction, images, extrinsics, intrinsics, payload
    print(f'All {len(windows)} frozen windows exported and validated.')


if __name__ == '__main__':
    main()
