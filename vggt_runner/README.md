# VGGT-1B runner

Template for an off-box VGGT-1B inference pack. `scripts/build_vggt_pack_v2.py` and
`scripts/build_vggt_pack_site_replication_r1.py` copy these files into a pack next to the selected RGB frames and
their hashes. The pack holds only frames, timestamps and model pins: no reference trajectory and no GNSS values.

On a CUDA GPU host (12 GB VRAM or more) with Python 3.11, `git` and internet access, from inside the pack:

```sh
python verify.py       # check the frozen inputs (no GPU needed)
python bootstrap.py    # isolated venv, pinned VGGT revision, SHA-256-checked checkpoint, inference, output checks
```

| File | Role |
|---|---|
| `bootstrap.py` | creates `runtime/`, installs `requirements.txt`, runs `run_vggt.py`, then `verify.py` |
| `run_vggt.py` | VGGT-1B inference over every frozen window |
| `verify.py` | validates inputs against their hashes and every output against the schema |
| `model_lock.json` | VGGT source revision and checkpoint SHA-256 |

Each window produces `outputs/<seq>/tNNN_NNN.npz` with a `.json` sidecar. The npz holds `timestamps` (N),
`T_w2c` (N,3,4), `K` (N,3,3), `depth_z` (N,H,W), `depth_conf` (N,H,W), `is_metric=false` and `model_id`.
Cameras are OpenCV world-to-camera (camera centre `-R.T @ t`). Depth is camera Z in the model's unknown scale.
