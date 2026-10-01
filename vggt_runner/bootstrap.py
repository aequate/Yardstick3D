"""One command on the CUDA GPU host: python3.11 bootstrap.py."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent
if __name__ == '__main__':
    if sys.version_info[:2] != (3, 11):
        raise SystemExit('Use Python 3.11 and a CUDA GPU host with git installed.')
    if shutil.which('git') is None:
        raise SystemExit('git not found on PATH; install git on the GPU host first.')
    env = ROOT / 'runtime'
    if not env.exists():
        venv.EnvBuilder(with_pip=True).create(env)
    python = env / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    subprocess.run([str(python), '-m', 'pip', 'install', '-r', str(ROOT / 'requirements.txt')], check=True)
    subprocess.run([str(python), str(ROOT / 'run_vggt.py')], check=True)
    subprocess.run([str(python), str(ROOT / 'verify.py'), '--outputs'], check=True)
