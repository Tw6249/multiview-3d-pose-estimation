"""CPU deployment entry point: setup, test, baseline, offline, render.

Uses subprocess argument lists and process-local environment variables, without
changing the user's shell profile, CUDA paths or machine security settings.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("task", choices=("setup","test","baseline","offline","render"))
    ap.add_argument("--model-zip", type=Path)
    args = ap.parse_args()
    python = ROOT / "outputs/local_cpu/runtime/Scripts/python.exe"
    env = dict(os.environ, G1_IMITATION_CONFIG=str(ROOT / "outputs/local_cpu/pipeline_config.json"),
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

    def run(*command):
        subprocess.run([str(s) for s in command], cwd=ROOT, env=env, check=True)

    if args.task == "setup":
        if not python.exists():
            run("uv","venv","--python","3.11.9","outputs/local_cpu/runtime")
        run("uv","pip","sync","--python",python,"--index-url","https://pypi.tuna.tsinghua.edu.cn/simple",
            "requirements-cpu-lock.txt")
        model_args = ["--model-zip",args.model_zip.resolve()] if args.model_zip else []
        run(python,"scripts/prepare_local_cpu.py",*model_args)
        manifest = json.loads((ROOT / "g1_imitation/config/local_cpu_assets.json").read_text(encoding="utf-8"))
        run(python,"scripts/prepare_cpu_vision.py","--mamma-commit",manifest["mamma_calibration_commit"])
    elif args.task == "test":
        names = ("local_config","cpu_offline_calibration","s2_triangulation","s3_temporal_filter",
                 "s1b_person_association","temporal_alignment","s45c_keypoint_ik","adv_eval",
                 "control_rate_reference","ik_observation_gate","retarget_experiments",
                 "structured_history_ik","s6_sonic_policy","hardware_prep_a1_a5","calibrate_hik_cameras")
        run(python,"-m","pytest",*[f"tests/test_{name}.py" for name in names],
            "-k","not roban and not heft","-q","--junitxml=outputs/local_cpu/regression.xml")
    elif args.task == "baseline":
        run(python,"scripts/local_cpu_baseline.py")
    elif args.task == "offline":
        run(python,"scripts/local_cpu_offline.py","--frames","426","--out","outputs/local_cpu/offline_mamma_full")
    else:
        run(python,"scripts/render_local_cpu.py")


if __name__ == "__main__":
    main()
