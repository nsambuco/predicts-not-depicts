#!/usr/bin/env python3
"""
run_demo.py
One command to show the analysis code runs end to end on the synthetic demo data.

It runs, in order:
  1. code/mrieat_analysis_pipeline.py  - ROI betas, within-subject contrasts,
                                         RSA, juice-acceptance behavior, and all figures
  2. code/slim_analyses.py             - SLIM exploratory correlations

Everything reads from data_demo/ and writes plots + stats into figures/.

Reminder: the demo data are synthetic, so the numbers and figures are NOT
meaningful. This is only here to show the pipeline works.

Run it with:
  python run_demo.py
"""
import os
import sys
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(ROOT, 'code')

SCRIPTS = [
    'mrieat_analysis_pipeline.py',   # ROI, contrasts, RSA, behavior, figures
    'slim_analyses.py',              # SLIM exploratory analysis
]


def main():
    for script in SCRIPTS:
        path = os.path.join(CODE, script)
        print('\n' + '=' * 70)
        print(f"Running {script}")
        print('=' * 70)
        result = subprocess.run([sys.executable, path], cwd=CODE)
        if result.returncode != 0:
            print(f"\n{script} exited with code {result.returncode}. Stopping.")
            sys.exit(result.returncode)

    print('\n' + '=' * 70)
    print("Demo finished. Figures and stats are in the figures/ folder.")
    print('=' * 70)


if __name__ == '__main__':
    main()
