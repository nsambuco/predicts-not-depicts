**Paper:** What a stimulus predicts, not what it depicts, determines striatal reward signals<img width="468" height="30" alt="image" src="https://github.com/user-attachments/assets/27c706ac-c391-48ad-a07e-5905682ff321" />

**Authors:** Nicola Sambuco and Francesco Versace.

## What this is

This repo holds the analysis code for the MRIeat study. In the experiment,
participants viewed pictures (pleasant, unpleasant, neutral, and food cues) while
they were scanned, and on food trials they decided whether to accept a juice
outcome. I was interested in how reward-related regions (striatum, vmPFC,
insula, and others) respond to food cues versus other emotional pictures, how
they track wanting (the cue) versus liking (the juice outcome), and how the
neural representations relate to behavior and to perceived satiety.

The code here takes ROI-level betas and the trial-level behavioral file and
produces every analysis and figure that goes into the paper:

- ROI beta extraction and one-sample tests against zero
- within-subject paired contrasts between conditions
- representational similarity analysis (RSA) with model RDMs and a permutation test
- the juice-acceptance behavioral analysis (acceptance rate and the alliesthesia slope across runs)
- the wanting versus liking Yes/No contrasts
- the SLIM exploratory analysis (pre-scan satiety against behavior and brain responses)
- the figures (functional ROIs, a priori ROIs, wanting/liking, RSA, behavior)
- the ROI surface and volume renders

## Repo layout

```
GitHub/
  README.md            this file
  LICENSE              MIT (covers the code only)
  requirements.txt     Python packages used by the scripts
  .gitignore           keeps real data out of the repo
  run_demo.py          runs the whole pipeline on the demo data
  code/                the analysis scripts
  data_demo/           synthetic demo data (see note below)
  figures/             where the outputs land when you run the demo
```

What is in each folder:

- `code/`
  - `mrieat_analysis_pipeline.py` runs the ROI analyses, the RSA, the
    juice-acceptance behavioral analysis, and makes all the figures.
  - `slim_analyses.py` runs the SLIM (pre-scan satiety) exploratory correlations.
  - `roi_visualization.py` renders the ROI masks on a volume and on a cortical
    surface. This one is reference only (see below).
  - `afni_proc.py` is the AFNI preprocessing command I used. It is reference
    only and the paths in it are placeholders (see below).
- `data_demo/` holds the synthetic CSV files the scripts read.
- `figures/` starts empty and fills up when you run the demo.

## Install and run

You need Python 3.10 or newer. From inside the `GitHub/` folder:

```
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_demo.py
```

That runs the main pipeline and the SLIM analysis on the demo data and writes
all the plots and stats reports into `figures/`. The RSA step runs 10,000
permutations, so give it a minute or two.

If you only want the core pipeline, you can skip nibabel and nilearn; those two
are only needed for `roi_visualization.py`.

## About the data

I cannot share the real participant data. It is human neuroimaging and
behavioral data and it is covered by data-use restrictions and participant
consent that do not allow public release.

So everything in `data_demo/` is **synthetic**. I generated it from random
noise with small condition offsets, using fake subject IDs (`sub-01` through
`sub-23`). It has the same column names, condition labels, and ROI columns as
the real input files, and it exists for two reasons only: to show you exactly
what format the real input files take, and to let the whole pipeline run end to
end so you can see the code works. The numbers, statistics, and figures that
come out of the demo are **not meaningful** and should not be interpreted or
cited as results. They are placeholders.

When you run the real study, you would drop your own ROI betas and behavioral
files into `data_demo/` (or repoint the paths at the top of each script) using
the same column layout as the demo files.

## Reference-only scripts

Two scripts are included for documentation but are not expected to run on the
demo:

- `code/afni_proc.py` is the AFNI preprocessing command. Every absolute subject
  path and local directory has been stripped and replaced with the placeholder
  `/path/to/data`, so it will not run as-is. It is there to document how the
  fMRI data were preprocessed.
- `code/roi_visualization.py` needs the ROI mask NIfTI files, which are not part
  of the demo. If the masks are not found it prints a note and exits cleanly.

## License

The code is released under the MIT License (see `LICENSE`), with copyright
2026 Nicola Sambuco. The license covers the **code only**. It does not cover any
data: the real participant data are not included and are not licensed for
release, and the synthetic demo data are provided purely as a format example.

## Citation

If you use this code, please cite the paper:

> Sambuco, N., et al. (year). Title. Journal, volume(issue), pages. DOI.

(Full citation to be filled in once the paper is published.)
