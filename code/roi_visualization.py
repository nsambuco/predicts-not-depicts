#!/usr/bin/env python3
"""
roi_visualization.py
Renders the a priori and functional ROI masks on a volume (orthogonal slices)
and on an inflated cortical surface, for the brain figures in the paper.

This one is reference only. The ROI mask files (.nii / .nii.gz) are anatomical
masks in MNI space and are not part of the demo, so by default there is nothing
for this script to draw. Point ROI_MASKS at your own mask files (one NIfTI per
ROI, or a single labelled atlas) and it will produce the renders. If the masks
are not found it just prints a note and exits cleanly.

Outputs (when masks are present): ../figures/ROI_volume.png, ../figures/ROI_surface.png
"""
import os
import nibabel as nib
from nilearn import plotting, datasets
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, '..', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# Point these at your own ROI masks in MNI space (placeholders by default).
MASK_DIR = '/path/to/data/roi_masks'
ROI_MASKS = {
    'NAc':      os.path.join(MASK_DIR, 'NAc.nii.gz'),
    'PUT':      os.path.join(MASK_DIR, 'PUT.nii.gz'),
    'vmPFC':    os.path.join(MASK_DIR, 'vmPFC.nii.gz'),
    'dAI':      os.path.join(MASK_DIR, 'dAI.nii.gz'),
    'vAI':      os.path.join(MASK_DIR, 'vAI.nii.gz'),
    'PI':       os.path.join(MASK_DIR, 'PI.nii.gz'),
    'Visual':   os.path.join(MASK_DIR, 'Visual.nii.gz'),
    'mPFC':     os.path.join(MASK_DIR, 'mPFC.nii.gz'),
    'IFG':      os.path.join(MASK_DIR, 'IFG.nii.gz'),
    'Amygdala': os.path.join(MASK_DIR, 'Amygdala.nii.gz'),
}

# Distinct color per ROI for the overlays.
ROI_COLORS = {
    'NAc': '#e41a1c', 'PUT': '#377eb8', 'vmPFC': '#4daf4a', 'dAI': '#984ea3',
    'vAI': '#ff7f00', 'PI': '#a65628', 'Visual': '#f781bf', 'mPFC': '#999999',
    'IFG': '#66c2a5', 'Amygdala': '#ffd92f',
}


def main():
    available = {name: p for name, p in ROI_MASKS.items() if os.path.exists(p)}
    if not available:
        print("No ROI mask files found.")
        print(f"Edit MASK_DIR / ROI_MASKS to point at your masks (looked in {MASK_DIR}).")
        print("This script is reference only and is not expected to run on the demo data.")
        return

    print(f"Found {len(available)} ROI masks: {', '.join(available)}")

    # --- Volume render: orthogonal slices with each ROI overlaid ---
    print("Rendering volume view...")
    display = plotting.plot_anat(display_mode='ortho', title='ROIs (volume)')
    for name, path in available.items():
        display.add_contours(nib.load(path), levels=[0.5],
                             colors=[ROI_COLORS.get(name, 'red')], linewidths=1.5)
    out_vol = os.path.join(FIG_DIR, 'ROI_volume.png')
    display.savefig(out_vol, dpi=300)
    display.close()
    print(f"  Saved {out_vol}")

    # --- Surface render: project masks onto an inflated fsaverage surface ---
    print("Rendering surface view...")
    fsaverage = datasets.fetch_surf_fsaverage()
    fig, axes = plt.subplots(1, len(available), figsize=(4 * len(available), 4),
                             subplot_kw={'projection': '3d'})
    if len(available) == 1:
        axes = [axes]
    for ax, (name, path) in zip(axes, available.items()):
        tex = plotting.surface.vol_to_surf(nib.load(path), fsaverage.pial_left)
        plotting.plot_surf_roi(fsaverage.infl_left, roi_map=tex,
                               hemi='left', view='lateral',
                               bg_map=fsaverage.sulc_left, axes=ax,
                               title=name, cmap='autumn')
    out_surf = os.path.join(FIG_DIR, 'ROI_surface.png')
    fig.savefig(out_surf, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out_surf}")


if __name__ == '__main__':
    main()
