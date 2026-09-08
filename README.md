# BoneAGUNet

Atlas-guided erosion segmentation for an already-cropped MCP2 or MCP3 joint
image stack. The package interface deliberately excludes cohort discovery,
SLURM, and ARC filesystem paths.

## Pipeline overview

<p align="center">
  <img src="assets/pipeline-overview.png" width="680" alt="AG-UNet erosion detection and segmentation pipeline">
</p>

The input image is converted into an edge mask and a closed bone mask while a
healthy atlas is deformably registered to the same anatomy. Their difference
identifies erosion candidates. Each candidate is passed to AG-UNet as a
three-channel ROI containing the image, bone edges, and registered atlas.

## Installation and use

```bash
git clone https://github.com/ManskeLab/BoneAGUNet.git
cd BoneAGUNet
pip install .
boneagunet-install ~/.cache/boneagunet
boneagunet -i joint_mcp2.nii.gz -o erosions.nii.gz --mcp 2
```

Python:

```python
from boneagunet import run
run("joint_mcp3.nii.gz", "erosions.nii.gz", mcp=3)
```

ANTs (`antsRegistration` and `antsApplyTransforms`) must be installed separately.
The package installs the pinned Manske Lab nnU-Net fork required by the attention
checkpoints.

The input is one unprocessed, already stack-registered 3-D MCP joint image. The
pipeline performs soft-tissue stripping, MC/PP masking, edge and closed-edge
prediction, atlas registration, candidate extraction, erosion prediction, and
recombination into the input image geometry. `--keep-work` preserves all
intermediates for inspection.

Use `--modality sr-cbct` for SR-CBCT inputs; HR-pQCT is the default.

## Atlases and intermediate masks

<table>
  <tr>
    <th>MCP2 atlas</th>
    <th>MCP3 atlas</th>
  </tr>
  <tr>
    <td><img src="assets/mcp2-atlas.png" width="330" alt="Healthy MCP2 atlas maximum intensity projection"></td>
    <td><img src="assets/mcp3-atlas.png" width="330" alt="Healthy MCP3 atlas maximum intensity projection"></td>
  </tr>
</table>

The healthy representative HR-pQCT atlases are shown as sagittal maximum
intensity projections.

<table>
  <tr>
    <th>Input image</th>
    <th>Predicted edge mask</th>
    <th>Closed bone mask</th>
  </tr>
  <tr>
    <td><img src="assets/edge-input.png" width="220" alt="Original HR-pQCT bone image"></td>
    <td><img src="assets/edge-mask.png" width="220" alt="Predicted cortical bone edge mask"></td>
    <td><img src="assets/closed-edge-mask.png" width="220" alt="Predicted closed cortical bone mask"></td>
  </tr>
</table>

The second segmentation model receives both the image and predicted edge mask
to complete the cortical surface used during atlas subtraction.

## Model assets

The trained models and MCP2/MCP3 atlases are hosted at
[YousifKhoury/BoneAGUNet](https://huggingface.co/YousifKhoury/BoneAGUNet) and are
downloaded automatically or explicitly with `boneagunet-install`.