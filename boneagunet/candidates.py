import argparse
import SimpleITK as sitk
import numpy as np
import os
from scipy.ndimage import binary_opening, binary_closing, distance_transform_edt, \
    binary_fill_holes, binary_erosion, binary_dilation
from scipy.ndimage import label as nd_label
from skimage.measure import label, regionprops, marching_cubes, mesh_surface_area
from scipy.spatial import cKDTree
import csv

from skimage import morphology

def merge_close_erosions(labeled, spacing=(0.0607,0.0607,0.0607), max_gap_mm=1):
    """
    Merge components whose surfaces are within max_gap_mm (approx) without growing them.

    labeled: int array (0=bg, 1..N components)
    spacing: voxel spacing (zyx) in mm
    max_gap_mm: merge if blobs are closer than this (mm)
    """
    # Approximate radius in voxels. Use smallest spacing so we don't miss merges.
    min_sp = float(min(spacing))
    r_vox = max(1, int(np.ceil(max_gap_mm / min_sp)))

    # 1) Work on binary to find connectivity after dilation
    binmask = (labeled > 0)
    ball = morphology.ball(r_vox)  # 3D isotropic structuring element
    bin_dil = morphology.binary_dilation(binmask, ball)

    # 2) Label the dilated mask — blobs whose dilations touch get one merged id
    merged_dil_labels, _ = nd_label(bin_dil)

    # 3) Map merged-dil regions back to unions of ORIGINAL components (no growth)
    out = np.zeros_like(labeled, dtype=np.uint16)
    next_id = 1
    # Iterate over merged regions
    for merged_id in range(1, merged_dil_labels.max() + 1):
        idx = (merged_dil_labels == merged_id)
        # Which original labels live inside this merged region?
        orig_ids = np.unique(labeled[idx])
        orig_ids = orig_ids[orig_ids > 0]
        if orig_ids.size == 0:
            continue
        # Union of the original components (no dilation)
        union_mask = np.isin(labeled, orig_ids)
        out[union_mask] = next_id
        next_id += 1

    return out

def run_gac_levelset(edge_img, erosion_seed, atlas_mask, iterations=80):
    """
    Refine erosion mask using SimpleITK's geodesic active contour level set in 3D.

    Parameters:
        edge_seg:     binary 3D edge segmentation (numpy array, 1 = edge)
        erosion_seed: binary 3D erosion seed (numpy array)
        atlas_mask:   binary 3D bone mask (numpy array)
        iterations:   number of iterations
        smoothing:    unused (for compatibility)
        balloon:      propagation scaling factor

    Returns:
        3D refined erosion mask (uint8)
    """

    # Convert numpy arrays to SimpleITK images
    seed_img = sitk.GetImageFromArray(erosion_seed.astype(np.uint8))
    mask_img = sitk.GetImageFromArray(atlas_mask.astype(np.uint8))

    seed_img.CopyInformation(edge_img)
    mask_img.CopyInformation(edge_img)

    # Pre-smooth edge map
    edge_float = sitk.Cast(edge_img, sitk.sitkFloat32)
    smoothed = sitk.CurvatureFlow(image1=edge_float, timeStep=0.125, numberOfIterations=5)

    # Convert to signed distance map for initial level set
    distance_filter = sitk.SignedMaurerDistanceMapImageFilter()
    distance_filter.SetInsideIsPositive(True)
    distance_filter.SetUseImageSpacing(False)
    distance_filter.SetSquaredDistance(False)
    distance_filter.SetBackgroundValue(0)
    init_ls = distance_filter.Execute(seed_img)
    # init_ls = sitk.Clamp(init_ls, lowerBound=-100.0, upperBound=100.0)
    # sitk.WriteImage(init_ls, 'init_ls.nii.gz')
    # sitk.WriteImage(seed_img, 'seed_img.nii.gz')

    # Threshold level set
    thresh_filter = sitk.ThresholdSegmentationLevelSetImageFilter()
    thresh_filter.SetLowerThreshold(0.8)  # These assume edge map is binary or [0,1]
    thresh_filter.SetUpperThreshold(1)
    thresh_filter.SetCurvatureScaling(0.5)
    thresh_filter.SetNumberOfIterations(iterations)
    thresh_filter.SetMaximumRMSError(0.01)

    ls_out = thresh_filter.Execute(sitk.Cast(init_ls, sitk.sitkFloat32), sitk.Cast(edge_float, sitk.sitkFloat32))
    # sitk.WriteImage(ls_out, 'ls_out.nii.gz')
    # Threshold final result and mask
    output = sitk.GetArrayFromImage(ls_out > 0)
    atlas_np = sitk.GetArrayFromImage(mask_img)
    return (output & atlas_np).astype(np.uint8)

def get_signed_distance_map(image_np):

    image = sitk.GetImageFromArray(image_np)

    dist_map = sitk.SignedMaurerDistanceMap(
    image, insideIsPositive=False, useImageSpacing=False)

    return sitk.GetArrayFromImage(dist_map)

def compute_erosion_metrics(subtraction, reference_image, subject_name="unknown",
                            spacing=(0.0607, 0.0607, 0.0607)):
    """
    Compute morphological descriptors from a binary 3D erosion mask.

    Parameters:
        subtraction (ndarray): binary erosion mask
        reference_image (sitk.Image): original RA or atlas image (to get spacing)
        subject_name (str): identifier

    Returns:
        dict of metrics
    """
    spacing = tuple(float(value) for value in spacing)
    labeled = label(subtraction)
    props = regionprops(labeled)

    if len(props) == 0:
        print(f"No erosion found in: {subject_name}")
        return None

    # Assuming one erosion component — you can loop if needed
    region = props[0]
    voxel_volume = np.prod(spacing)

    # Volume in mm³
    volume = region.area * voxel_volume

    # Centroid in physical coordinates
    centroid_index = np.array(region.centroid)
    centroid_phys = np.array(reference_image.TransformContinuousIndexToPhysicalPoint(centroid_index[::-1]))

    # Bounding box dimensions (in mm)
    bbox = region.bbox
    bbox_dims_vox = np.array(bbox[3:]) - np.array(bbox[:3])
    bbox_dims_mm = bbox_dims_vox * spacing

    # Max diameter (approximate via distance transform)
    dist = distance_transform_edt(subtraction, sampling=spacing)
    max_diameter = dist.max() * 2

    # Surface area using marching cubes
    verts, faces, _, _ = marching_cubes(subtraction, level=0.5, spacing=spacing)
    surface_area = mesh_surface_area(verts, faces)

    # Shape descriptors
    sphericity = (np.pi ** (1/3)) * ((6 * volume) ** (2/3)) / (surface_area + 1e-8)
    compactness = (volume ** 2) / (surface_area ** 3 + 1e-8)

    # Elongation from inertia tensor (via PCA)
    cov = region.inertia_tensor
    eigvals = np.linalg.eigvalsh(cov)
    elongation = eigvals[-1] / (eigvals[0] + 1e-8)  # Major / minor

    return {
        "Subject": subject_name,
        "Volume_mm3": volume,
        "SurfaceArea_mm2": surface_area,
        "MaxDiameter_mm": max_diameter,
        "Sphericity": sphericity,
        "Compactness": compactness,
        "Centroid_mm": centroid_phys,
        "BBox_mm": bbox_dims_mm,
        "Elongation": elongation
    }

def get_outer_contour(mask, iterations=1):
    """
    Returns the outer 1-voxel thick contour of a binary 3D mask.

    Parameters:
        mask (ndarray): binary 3D mask
        iterations (int): thickness of erosion

    Returns:
        ndarray: binary outer contour
    """
    eroded = binary_erosion(mask, iterations=iterations)
    contour = mask & ~eroded
    return contour.astype(np.uint8)

def snap_outer_shell_to_edge_kdtree(filled_mask, edge_map, snap_radius=1000):
    """
    Snap the outer boundary of a soft filled mask to the nearest cortical edge using KDTree.

    Parameters:
        filled_mask (ndarray): binary 3D mask of MC bone
        edge_map (ndarray): binary 3D cortical edge map
        snap_radius (int): max distance (in voxels) to allow snapping

    Returns:
        ndarray: refined mask with snapped cortical surface
    """
    # Step 1: Extract boundary of filled mask
    eroded = binary_erosion(filled_mask)
    boundary = filled_mask ^ eroded
    boundary_coords = np.argwhere(boundary)

    # edge_map = binary_dilation(edge_map, iterations=2)

    # Step 2: Get coordinates of cortical edges
    edge_coords = np.argwhere(edge_map > 0)
    if edge_coords.size == 0:
        print("Warning: Edge map is empty.")
        return filled_mask.copy()

    # Step 3: Snap boundary points to nearest edge voxel using KDTree
    tree = cKDTree(edge_coords)
    distances, indices = tree.query(boundary_coords)

    # Keep only boundary points that are near some edge voxel
    keep_mask = distances <= 1000
    snapped_coords = edge_coords[indices[keep_mask]]

    # Step 4: Build refined mask
    refined = np.zeros_like(filled_mask, dtype=bool)
    refined |= eroded  # inner part stays the same

    # Add snapped outer shell
    for coord in snapped_coords:
        z, y, x = coord
        refined[z, y, x] = 1

    # Step 5: Keep largest connected component
    labeled, num = nd_label(refined)
    if num == 0:
        return np.zeros_like(refined, dtype=np.uint8)

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    largest = np.argmax(sizes)
    return (labeled == largest).astype(np.uint8)

def match_histogram_simpleitk(source_img, reference_img):
    return sitk.HistogramMatching(source_img, reference_img)

def make_filled_mask(edge_np):
    # Fill holes slice-by-slice in all 3 directions
    filled = np.zeros_like(edge_np, dtype=bool)

    # Axial (z-axis)
    for z in range(edge_np.shape[0]):
        filled[z, :, :] = binary_fill_holes(edge_np[z, :, :])

    # Coronal (y-axis)
    for y in range(edge_np.shape[1]):
        filled[:, y, :] = binary_fill_holes(filled[:, y, :])

    # Sagittal (x-axis)
    for x in range(edge_np.shape[2]):
        filled[:, :, x] = binary_fill_holes(filled[:, :, x])

    return filled.astype(np.uint8)

def segment_erosions(atlas_path, ra_path, edge_path, ra_mask_path, sr, output_dir,
                     native_spacing=(0.0607, 0.0607, 0.0607)):

    # File name
    filename = os.path.basename(ra_path)
    subject = os.path.splitext(filename)[0]
    subject = os.path.splitext(subject)[0]

    # Load images
    atlas = sitk.ReadImage(atlas_path, sitk.sitkFloat32)

    atlas_np = sitk.GetArrayFromImage(atlas)

    atlas_mask = (atlas_np>0.02) | (atlas_np<-0.02)
    atlas_mask = binary_fill_holes(atlas_mask)
    atlas_mask = binary_erosion(atlas_mask > 0, iterations=2)

    ra_org = sitk.ReadImage(ra_path, sitk.sitkFloat32)
    ra_mask = ra_org != 0
    ra = match_histogram_simpleitk(ra_org, atlas)

    edge = sitk.ReadImage(edge_path, sitk.sitkUInt8)
    edge_np = sitk.GetArrayFromImage(edge)

    ra_np = sitk.GetArrayFromImage(ra)
    ra_bone_hd_tresh = np.percentile(ra_np, 90)
    print(ra_bone_hd_tresh)
    ra_winsorized_np = ra_np >= ra_bone_hd_tresh

    edge_winsorized_np = edge_np
    atlas_mask = atlas_mask | ra_winsorized_np

    # ITERATIONS=4
    # # if sr:
    # #     edge_np = binary_erosion(edge_np > 0, iterations=1)

    # edge_dilated_np = binary_dilation(edge_winsorized_np > 0, iterations=ITERATIONS)
    
    # for z in range(1, 80):
    #     edge_dilated_np[-z, :, :] = binary_fill_holes(edge_dilated_np[-z, :, :])

    # edge_filled_np = binary_fill_holes(edge_dilated_np)
    # edge_filled_eroded_np = binary_erosion(edge_filled_np > 0, iterations=ITERATIONS+2)
    # edge_filled_eroded_np = edge_filled_eroded_np | ra_winsorized_np
    
    # edge = sitk.ReadImage(edge_path, sitk.sitkUInt8)
    # edge_np = sitk.GetArrayFromImage(edge)

    # if sr:
    #     edge_np = binary_erosion(edge_np > 0, iterations=2)
    #     edge_np = edge_np | (ra_np > 1)

    # edge_new = edge_np | edge_filled_eroded_np

    # edge_new_img = sitk.GetImageFromArray(edge_new.astype(np.uint8))
    # edge_new_img.CopyInformation(edge)
    # # sitk.WriteImage(edge_new_img, 'edge_new_img.nii.gz')

    # for z in range(1, 80):
    #     edge_new[-z, :, :] = binary_fill_holes(edge_new[-z, :, :])

    # # for z in range(0, edge_new.shape[0], 20):
    # #     edge_new[z, :, :] = binary_fill_holes(edge_new[z, :, :])

    # closing_filter = sitk.BinaryMorphologicalClosingImageFilter()
    # closing_filter.SetKernelRadius(2)
    # closing_filter.SetForegroundValue(1)

    # edge_new = sitk.GetImageFromArray(edge_new.astype(np.uint8))
    # edge_new.CopyInformation(edge)
    # edge_filled = sitk.BinaryFillhole(edge_new)

    # edge_filled = closing_filter.Execute(edge_filled)
    # edge_filled = sitk.BinaryFillhole(edge_filled)

    # # edge_filled.CopyInformation(ra_mask)

    # ra_mask = sitk.Mask(edge_filled, ra_mask)

    ra_mask = sitk.ReadImage(ra_mask_path)
    sitk.WriteImage(ra_mask, os.path.join(output_dir, f"{subject}_ra_mask.nii.gz"))
    ra_mask = sitk.GetArrayFromImage(ra_mask)

    # Convert edge map to binary
    edge_bin = (edge_np > 0).astype(np.uint8)

    atlas_outer = get_outer_contour(atlas_mask, iterations=2)

    # subtraction = atlas - ra
    # sitk.WriteImage(subtraction>0, 'subtraction.nii.gz')
    atlas_mask_img = sitk.GetImageFromArray(atlas_mask.astype(np.uint32))
    atlas_mask_img.CopyInformation(ra)
    sitk.WriteImage(atlas_mask_img, os.path.join(output_dir, f"{subject}_atlas.nii.gz"))

    if sr:
        atlas_mask = binary_erosion(atlas_mask > 0, iterations=3).astype(np.uint8)
        ra_mask = binary_erosion(ra_mask > 0, iterations=3).astype(np.uint8)

    subtraction = atlas_mask - ra_mask
    subtraction = subtraction == 1
    
    subtraction = subtraction - edge_bin
    subtraction = subtraction == 1

    # subtraction = binary_erosion(subtraction > 0, iterations=)

    # subtraction[-80:, :, :] = 0

    # labeled, num_features = nd_label(subtraction)
    # if num_features == 0:
    #     return np.zeros_like(subtraction, dtype=np.uint8)

    # counts = np.bincount(labeled.ravel())
    # counts[0] = 0  # Ignore background

    labeled, num_features = nd_label(subtraction)
    if num_features == 0:
        return np.zeros_like(subtraction, dtype=np.uint8)

    # Merge nearby erosions (tune max_gap_mm)
    # Arrays are z,y,x while SimpleITK spacing is x,y,z.
    spacing_zyx = tuple(reversed(native_spacing))
    labeled = merge_close_erosions(labeled, spacing=spacing_zyx, max_gap_mm=0.4)

    # Now recompute counts / valid labels after merging
    counts = np.bincount(labeled.ravel())
    counts[0] = 0

    # Get all erosion labels ≥ 32678 voxels
    print(counts)
    valid_labels = np.where(counts >= 1024)[0]
    print(valid_labels)
    print(f"Found {len(valid_labels)} erosion(s) ≥ 1024 voxels")

    subtraction_img = sitk.GetImageFromArray(labeled.astype(np.uint32))
    subtraction_img.CopyInformation(ra)
    sitk.WriteImage(subtraction_img, os.path.join(output_dir, f"{subject}_subtract.nii.gz"))

    # Prepare final labeled volume
    final_labeled_erosions = np.zeros_like(subtraction, dtype=np.uint16)

    all_metrics = []
    for i, label_id in enumerate(valid_labels, start=1):
        print(f"\n--- Processing erosion {i} (label {label_id}) ---")
        erosion = (labeled == label_id).astype(np.uint8)

        if sr:
            dilation_kernel = 6
        else:
            dilation_kernel = 2

        # Slight dilation
        erosion = binary_dilation(erosion, iterations=dilation_kernel)

        # Limit by outer cortex proximity
        if not sr:
            erosion_out_bound = erosion & atlas_outer
            erosion_out_bound = binary_dilation(erosion_out_bound, iterations=dilation_kernel)

            dist_map_np = distance_transform_edt(erosion_out_bound==0)
            erosion_mask_limits = dist_map_np < 50
            erosion = erosion & erosion_mask_limits

        # Optional snapping
        # if not sr:
        erosion = snap_outer_shell_to_edge_kdtree(erosion, edge_np)
        erosion = binary_dilation(erosion, iterations=1)

        # Final cleanup
        erosion = erosion & atlas_mask
        erosion = binary_fill_holes(erosion)

        # erosion = run_gac_levelset(edge, erosion, atlas_mask)

        flag=""

        # Print metrics for this erosion (no size filtering — keep all erosions)
        metrics = compute_erosion_metrics(
            erosion, ra, f"{subject}_erosion{i}", spacing=spacing_zyx)
        if metrics is None:
            continue
        for k, v in metrics.items():
            print(f"{k}: {v}")

        # Assign this erosion a unique label in the final volume
        final_labeled_erosions[erosion > 0] = i

        # ---- 3-channel patch and label extraction ----
        # Create bounding box around erosion
        coords = np.argwhere(erosion)
        zmin, ymin, xmin = coords.min(axis=0)
        zmax, ymax, xmax = coords.max(axis=0)

        # Add margin
        margin = 10
        zmin = max(zmin - margin, 0)
        ymin = max(ymin - margin, 0)
        xmin = max(xmin - margin, 0)
        zmax = min(zmax + margin, erosion.shape[0] - 1)
        ymax = min(ymax + margin, erosion.shape[1] - 1)
        xmax = min(xmax + margin, erosion.shape[2] - 1)

        # Extract patches for multi-channel input
        patch_ra = ra_np[zmin:zmax+1, ymin:ymax+1, xmin:xmax+1]
        patch_edge = edge_np[zmin:zmax+1, ymin:ymax+1, xmin:xmax+1]
        patch_atlas = atlas_np[zmin:zmax+1, ymin:ymax+1, xmin:xmax+1]
        patch_label = erosion[zmin:zmax+1, ymin:ymax+1, xmin:xmax+1]

        # Physical origin of the crop corner (index order is x, y, z for SimpleITK)
        # so the patch keeps its true position in the full image and can be
        # resampled back to full size later (do NOT use ra.GetOrigin() here).
        crop_origin = ra.TransformIndexToPhysicalPoint((int(xmin), int(ymin), int(zmin)))

        # Stack channels and convert to SimpleITK
        patch_input = np.stack([patch_ra, patch_edge, patch_atlas], axis=-1)
        patch_input_img = sitk.GetImageFromArray(patch_input)
        patch_input_img.SetOrigin(crop_origin)
        patch_input_img.SetSpacing(ra.GetSpacing())
        patch_input_img.SetDirection(ra.GetDirection())

        patch_label_img = sitk.GetImageFromArray(patch_label.astype(np.uint8))
        patch_label_img.SetOrigin(crop_origin)
        patch_label_img.SetSpacing(ra.GetSpacing())
        patch_label_img.SetDirection(ra.GetDirection())

        sitk.WriteImage(patch_input_img, os.path.join(output_dir, f"{subject}_erosion{i}_input{flag}.nii.gz"))
        sitk.WriteImage(patch_label_img, os.path.join(output_dir, f"{subject}_erosion{i}_label{flag}.nii.gz"))

        all_metrics.append(metrics)

    # Save final labeled erosion mask
    final_mask_img = sitk.GetImageFromArray(final_labeled_erosions)
    final_mask_img.CopyInformation(ra)
    sitk.WriteImage(final_mask_img, os.path.join(output_dir, subject + "_labeled.nii.gz"))
    print(f"Combined labeled erosion mask saved to: {subject}_labeled.nii.gz")

    if not all_metrics:
        print(f"No erosions found for {subject}; skipping metrics CSV.")
        return

    csv_path = os.path.join(output_dir, f"{subject}.csv")

    # Write to CSV
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
        writer.writeheader()
        for row in all_metrics:
            writer.writerow(row)

    print(f"Erosion metrics saved to: {csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Segment erosions by comparing RA image with atlas and edge map")
    parser.add_argument("--atlas", required=True, help="Path to deformed atlas image (NIfTI)")
    parser.add_argument("--ra", required=True, help="Path to RA image (NIfTI)")
    parser.add_argument("--edge", required=True, help="Path to edge segmentation (NIfTI)")
    parser.add_argument("--ra_mask", required=True, help="Path to edge mask closed (NIfTI)")
    parser.add_argument("--sr", action='store_true')
    parser.add_argument("--output_dir", required=True)

    args = parser.parse_args()
    segment_erosions(args.atlas, args.ra, args.edge, args.ra_mask, args.sr, args.output_dir)

if __name__ == "__main__":
    main()
