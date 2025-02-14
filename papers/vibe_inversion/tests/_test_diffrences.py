import numpy as np
from TPTBox import NII, to_nii

recon = to_nii("/media/data/NAKO/dataset-nako/derivatives_inversion/100/100073/mevibe/sub-100073_sequ-59_acq-ax_part-water_desc-reconstructed_mevibe.nii.gz").set_dtype_(
    float
)
org = to_nii("/media/data/NAKO/dataset-nako/rawdata/100/100073/mevibe/sub-100073_sequ-59_acq-ax_part-water_mevibe.nii.gz").set_dtype_(float)

(recon - org).save("/media/data/NAKO/dataset-nako/derivatives_inversion/100/100073/mevibe/diff.nii.gz")
dif = recon - org
print(dif.mean(), np.std(dif))


def compute_patch_averages(nii: NII, n):
    array = nii.get_array()
    # Ensure the array dimensions are divisible by n
    d, h, w = array.shape
    if d % n != 0 or h % n != 0 or w % n != 0:
        raise ValueError("Array dimensions must be divisible by patch size n.")

    # Reshape the array to separate patches
    reshaped = array.reshape(d // n, n, h // n, n, w // n, n)
    # Compute the average over each patch
    patch_averages = reshaped.mean(axis=(1, 3, 5))
    return nii.rescale(i * n for i in nii.zoom).set_array_(patch_averages)


dif = compute_patch_averages(recon - org, 4)
dif.save("/media/data/NAKO/dataset-nako/derivatives_inversion/100/100073/mevibe/diff2.nii.gz")


recon = to_nii("/media/data/NAKO/dataset-nako/derivatives_inversion/100/100073/mevibe/sub-100073_sequ-59_acq-ax_part-fat_desc-reconstructed_mevibe.nii.gz").set_dtype_(
    float
)
org = to_nii("/media/data/NAKO/dataset-nako/rawdata/100/100073/mevibe/sub-100073_sequ-60_acq-ax_part-fat_mevibe.nii.gz").set_dtype_(float)

(recon - org).save("/media/data/NAKO/dataset-nako/derivatives_inversion/100/100073/mevibe/diff3.nii.gz")
dif = recon - org
print(dif.mean(), np.std(dif))


def compute_patch_averages(nii: NII, n):
    array = nii.get_array()
    # Ensure the array dimensions are divisible by n
    d, h, w = array.shape
    if d % n != 0 or h % n != 0 or w % n != 0:
        raise ValueError("Array dimensions must be divisible by patch size n.")

    # Reshape the array to separate patches
    reshaped = array.reshape(d // n, n, h // n, n, w // n, n)
    # Compute the average over each patch
    patch_averages = reshaped.mean(axis=(1, 3, 5))
    return nii.rescale(i * n for i in nii.zoom).set_array_(patch_averages)


dif = compute_patch_averages(recon - org, 4)
dif.save("/media/data/NAKO/dataset-nako/derivatives_inversion/100/100073/mevibe/diff4.nii.gz")
