#import matplotlib
##matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def show_slices(slices, epoch):
    """ Function to display row of image slices """
    plt.close()
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")
    plt.suptitle(f"Selected Centre Slices of NERF-based brain after {epoch}.")
    return fig

def show_slices_gt(slices, gt_slices, epoch):
    """ Function to display row of image slices """
    plt.close()
    fig, axes = plt.subplots(2, len(slices), dpi=150)

    for i, slice in enumerate(slices):
        axes[0][i].imshow(slice.T, cmap="gray", origin="lower")
    for i, slice in enumerate(gt_slices):
        axes[1][i].imshow(slice.T, cmap="gray", origin="lower")
    plt.suptitle(f"Selected Centre Slices of NERF-based brain after {epoch}.")
    plt.tight_layout()
    return fig
