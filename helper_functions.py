import compute_triplets
import compute_baseline
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import TwoSlopeNorm
import matplotlib.patches as patches

def plot_heatmaps(hist_3d, title="3D Difference Histogram (True - Background)"):
    """
    Plots 2D X/Y heatmaps for each Z slice of a 3D histogram.
    
    Parameters:
    - hist_3d: 3D numpy array of shape (X, Y, Z)
    """
    num_x, num_y, num_z = hist_3d.shape
    c_x, c_y, c_z = num_x // 2, num_y // 2, num_z // 2
    
    # 1. Setup 3xN layout
    num_rows = 3
    num_cols = int(np.ceil(num_z / num_rows))
    
    # 2. Define custom colormap: Red (negative) -> White (zero) -> Green (positive)
    colors = ["#d73027", "#ffffff", "#1a9850"]
    rwg_cmap = LinearSegmentedColormap.from_list("RedWhiteGreen", colors)
    
    # 3. Center colormap strictly at zero
    max_abs_val = np.max(np.abs(hist_3d))
    if max_abs_val == 0:
        max_abs_val = 1.0
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-max_abs_val, vmax=max_abs_val)
    
    # 4. Initialize figure grid
    fig, axes = plt.subplots(
        num_rows, num_cols, 
        figsize=(2.8 * num_cols, 3.2 * num_rows), 
        squeeze=False
    )
    
    im = None
    # 5. Populate heatmaps
    for z in range(num_z):
        row = z // num_cols
        col = z % num_cols
        ax = axes[row, col]
        
        slice_xy = hist_3d[:, :, z]
        
        im = ax.imshow(
            slice_xy, 
            cmap=rwg_cmap, 
            norm=norm, 
            origin='lower',
            interpolation='nearest'
        )
        
        # 6. Draw bounding box around (c_x, c_y) pixel center
        # imshow pixel boundaries span [index - 0.5, index + 0.5]
        is_center_z = (z == c_z)
        box_color = '#d90429' if is_center_z else 'black'  # Highlight true 3D center in red/magenta
        line_w = 2.0 if is_center_z else 1.2
        
        rect = patches.Rectangle(
            (c_x - 0.5, c_y - 0.5), 1, 1, 
            linewidth=line_w, 
            edgecolor=box_color, 
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # Title formatting
        title_text = f"Z = {z} (Center)" if is_center_z else f"Z = {z}"
        ax.set_title(title_text, fontsize=9, pad=3, fontweight='bold' if is_center_z else 'normal')
        ax.set_xlabel("X", fontsize=7, labelpad=1)
        ax.set_ylabel("Y", fontsize=7, labelpad=1)
        ax.tick_params(labelsize=6)

    # Turn off unused axes
    for z in range(num_z, num_rows * num_cols):
        row = z // num_cols
        col = z % num_cols
        axes[row, col].axis('off')

    # 7. Horizontal Colorbar
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.14, top=0.92)
    
    cbar_ax = fig.add_axes([0.20, 0.05, 0.60, 0.025])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label("Net Counts (True-Baseline)", fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    
    plt.suptitle(title, fontsize=13, fontweight='bold', y=0.98)
    # Save the figure
    plt.savefig('heatmap_plot.png', dpi=300, bbox_inches='tight')
    plt.show()


def compare_experiments(data, bins_per_dim):
    # 1. Compute raw counts per experiment (Shape: E, B, B, B)
    # 1.1 Total 3-body contribution
    rvalue = compute_triplets.compute_triplets_numba(data, bins_per_dim)
    true_samples = rvalue['data']
    # Normalize histogram
    true_samples = true_samples/true_samples.sum(axis=(1, 2, 3))[:, None, None, None]
    
    # 1.2 Lower order contributions
    norm_factor = 0.001
    # 1.2.a 2+1 body contribution
    cross_samples = compute_baseline.compute_triplets_numba_cross(
        data, bins_per_dim, rvalue['power'], rvalue['max_mod'], norm_factor_cross=norm_factor
    )['data']
    # Normalize histogram
    cross_samples = cross_samples/cross_samples.sum(axis=(1, 2, 3))[:, None, None, None]
    # 1.2.b 1+1+1 body contribution
    norm_samples = compute_baseline.compute_triplets_numba_norm(
        data, bins_per_dim, rvalue['power'], rvalue['max_mod'], norm_factor_norm=norm_factor
    )['data']
    # Normalize histogram
    norm_samples = norm_samples/norm_samples.sum(axis=(1, 2, 3))[:, None, None, None]
    
    # 2. Raw combined background array (Shape: E, B, B, B)
    raw_base_samples = 3*cross_samples - 2*norm_samples

    # 3. Extract center zero bin per experiment
    c = bins_per_dim // 2
    true_zero = true_samples[:, c, c, c]
    base_zero = raw_base_samples[:, c, c, c]

    # 4. Global reporting across full dataset
    hist_true = true_samples.mean(axis=0)
    hist_base = raw_base_samples.mean(axis=0)

    print(f"True Zero Bin Total: {hist_true[c,c,c]}")
    print(f"Scaled Background Zero Bin Total: {hist_base[c,c,c]:.2f}")
    print(f"Mean Net Signal Zero Bin Total: {hist_true[c,c,c] - hist_base[c,c,c]:.2f}")
    
    plot_heatmaps(hist_true - hist_base)
    return hist_true - hist_base, true_zero, base_zero