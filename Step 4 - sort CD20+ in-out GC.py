# ---------------- Core Python Libraries ---------------- #
import os
import re
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from numba import njit

# ---------------- Visualization ---------------- #
import matplotlib
# matplotlib.use('Agg')  # Uncomment if running without GUI backend (e.g., on a server)
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import Normalize
from matplotlib.colors import ListedColormap

# ---------------- Scikit-Image ---------------- #
from skimage import (
    io,
    exposure,
    filters,
    morphology,
    segmentation,
    measure,
    feature,
    img_as_ubyte,
    data
)
from skimage.io import imread
from skimage.exposure import rescale_intensity, equalize_hist
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import (
    binary_erosion, remove_small_objects, disk, white_tophat, black_tophat, ball
)
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.measure import mesh_surface_area

# ---------------- TIFF & Image I/O ---------------- #
import tifffile as tiff
from tifffile import imread

# ---------------- SciPy ---------------- #
import scipy.io as spio
import scipy.ndimage as ndi
from scipy.ndimage import zoom
from scipy.stats import gaussian_kde
from scipy.spatial import KDTree
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans



# ---------------- ML & Dimensionality Reduction ---------------- #
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.datasets import fetch_openml
import sklearn.cluster as cluster
import umap
import hdbscan

# ---------------- Image Tools: pyclesperanto ---------------- #
import pyclesperanto_prototype as cle
from pyclesperanto_prototype import imshow

# ---------------- Stardist (Optional) ---------------- #
# from stardist import fill_label_holes, relabel_image_stardist, random_label_cmap
# from stardist.matching import matching_dataset

# ---------------- CSBDeep Utils ---------------- #
from csbdeep.utils import Path, download_and_extract_zip_file

# ---------------- unwrap3D Library ---------------- #
import unwrap3D.Visualisation.colors as vol_colors
import unwrap3D.Segmentation.segmentation as segmentation
import unwrap3D.Utility_Functions.file_io as fio
import unwrap3D.Image_Functions.image as image_fn
import unwrap3D.Mesh.meshtools as meshtools
import unwrap3D.Parameters.params as params
import unwrap3D.Geometry.geometry as geometry
import unwrap3D.Unzipping.unzip as uzip
import unwrap3D.Registration.registration as registration
import unwrap3D.Analysis_Functions.topography as topo_tools

# ---------------- Excel ---------------- #
import openpyxl as px

import gc
import tracemalloc


def remove_large_objects(ar, min_size=64, connectivity=1, *, out=None):
    if out is None:
            out = ar.copy()
    else:
        out[:] = ar

    if min_size == 0:  # shortcut for efficiency
        return out

    if out.dtype == bool:
        footprint = ndi.generate_binary_structure(ar.ndim, connectivity)
        ccs = np.zeros_like(ar, dtype=np.int32)
        ndi.label(ar, footprint, output=ccs)
    else:
        ccs = out

    try:
        component_sizes = np.bincount(ccs.ravel())
    except ValueError:
        raise ValueError("Negative value labels are not supported. Try "
                        "relabeling the input with `scipy.ndimage.label` or "
                        "`skimage.morphology.label`.")

    if len(component_sizes) == 2 and out.dtype != bool:
        warn("Only one label was provided to `remove_small_objects`. "
             "Did you mean to use a boolean array?")

    too_large = component_sizes > min_size
    too_large_mask = too_large[ccs]
    out[too_large_mask] = 0

    return out

############# Import neccessary libraries


def subtract_background_3d(stack, radius=5, light_bg=False):
    # Create a 3D structuring element (ball shape)
    str_el = ball(radius)
    
    # Ensure the structuring element is a 3D array
    if len(str_el.shape) != 3:
        raise ValueError("Structuring element must be a 3D array.")
    
    if light_bg:
        result = black_tophat(stack, str_el)
    else:
        result = white_tophat(stack, str_el)
    
    return result

def subtract_background_2d(stack, radius=5, light_bg=False):
    # Create a 3D structuring element (ball shape)
    str_el = disk(radius)
    
    # Ensure the structuring element is a 3D array
    if len(str_el.shape) != 2:
        raise ValueError("Structuring element must be a 2D array.")
    
    if light_bg:
        result = black_tophat(stack, str_el)
    else:
        result = white_tophat(stack, str_el)
    
    return result

def generate_random_colors(num_colors):
    return np.random.randint(0, 65536, size=(num_colors, 3), dtype=np.uint16)

def label_to_color(segmented, label_colors):
    colored_image = np.zeros(segmented.shape + (3,), dtype=np.uint16)

    unique_labels = np.unique(segmented)
    for i, label in enumerate(unique_labels):
        mask = segmented == label
        colored_image[mask] = label_colors[i]

    return colored_image

            
def random_colormap(n_labels, seed=None):
    if seed is not None:
        np.random.seed(seed)
    colors = np.random.rand(n_labels, 3)  # Random RGB colors
    colors[0] = [0, 0, 0]  # Ensure the background (label 0) is black
    return ListedColormap(colors)

def extract_tiles(image, tile_size):
    """
    Split an image into smaller tiles.
    
    Parameters:
    - image: np.ndarray, the input image.
    - tile_size: tuple, the height and width of each tile (e.g., (512, 512)).
    
    Returns:
    - tiles: list of np.ndarray, the tiles extracted from the image.
    """
    tiles = []
    img_height, img_width = image.shape[:2]
    tile_height, tile_width = tile_size
    
    for y in range(0, img_height, tile_height):
        for x in range(0, img_width, tile_width):
            tile = image[y:y + tile_height, x:x + tile_width]
            tiles.append(tile)
    
    return tiles


def stitch_tiles(tiles, image_shape, tile_size):
    """
    Combine tiles back into a single image.
    
    Parameters:
    - tiles: list of np.ndarray, the tiles to combine.
    - image_shape: tuple, the shape of the original image (height, width).
    - tile_size: tuple, the height and width of each tile (e.g., (512, 512)).
    
    Returns:
    - stitched_image: np.ndarray, the reassembled image.
    """
    img_height, img_width = image_shape
    tile_height, tile_width = tile_size
    stitched_image = np.zeros((img_height, img_width), dtype=tiles[0].dtype)
    
    i = 0
    for y in range(0, img_height, tile_height):
        for x in range(0, img_width, tile_width):
            stitched_image[y:y + tile_height, x:x + tile_width] = tiles[i]
            i += 1
    
    return stitched_image

def mean_contour_intensity(regionmask, intensity_image, radius=1):

    dilated_mask = skmorph.dilation(regionmask, skmorph.disk(radius))
    contour_mask = dilated_mask & ~regionmask
    return intensity_image[contour_mask].mean()


# def compute_distances_with_overlap(source_df, target_df, source_channel, target_channel):
#     # Extract and scale centroid coordinates
#     source_coords = source_df[['centroid-1', 'centroid-0']].values 
#     target_coords = target_df[['centroid-1', 'centroid-0']].values 

#     # Compute pairwise Euclidean distances
#     distances = cdist(source_coords, target_coords, metric='euclidean')
#     distances = distances*pixel_x*pixel_y

#     # Handle overlapping markers (distance = 0 if cells share the same label)
#     if not source_df.empty and not target_df.empty:
#         source_labels = source_df['label'].values
#         target_labels = target_df['label'].values

#         for i, src_label in enumerate(source_labels):
#             for j, tgt_label in enumerate(target_labels):
#                 if src_label == tgt_label:  # Same label implies overlap
#                     distances[i, j] = 0

#     return distances

def cross_g_function(source_positions, target_positions, radii):
    # Build KDTree for the target
    target_tree = KDTree(target_positions)
    
    # Calculate nearest neighbor distances for source to target
    distances, _ = target_tree.query(source_positions)
    
    # Calculate G(r) for each radius
    g_values = []
    for r in radii:
        g_values.append((distances <= r).sum() / len(source_positions))
    
    return g_values


# def cross_g_function(source_positions, target_positions, radii, area):
#     """
#     Compute the cross G function using the theoretical formula.

#     Parameters:
#     - source_positions: Coordinates of source points (array of shape [n, 2]).
#     - target_positions: Coordinates of target points (array of shape [m, 2]).
#     - radii: List of radii to evaluate G(r).
#     - area: Total area of the observation window.

#     Returns:
#     - g_values: Computed G(r) values for each radius.
#     """

#     lambda_y = len(target_positions) / area

#     g_values = [1 - np.exp(-lambda_y * np.pi * r**2) for r in radii]
    
#     return g_values


def theoretical_cross_g(radii, lambda_B):
    radii = np.array(radii)  # Ensure radii is a NumPy array
    return 1 - np.exp(-lambda_B * np.pi * radii**2)

#### see https://www.geo.fu-berlin.de/en/v/soga-r/Advances-statistics/Spatial-Point-Patterns/Analysis-of-Spatial-Point-Patterns/Interactions-in-Point-Pattern-Analysis/index.html 


def monte_carlo_simulation(source_positions, target_positions, radii, area, num_simulations=100):
    """Perform Monte Carlo simulation for CSR to generate confidence envelopes."""
    lambda_B = len(target_positions) / area
    simulated_g_values = []
    
    for _ in range(num_simulations):
        # Randomize target positions
        randomized_target_positions = np.random.rand(len(target_positions), 2) * np.sqrt(area)
        
        # Cross-G Function for randomization
        g_values = cross_g_function(source_positions, randomized_target_positions, radii)
        simulated_g_values.append(g_values)
        
    
    # Calculate envelopes (5th and 95th percentiles)
    lower_g = np.percentile(simulated_g_values, 5, axis=0)
    upper_g = np.percentile(simulated_g_values, 95, axis=0)
    
    return lower_g, upper_g




# def k_function(positions, radii, area):
#     # Build KDTree for positions
#     tree = KDTree(positions)
    
#     # Calculate K(r) for each radius
#     k_values = []
#     for r in radii:
#         count = (tree.query_ball_point(positions, r, return_length=True)).sum()
#         k_values.append(count / len(positions) / area)
    
#     return k_values








pixel_x = 0.325  # µm per pixel
pixel_y = 0.325  # µm per pixel


###### Load and produce dilation of segmented nuclei for measuring signal around edge vs. centre to identify marker associated to a given nuclei #####################

all_folders = 'C:/Users/edwardj/Desktop/Data to analyse for others/Jo'
segment_and_filtred = os.path.join(all_folders,'segmented and filtered')

all_data_path = os.path.join(segment_and_filtred, 'Updated_Centroids_Intensity_with_Positive_Labels.xlsx')
all_data_df = pd.read_excel(all_data_path)
nuclei_folder = os.path.join(all_folders,'Channel_2')
segmented_nuclei_folder = os.path.join(nuclei_folder,'segmentation')
segmented_nuclei = os.path.join(segmented_nuclei_folder, 'segmented_dilation_stardist.tif')
segmented_nuclei_image = skio.imread(segmented_nuclei)
# gc_image = os.path.join(segment_and_filtred,'filtered_labels_CD20.tif')
# gc_image skio.imread(gc_image)


# Step 2: Filter centroids where CD20_positive = 1 and others are 0
filtered_data = all_data_df[
    (all_data_df['CD20_positive'] == 1) 
    & (all_data_df[['CD3_positive', 'CD56_positive', 'CD68_positive', 'tryptase_positive']].sum(axis=1) == 0)
]
cd20_centroids = filtered_data[['centroid-1', 'centroid-0']].to_numpy()


dist_matrix = cdist(filtered_data, filtered_data)

dist_matrix = distance_matrix(cd20_centroids, cd20_centroids, p=2)
print (dist_matrix.shape, dist_matrix)
dist_matrix_um = dist_matrix * (pixel_x * pixel_y)
print(dist_matrix_um.shape, dist_matrix_um)

thresholds = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44]
surrounding_cells_counts = []

for threshold in thresholds:
    # Count the number of CD20-positive cells within the distance threshold for each centroid
    count_cells = np.sum(dist_matrix_um <= threshold, axis=1) - 1  # Exclude the centroid itself (self-count)
    surrounding_cells_counts.append(count_cells)

fig, axes = plt.subplots(3, 4, figsize=(20, 18))  # Adjust the figure size for better spacing
axes = axes.flatten()  # Flatten the 2D array to make indexing easier

# Plot each threshold
for i, threshold in enumerate(thresholds):
    ax = axes[i]
    
    # Color the centroids based on the number of surrounding cells within each threshold
    scatter = ax.scatter(
        cd20_centroids[:, 0], cd20_centroids[:, 1], 
        c=surrounding_cells_counts[i], cmap='Spectral_r', s=0.5, alpha=0.6
    )
    ax.invert_yaxis()
    ax.set_title(f'CD20+ Centroids with {threshold} µm Threshold', fontsize=10)
    ax.set_xlabel('X Coordinate', fontsize=8)
    ax.set_ylabel('Y Coordinate', fontsize=8)
    fig.colorbar(scatter, ax=ax, orientation='vertical', label='Number of Surrounding CD20+ Cells')

# Adjust spacing between subplots
plt.subplots_adjust(hspace=0.4, wspace=0.4)  # Increase space between rows and columns

# Use tight_layout to ensure labels don't overlap
plt.tight_layout(pad=2.0)  # Increase padding around the plots to prevent overlapping titles/labels

# figure_path = os.path.join(segment_and_filtred, 'CD20_Centroids_distances.png')
# plt.savefig(figure_path, dpi=1200)  # Save with high resolution (300 DPI)
# print(f"Figure saved at: {figure_path}")



threshold = 10  # 10 µm threshold - this is going to count thenumber of cells within 10um radius for each cell
count_cells_10um = np.sum(dist_matrix_um <= threshold, axis=1) - 1  # Exclude the centroid itself
print ('minimum count cells')
print (np.min(count_cells_10um))


# Step 5: Filter out centroids that have more than 50 surrounding cells
remaining_centroids = cd20_centroids[count_cells_10um <= 15] ## this now going to remove any cells which have more than 20 cells in 10um diameter

# Step 6: Plot the remaining centroids with color indicating number of surrounding cells
fig, ax = plt.subplots(figsize=(10, 8))

# Color the centroids based on the number of surrounding cells within 10 µm threshold
scatter = ax.scatter(
    remaining_centroids[:, 0], remaining_centroids[:, 1], 
    c=count_cells_10um[count_cells_10um <= 15], cmap='Spectral_r', s=0.5, alpha=0.6
)
ax.invert_yaxis()
ax.set_title('Remaining CD20+ Centroids with <= 50 Connections (10 µm Threshold)', fontsize=14)
ax.set_xlabel('X Coordinate', fontsize=12)
ax.set_ylabel('Y Coordinate', fontsize=12)
fig.colorbar(scatter, ax=ax, orientation='vertical', label='Number of Surrounding CD20+ Cells')

plt.tight_layout()
figure_path = os.path.join(segment_and_filtred, 'CD20_Centroids_distances_thresholded.png')
plt.savefig(figure_path, dpi=1200)  # Save with high resolution (300 DPI)



# Step 5: Determine GC status
gc_yes_no_filtered = np.where(count_cells_10um > 15, 1, 0)  # 1 for inside GC, 0 for outside GC

# Step 6: Add default GC_yes_no column to the original DataFrame
all_data_df['GC_yes_no'] = -1  # Default value for all rows

# Step 7: Update GC_yes_no for CD20-positive rows
filtered_data['GC_yes_no'] = gc_yes_no_filtered  # Assign 0 or 1 to filtered rows
all_data_df.update(filtered_data[['centroid-0', 'centroid-1', 'GC_yes_no']])  # Update matching rows in original DataFrame

# Step 8: Save the updated Excel file
updated_path = os.path.join(segment_and_filtred, 'Updated_Centroids_Intensity_with_Positive_Labels_GC.xlsx')
all_data_df.to_excel(updated_path, index=False)

print(f"Updated file saved at {updated_path}")

updated_data_path = os.path.join(segment_and_filtred, 'Updated_Centroids_Intensity_with_Positive_Labels_GC.xlsx')
updated_df = pd.read_excel(updated_data_path)

# outside_gc_labels = updated_df[updated_df['GC_yes_no'] == 0]['label'].unique()

# # Filter the segmented nuclei image: keep only labels present in `outside_gc_labels`
# filtered_image = np.where(np.isin(segmented_nuclei_image, outside_gc_labels), segmented_nuclei_image, 0)

# # Save the filtered image
# filtered_image_path = os.path.join(segment_and_filtred, 'filtered_centroids_CD20_outside_GC.tif')
# skio.imsave(filtered_image_path, filtered_image.astype(np.uint32))



inside_gc_labels = updated_df[updated_df['GC_yes_no'] == 1]['label'].unique()

# Step 2: Create mask for centroids outside GC
outside_gc_labels = updated_df[updated_df['GC_yes_no'] == 0]['label'].unique()

# Step 3: Filter the segmented nuclei image to create images based on GC status
# For centroids outside GC
outside_gc_image = np.where(np.isin(segmented_nuclei_image, outside_gc_labels), segmented_nuclei_image, 0)

# For centroids inside GC
inside_gc_image = np.where(np.isin(segmented_nuclei_image, inside_gc_labels), segmented_nuclei_image, 0)

# Step 4: Save the filtered images
outside_gc_filtered_path = os.path.join(segment_and_filtred, 'filtered_centroids_CD20_outside_GC.tif')
inside_gc_filtered_path = os.path.join(segment_and_filtred, 'filtered_centroids_CD20_inside_GC.tif')

# Save the images
skio.imsave(outside_gc_filtered_path, outside_gc_image.astype(np.uint32))
skio.imsave(inside_gc_filtered_path, inside_gc_image.astype(np.uint32))

# Also save a combined image for both inside and outside GC
combined_gc_image = np.where(np.isin(segmented_nuclei_image, outside_gc_labels) | 
                             np.isin(segmented_nuclei_image, inside_gc_labels), 
                             segmented_nuclei_image, 0)

combined_gc_image_path = os.path.join(segment_and_filtred, 'combined_centroids_CD20_GC_status.tif')
skio.imsave(combined_gc_image_path, combined_gc_image.astype(np.uint32))

print(f"Filtered centroids image (outside GC) saved at: {outside_gc_filtered_path}")
print(f"Filtered centroids image (inside GC) saved at: {inside_gc_filtered_path}")
print(f"Combined GC centroids image saved at: {combined_gc_image_path}")


# Step 1: Create a blank image for visualization
image_shape = segmented_nuclei_image.shape  # Get dimensions from original image
visualization = np.zeros(image_shape, dtype=np.uint8)

# Step 2: Generate visualization layers
original_layer = np.zeros_like(visualization)
outside_gc_layer = np.zeros_like(visualization)

# Original CD20-positive centroids: White (255)
original_layer[np.isin(segmented_nuclei_image, filtered_data['label'])] = 255

# Thresholded CD20+ cells outside GC: Red (255 in the red channel)
outside_gc_layer[np.isin(segmented_nuclei_image, outside_gc_labels)] = 255

# Step 3: Combined visualization with black background
# Combined visualization will have:
# - Red for cells outside GC (Red channel = 255)
# - White for CD20+ centroids in GC (Blue channel = 255)
combined_visualization = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)

# Set the red channel for outside GC cells
combined_visualization[outside_gc_layer == 255, 0] = 255  # Red channel

# Set the blue channel for CD20+ centroids (inside GC)
combined_visualization[original_layer == 255, 2] = 255  # Blue channel

# Step 4: Plot the three panels

fig, axes = plt.subplots(1, 3, figsize=(24, 8))

# Original CD20+ Centroids: White on black background
axes[0].imshow(original_layer, cmap='gray')
axes[0].set_title('Original CD20+ Centroids', fontsize=14)
axes[0].axis('off')

# CD20+ Centroids Outside GC: Red on black background
axes[1].imshow(outside_gc_layer, cmap='Reds')
axes[1].set_title('Thresholded Outside GC', fontsize=14)
axes[1].axis('off')

# Combined View: White for CD20 in GC, Red for Outside GC, Black background
axes[2].imshow(combined_visualization)
axes[2].set_title('Combined View (GC in White, Outside GC in Red)', fontsize=14)
axes[2].axis('off')

# Save the PNG
png_path = os.path.join(segment_and_filtred, 'CD20_Centroids_Comparison.png')
plt.tight_layout()
plt.savefig(png_path, dpi=300)
print(f"Comparison PNG saved at: {png_path}")
