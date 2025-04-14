import matplotlib
# matplotlib.use('Agg')
import numpy as np  
import pylab as plt 
import skimage.io as skio 
# from unets import att_unet
# from keras.optimizers import Adam
from skimage.exposure import rescale_intensity
from skimage.filters import threshold_otsu
from skimage.filters import gaussian
# from dipy.denoise.patch2self import patch2self
from skimage.filters import gaussian 
import glob
import os 
from scipy.ndimage import zoom
import scipy.io as spio 
from tqdm import tqdm 
from skimage.exposure import equalize_hist
import scipy.ndimage as ndimage
import skimage.exposure as skexposure
import skimage.filters as skfilters
import skimage.morphology as skmorph
import skimage.segmentation as sksegmentation 
from skimage.measure import mesh_surface_area
import tifffile as tiff
import unwrap3D.Visualisation.colors as vol_colors
import unwrap3D.Segmentation.segmentation as segmentation 
import unwrap3D.Utility_Functions.file_io as fio
import unwrap3D.Image_Functions.image as image_fn
import unwrap3D.Mesh.meshtools as meshtools
import unwrap3D.Parameters.params as params
import unwrap3D.Geometry.geometry as geometry
import unwrap3D.Unzipping.unzip as uzip
import unwrap3D.Registration.registration as registration 
import pandas as pd
import igl
from matplotlib import cm
from skimage import io, img_as_ubyte
import unwrap3D.Analysis_Functions.topography as topo_tools
import skimage.filters as skfilters
import skimage.morphology as skmorph
import skimage.segmentation as sksegmentation 
import igl
import openpyxl as px
import trimesh
import skimage.measure as skmeasure 
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt

from tqdm import tqdm 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import umap

import hdbscan
import sklearn.cluster as cluster
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
import re
from scipy.stats import gaussian_kde
from mpl_toolkits.mplot3d import Axes3D

import os
import numpy as np
import skimage.io as skio
import skimage.filters as skfilters
import skimage.morphology as skmorph
import skimage.measure as skmeasure
import scipy.ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.morphology import binary_erosion, remove_small_objects, disk

from skimage import measure, io
import pandas as pd

from glob import glob
from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import Path, download_and_extract_zip_file

# from stardist import fill_label_holes, relabel_image_stardist, random_label_cmap
# from stardist.matching import matching_dataset

from skimage.io import imread
from pyclesperanto_prototype import imshow
import pyclesperanto_prototype as cle
import matplotlib.pyplot as plt
from numba import njit
from skimage.morphology import white_tophat, black_tophat, ball

import matplotlib.patches as patches
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import tracemalloc
from scipy.spatial import KDTree
from matplotlib.colors import Normalize


# Prepare data and apply histogram equalization

from skimage.data import cells3d
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

from matplotlib.colors import ListedColormap
            
def random_colormap(n_labels, seed=None):
    if seed is not None:
        np.random.seed(seed)
    colors = np.random.rand(n_labels, 3)  # Random RGB colors
    colors[0] = [0, 0, 0]  # Ensure the background (label 0) is black
    return ListedColormap(colors)


import numpy as np

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

from scipy.spatial.distance import cdist

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

import numpy as np
from scipy.spatial import KDTree

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

from scipy.spatial import distance_matrix
import gc

###### Load and produce dilation of segmented nuclei for measuring signal around edge vs. centre to identify marker associated to a given nuclei #####################

all_folders = 'C:/Users/edwardj/Desktop/Data to analyse for others/Jo'
segment_and_filtred = os.path.join(all_folders,'segmented and filtered')

all_data_path = os.path.join(segment_and_filtred, 'Updated_Centroids_Intensity_with_Positive_Labels_GC.xlsx')
all_data = pd.read_excel(all_data_path)
save_path = os.path.join(segment_and_filtred, 'probability density maps')
fio.mkdir(save_path)


# Define channel-positive columns
channels = ['CD20', 'CD3', 'CD56', 'CD68', 'tryptase']
positive_columns = [f'{channel}_positive' for channel in channels]
radii = [0, 1, 5, 10, 25, 50, 100, 200]


gc_conditions = [
    (all_data['GC_yes_no'].isin([0, -1])),  # Condition 1: GC = 0 and -1
    (all_data['GC_yes_no'].isin([1, -1])),  # Condition 2: GC = 1 and -1
    (all_data['GC_yes_no'].isin([0, 1, -1]))  # Condition 3: GC = 0, 1, and -1
]

# Add labels for each condition (optional, for clarity)
gc_labels = ['outside GC', 'inside GC', 'all']

tracemalloc.start()


# # Loop through each GC condition and label
# for gc_condition, gc_label in zip(gc_conditions, gc_labels):
#     print(f"Processing data for condition: {gc_label}")

#     # Filter the data based on the current GC condition
#     all_data_df = all_data[gc_condition].copy()
#     print(f"Data after GC filter: {all_data_df.shape}")

#     # Create a figure for all channel plots (5 channels per figure)
#     fig, axes = plt.subplots(1, 5, figsize=(20, 6))  # 1 row, 5 columns for the channels
#     fig.suptitle(f'Probability Density Maps for {gc_label}', fontsize=16)

#     for i, source_channel in enumerate(channels):  # Use enumerate to get both index and channel
#         print(f"Processing channel: {source_channel}")

#         # Filter for CD20-specific logic
#         if source_channel == 'CD20':
#         # CD20-specific filtering: CD20 == 1 and other channels == 0
#             source_data = all_data_df[
#                 (all_data_df['CD20_positive'] == 1) & 
#                 (all_data_df[[f'{channel}_positive' for channel in channels if channel != 'CD20']].sum(axis=1) == 0)
#             ].copy()
        
#         if source_channel == 'CD3':
#             # CD20-specific filtering: CD20 == 1 and other channels == 0
#             source_data = all_data_df[
#                 (all_data_df['CD3_positive'] == 1) & 
#                 (all_data_df[[f'{channel}_positive' for channel in channels if channel != 'CD3']].sum(axis=1) == 0)
#             ].copy()

#         if source_channel == 'CD56':
#             # CD20-specific filtering: CD20 == 1 and other channels == 0
#             source_data = all_data_df[
#                 (all_data_df['CD56_positive'] == 1) & 
#                 (all_data_df[[f'{channel}_positive' for channel in channels if channel != 'CD56']].sum(axis=1) == 0)
#             ].copy()

#         if source_channel == 'CD68':
#             # CD20-specific filtering: CD20 == 1 and other channels == 0
#             source_data = all_data_df[
#                 (all_data_df['CD68_positive'] == 1) & 
#                 (all_data_df[[f'{channel}_positive' for channel in channels if channel != 'CD68']].sum(axis=1) == 0)
#             ].copy()
        
#         if source_channel == 'tryptase':
#             # CD20-specific filtering: CD20 == 1 and other channels == 0
#             source_data = all_data_df[
#                 (all_data_df['tryptase_positive'] == 1) & 
#                 (all_data_df[[f'{channel}_positive' for channel in channels if channel != 'tryptase']].sum(axis=1) == 0)
#             ].copy()

#         # Extract centroid values for KDE calculation
#         centroids_x = source_data['centroid-1'].values  # x-coordinates
#         centroids_y = source_data['centroid-0'].values  # y-coordinates

#         # Calculate probability density using Gaussian KDE for each centroid
#         xy = np.vstack([centroids_x, centroids_y])  # Stack the x and y coordinates
#         kde = gaussian_kde(xy, bw_method = 0.1)  # Create the KDE object
#         density = kde(xy)  # Evaluate the density at each point

#         # Scatter plot with KDE density
#         ax = axes[i]
#         scatter = ax.scatter(centroids_x, centroids_y, c=density, cmap='Spectral_r', s=2, alpha=0.6)
#         ax.set_title(f'{source_channel} Density')
#         ax.set_xlabel('X Coordinates')
#         ax.set_ylabel('Y Coordinates')

#         # Add color bar to show density scale
#         fig.colorbar(scatter, ax=ax, orientation='vertical', label='Density')

#         # Add the density values as new columns to the dataframe
#         all_data_df[f'{source_channel}_KDE_density'] = np.nan
#         all_data_df.loc[source_data.index, f'{source_channel}_KDE_density'] = density

#     # Save the updated dataframe with KDE columns to Excel for this condition
#     excel_save_path = os.path.join(save_path, f'{gc_label}_updated_data_with_KDE.xlsx')
#     all_data_df.to_excel(excel_save_path, index=False)
#     print(f"Saved updated data with KDE columns for {gc_label} to {excel_save_path}")

#     # Adjust layout to prevent overlap
#     plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust to avoid overlap with title

#     # Save the combined figure with all channel density plots in one PNG file
#     plot_save_path = os.path.join(save_path, f'{gc_label}_density_maps.png')
#     plt.savefig(plot_save_path, dpi=1200)
#     plt.close()

#     print(f"Saved probability density map for {gc_label} to {plot_save_path}")

    




excel_save_path = os.path.join(save_path, 'all_updated_data_with_KDE.xlsx')
data = pd.read_excel(excel_save_path)
data['CD56_density_class'] = -1
data['CD20_density_class'] = -1

cd56_positive_condition = (
    (data['CD56_positive'] == 1) &  # CD56_positive = 1
    (data[['CD20_positive', 'CD3_positive', 'CD68_positive', 'tryptase_positive']].sum(axis=1) == 0)  # All other positive columns = 0
)

cd20_positive_condition = (
    (data['CD20_positive'] == 1) &
    (data['GC_yes_no'] == 0) &
    (data[['CD56_positive', 'CD3_positive', 'CD68_positive', 'tryptase_positive']].sum(axis=1) == 0)  # All other positive columns = 0
)

print (data.shape)
filtered_data_cd20 = data[cd20_positive_condition].copy()
print(filtered_data_cd20.shape)

# Extract centroid values for KDE calculation
centroids_x = filtered_data_cd20['centroid-1'].values  # x-coordinates
centroids_y = filtered_data_cd20['centroid-0'].values  # y-coordinates
print (centroids_y.shape)

# Calculate probability density using Gaussian KDE for each centroid
xy = np.vstack([centroids_x, centroids_y])  # Stack the x and y coordinates
kde = gaussian_kde(xy, bw_method = 0.1)  # Create the KDE object
density = kde(xy)  # Evaluate the density at each point

print (density.shape)



data['CD20_outside_gc_KDE_density'] = np.nan
data.loc[filtered_data_cd20.index, 'CD20_outside_gc_KDE_density'] = density


filtered_data = data[cd56_positive_condition].copy()
filtered_data_cd20 = data[cd20_positive_condition].copy()

# Calculate the 25th and 75th percentiles for CD56_KDE_density
cd56_density_20th = filtered_data['CD56_KDE_density'].quantile(0.20)
cd56_density_40th = filtered_data['CD56_KDE_density'].quantile(0.40)
cd56_density_60th = filtered_data['CD56_KDE_density'].quantile(0.60)
cd56_density_80th = filtered_data['CD56_KDE_density'].quantile(0.80)


cd20_density_10th = filtered_data_cd20['CD20_outside_gc_KDE_density'].quantile(0.10)
cd20_density_20th = filtered_data_cd20['CD20_outside_gc_KDE_density'].quantile(0.20)
cd20_density_30th = filtered_data_cd20['CD20_outside_gc_KDE_density'].quantile(0.30)
cd20_density_40th = filtered_data_cd20['CD20_outside_gc_KDE_density'].quantile(0.40)
cd20_density_50th = filtered_data_cd20['CD20_outside_gc_KDE_density'].quantile(0.50)
cd20_density_60th = filtered_data_cd20['CD20_outside_gc_KDE_density'].quantile(0.60)
cd20_density_70th = filtered_data_cd20['CD20_outside_gc_KDE_density'].quantile(0.70)
cd20_density_80th = filtered_data_cd20['CD20_outside_gc_KDE_density'].quantile(0.80)
cd20_density_90th = filtered_data_cd20['CD20_outside_gc_KDE_density'].quantile(0.90)

# Assign new values based on density thresholds within the filtered data
filtered_data.loc[filtered_data['CD56_KDE_density'] > cd56_density_80th, 'CD56_density_class'] = 4  # Top 25%

filtered_data.loc[(filtered_data['CD56_KDE_density'] >= cd56_density_60th) & 
                  (filtered_data['CD56_KDE_density'] <= cd56_density_80th), 'CD56_density_class'] = 3  # Middle %

filtered_data.loc[(filtered_data['CD56_KDE_density'] >= cd56_density_40th) & 
                  (filtered_data['CD56_KDE_density'] <= cd56_density_60th), 'CD56_density_class'] = 2  # Middle %

filtered_data.loc[(filtered_data['CD56_KDE_density'] >= cd56_density_20th) & 
                  (filtered_data['CD56_KDE_density'] <= cd56_density_40th), 'CD56_density_class'] = 1  # Middle %

filtered_data.loc[filtered_data['CD56_KDE_density'] < cd56_density_20th, 'CD56_density_class'] = 0  # Bottom 25%

# Update the original data's `CD56_density_class` column
data.loc[filtered_data.index, 'CD56_density_class'] = filtered_data['CD56_density_class']


filtered_data_cd20.loc[filtered_data_cd20['CD20_outside_gc_KDE_density'] > cd20_density_90th, 'CD20_density_class'] = 9
filtered_data_cd20.loc[(filtered_data_cd20['CD20_outside_gc_KDE_density'] >= cd20_density_80th) & 
                  (filtered_data_cd20['CD20_outside_gc_KDE_density'] <= cd20_density_90th), 'CD20_density_class'] = 8
filtered_data_cd20.loc[(filtered_data_cd20['CD20_outside_gc_KDE_density'] >= cd20_density_70th) & 
                  (filtered_data_cd20['CD20_outside_gc_KDE_density'] <= cd20_density_80th), 'CD20_density_class'] = 7
filtered_data_cd20.loc[(filtered_data_cd20['CD20_outside_gc_KDE_density'] >= cd20_density_60th) & 
                  (filtered_data_cd20['CD20_outside_gc_KDE_density'] <= cd20_density_70th), 'CD20_density_class'] = 6
filtered_data_cd20.loc[(filtered_data_cd20['CD20_outside_gc_KDE_density'] >= cd20_density_50th) & 
                  (filtered_data_cd20['CD20_outside_gc_KDE_density'] <= cd20_density_60th), 'CD20_density_class'] = 5
filtered_data_cd20.loc[(filtered_data_cd20['CD20_outside_gc_KDE_density'] >= cd20_density_40th) & 
                  (filtered_data_cd20['CD20_outside_gc_KDE_density'] <= cd20_density_50th), 'CD20_density_class'] = 4
filtered_data_cd20.loc[(filtered_data_cd20['CD20_outside_gc_KDE_density'] >= cd20_density_30th) & 
                  (filtered_data_cd20['CD20_outside_gc_KDE_density'] <= cd20_density_40th), 'CD20_density_class'] = 3
filtered_data_cd20.loc[(filtered_data_cd20['CD20_outside_gc_KDE_density'] >= cd20_density_20th) & 
                  (filtered_data_cd20['CD20_outside_gc_KDE_density'] <= cd20_density_30th), 'CD20_density_class'] = 2
filtered_data_cd20.loc[(filtered_data_cd20['CD20_outside_gc_KDE_density'] >= cd20_density_10th) & 
                  (filtered_data_cd20['CD20_outside_gc_KDE_density'] <= cd20_density_20th), 'CD20_density_class'] = 1
filtered_data_cd20.loc[filtered_data_cd20['CD20_outside_gc_KDE_density'] < cd20_density_10th, 'CD20_density_class'] = 0 

data.loc[filtered_data_cd20.index, 'CD20_density_class'] = filtered_data_cd20['CD20_density_class']


# Save the updated filtered data to a new Excel file
excel_save_path = os.path.join(save_path, 'all_updated_data_with_KDE_CD56_CD20_density_filtered.xlsx')
data.to_excel(excel_save_path, index=False)

