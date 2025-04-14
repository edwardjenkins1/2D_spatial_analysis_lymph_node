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

# def cross_g_function(source_positions, target_positions, radii):
#     # Build KDTree for the target
#     target_tree = KDTree(target_positions)
    
#     # Calculate nearest neighbor distances for source to target
#     distances, _ = target_tree.query(source_positions)
    
#     # Calculate G(r) for each radius
#     g_values = []
#     for r in radii:
#         g_values.append((distances <= r).sum() / len(source_positions))
    
#     return g_values

def cross_g_function(source_positions, target_positions, radii, source_channel, target_channel):
    # Build KDTree for the target
    target_tree = KDTree(target_positions)
    
    # Calculate nearest neighbor distances for source to target
    # If the source channel and target channel are the same, exclude self-matches
    distances = []
    for source_point in source_positions:
        # Query the nearest neighbors, k=2 to include the point itself
        dists, _ = target_tree.query(source_point, k=2)
        
        if source_channel == target_channel:
            # If the source and target channels are the same, take the second nearest neighbor (exclude self-match)
            distances.append(dists[1])
        else:
            # If the source and target channels are different, take the first nearest neighbor
            distances.append(dists[0])
    
    # Calculate G(r) for each radius
    g_values = []
    for r in radii:
        # For each radius, count how many distances are less than or equal to r
        g_values.append((np.array(distances) <= r).sum() / len(source_positions))
    
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

all_data_path = os.path.join(segment_and_filtred, 'all_updated_data_with_KDE_CD56_cd20_density_filtered-copy.xlsx')
all_data = pd.read_excel(all_data_path)
excel_files = os.path.join(segment_and_filtred, 'Excel files')
fio.mkdir(excel_files)
graph_files = os.path.join(segment_and_filtred, 'Plots')
fio.mkdir(graph_files)
distance_files = os.path.join(segment_and_filtred, 'Distance numbers')
fio.mkdir(distance_files)
g_files = os.path.join(segment_and_filtred, 'g plots')
fio.mkdir(g_files)

excel_files_CD20 = os.path.join(excel_files, 'CD20 to marker')
excel_files_CD3 = os.path.join(excel_files, 'CD3 to marker')
excel_files_CD56 = os.path.join(excel_files, 'CD56 to marker')
excel_files_CD68 = os.path.join(excel_files, 'CD68 to marker')
excel_files_tryptase = os.path.join(excel_files, 'tryptase to marker')
fio.mkdir(excel_files_CD20)
fio.mkdir(excel_files_CD3)
fio.mkdir(excel_files_CD56)
fio.mkdir(excel_files_CD68)
fio.mkdir(excel_files_tryptase)

graph_files_CD20 = os.path.join(graph_files, 'CD20 to marker')
graph_files_CD3 = os.path.join(graph_files, 'CD3 to marker')
graph_files_CD56 = os.path.join(graph_files, 'CD56 to marker')
graph_files_CD68 = os.path.join(graph_files, 'CD68 to marker')
graph_files_tryptase = os.path.join(graph_files, 'tryptase to marker')
fio.mkdir(graph_files_CD20)
fio.mkdir(graph_files_CD3)
fio.mkdir(graph_files_CD56)
fio.mkdir(graph_files_CD68)
fio.mkdir(graph_files_tryptase)

distance_files_CD20 = os.path.join(distance_files, 'CD20 to marker')
distance_files_CD3 = os.path.join(distance_files, 'CD3 to marker')
distance_files_CD56 = os.path.join(distance_files, 'CD56 to marker')
distance_files_CD68 = os.path.join(distance_files, 'CD68 to marker')
distance_files_tryptase = os.path.join(distance_files, 'tryptase to marker')
fio.mkdir(distance_files_CD20)
fio.mkdir(distance_files_CD3)
fio.mkdir(distance_files_CD56)
fio.mkdir(distance_files_CD68)
fio.mkdir(distance_files_tryptase)

g_files_CD20 = os.path.join(g_files, 'CD20 to marker')
g_files_CD3 = os.path.join(g_files, 'CD3 to marker')
g_files_CD56 = os.path.join(g_files, 'CD56 to marker')
g_files_CD68 = os.path.join(g_files, 'CD68 to marker')
g_files_tryptase = os.path.join(g_files, 'tryptase to marker')
fio.mkdir(g_files_CD20)
fio.mkdir(g_files_CD3)
fio.mkdir(g_files_CD56)
fio.mkdir(g_files_CD68)
fio.mkdir(g_files_tryptase)


# Define channel-positive columns
# channels = ['CD20', 'CD3', 'CD56', 'CD68', 'tryptase']
channels = ['CD20', 'CD56']
positive_columns = [f'{channel}_positive' for channel in channels]
radii = [0, 1, 5, 10, 25, 50, 100, 200]


gc_conditions = [
    (all_data['GC_yes_no'].isin([0, -1])),  # Condition 1: GC = 0 and -1
    # (all_data['GC_yes_no'].isin([1, -1])),  # Condition 2: GC = 1 and -1
    # (all_data['GC_yes_no'].isin([0, 1, -1]))  # Condition 3: GC = 0, 1, and -1
]

# Add labels for each condition (optional, for clarity)
gc_labels = ['outside GC', 
            #  'inside GC', 
            #  'all'
             ]
cd56_density_values = [0,1,2,3]  # Individual and combined density values
# cd56_density_labels = [ 'NK_60th'  'NK_80th']
# cd56_density_values = [0, 1, 2, 3]  # Individual and combined density values
cd56_density_labels = [ 'NK sparse', 'NK small', 'NK medium', 'NK big']
# cd56_density_values = [ 3]  # Individual and combined density values
# cd56_density_labels = [ 'NK big']

cd20_density_values = [0,1,2,3,4,5,6,7,8,9]  # Individual and combined density values
cd20_density_labels = ['most sparse', '10th', '20th', '30th', '40th', '50th', '60th', '70th', '80th', '90th']


tracemalloc.start()

# Start the outer loop for GC filtering
for gc_condition, gc_label in zip(gc_conditions, gc_labels):
    print(f"Processing data for condition: {gc_label}")

    all_data_df = all_data[gc_condition].copy()
    print ('Data after gc filter', all_data_df.shape)

    # Nested loop for CD56 density filtering
    for density_values, density_label in zip(cd56_density_values, cd56_density_labels):
        print(f"Processing data for CD56 density condition: {density_label}")

        for density_values_20, density_label_20 in zip(cd20_density_values, cd20_density_labels):
            print(f"Processing data for CD56 density condition: {density_label_20}")

            for source_channel in channels:

            # if source_channel == 'CD20':
            #     # CD20-specific filtering: CD20 == 1 and other channels == 0
            #     source_data = all_data_df[
            #         (all_data_df['CD20_positive'] == 1) & 
            #         (all_data_df[[f'{channel}_positive' for channel in channels if channel != 'CD20']].sum(axis=1) == 0)
            #     ].copy()
            # else:
            #     # General filtering: only source_channel is positive
            #     source_data = all_data_df[all_data_df[f'{source_channel}_positive'] == 1].copy()

                if source_channel == 'CD20':
                # CD20-specific filtering: CD20 == 1 and other channels == 0
                    source_data = all_data_df[
                        (all_data_df['CD20_positive'] == 1) & 
                        (all_data_df[[f'{channel}_positive' for channel in channels if channel != 'CD20']].sum(axis=1) == 0) &
                        (all_data_df['CD20_density_class'] == density_values_20)
                        ].copy()
                
                if source_channel == 'CD3':
                    # CD20-specific filtering: CD20 == 1 and other channels == 0
                    source_data = all_data_df[
                        (all_data_df['CD3_positive'] == 1) & 
                        (all_data_df[[f'{channel}_positive' for channel in channels if channel != 'CD3']].sum(axis=1) == 0)
                    ].copy()

                if source_channel == 'CD56':
                    # CD20-specific filtering: CD20 == 1 and other channels == 0
                    source_data = all_data_df[
                        (all_data_df['CD56_positive'] == 1) & 
                        (all_data_df[[f'{channel}_positive' for channel in channels if channel != 'CD56']].sum(axis=1) == 0) &
                        (all_data_df['CD56_density_class'] == density_values)].copy()

                if source_channel == 'CD68':
                    # CD20-specific filtering: CD20 == 1 and other channels == 0
                    source_data = all_data_df[
                        (all_data_df['CD68_positive'] == 1) & 
                        (all_data_df[[f'{channel}_positive' for channel in channels if channel != 'CD68']].sum(axis=1) == 0)
                    ].copy()
                
                if source_channel == 'tryptase':
                    # CD20-specific filtering: CD20 == 1 and other channels == 0
                    source_data = all_data_df[
                        (all_data_df['tryptase_positive'] == 1) & 
                        (all_data_df[[f'{channel}_positive' for channel in channels if channel != 'tryptase']].sum(axis=1) == 0)
                    ].copy()

                x_s = source_data['centroid-1']
                y_s = source_data['centroid-0']
                source_positions = pd.DataFrame({'X': x_s, 'Y': y_s})

                for target_channel in channels:
                    # print(f"Analyzing {source_channel} vs {target_channel}")

                    # if target_channel == 'CD20':
                    #     # CD20-specific filtering: CD20 == 1 and other channels == 0
                    #     target_data = all_data_df[
                    #         (all_data_df['CD20_positive'] == 1) & 
                    #         (all_data_df[[f'{channel}_positive' for channel in channels if channel != 'CD20']].sum(axis=1) == 0)
                    #     ].copy()
                    # else:
                    #     # General filtering: only target_channel is positive
                    #     target_data = all_data_df[all_data_df[f'{target_channel}_positive'] == 1].copy()
                    
                    if target_channel == 'CD20':
                    # CD20-specific filtering: CD20 == 1 and other channels == 0
                        target_data = all_data_df[
                            (all_data_df['CD20_positive'] == 1) & 
                            (all_data_df[[f'{channel}_positive' for channel in channels if channel != 'CD20']].sum(axis=1) == 0) &
                            (all_data_df['CD20_density_class'] == density_values_20)
                        ].copy()
                
                    if target_channel == 'CD3':
                        # CD20-specific filtering: CD20 == 1 and other channels == 0
                        target_data = all_data_df[
                            (all_data_df['CD3_positive'] == 1) & 
                            (all_data_df[[f'{channel}_positive' for channel in channels if channel != 'CD3']].sum(axis=1) == 0)
                        ].copy()

                    if target_channel == 'CD56':
                        # CD20-specific filtering: CD20 == 1 and other channels == 0
                        target_data = all_data_df[
                            (all_data_df['CD56_positive'] == 1) & 
                            (all_data_df[[f'{channel}_positive' for channel in channels if channel != 'CD56']].sum(axis=1) == 0) &
                            (all_data_df['CD56_density_class'] == density_values)
                        ].copy() 

                    if target_channel == 'CD68':
                        # CD20-specific filtering: CD20 == 1 and other channels == 0
                        target_data = all_data_df[
                            (all_data_df['CD68_positive'] == 1) & 
                            (all_data_df[[f'{channel}_positive' for channel in channels if channel != 'CD68']].sum(axis=1) == 0)
                        ].copy() 
                    
                    if target_channel == 'tryptase':
                        # CD20-specific filtering: CD20 == 1 and other channels == 0
                        target_data = all_data_df[
                            (all_data_df['tryptase_positive'] == 1) & 
                            (all_data_df[[f'{channel}_positive' for channel in channels if channel != 'tryptase']].sum(axis=1) == 0)
                        ].copy() 


                    x_t = target_data['centroid-1']
                    y_t = target_data['centroid-0']
                    target_positions = pd.DataFrame({'X': x_t, 'Y': y_t})


                #    # Compute distance matrix using the custom function
                #     dist_matrix_source_target = compute_distances_with_overlap(source_data, target_data, source_channel, target_channel)

                    # dist_matrix = pd.DataFrame(distance_matrix(source_positions.values, target_positions.values, p=2), index=source_positions.index, columns=target_positions.index)
                    
                    # # print (dist_matrix.shape)
                    # dist_matrix_um = dist_matrix_source_target

                    print(f"Current memory usage before matrix: {tracemalloc.get_traced_memory()}")  # Prints memory usage

                    dist_matrix = distance_matrix(source_positions.values, target_positions.values, p=2)
                    print (dist_matrix.shape, dist_matrix)
                    dist_matrix_um = dist_matrix * (pixel_x * pixel_y)
                    print(dist_matrix_um.shape, dist_matrix_um)

                    #####################################################################################################
                    ###################### nearest neighbourhood analysis #################################################
                    #####################################################################################################
                    print ('plotting nearest neighbour')

                    # Build KDTree for efficient nearest-neighbor search
                    source_positions_scaled = source_positions * np.array([pixel_x, pixel_y])
                    target_positions_scaled = target_positions * np.array([pixel_x, pixel_y])

                    # Build KDTree using scaled target positions

                    target_tree = KDTree(target_positions_scaled)

                    # Query nearest neighbors, excluding exact matches (same position)
                    nearest_distances = []
                    for i, source_point in enumerate(source_positions_scaled.values):
                        distances, indices = target_tree.query(source_point, k=2)  # k=2 to include the point itself
                        
                        if source_channel == target_channel:
                            # If the source and target channels are the same, exclude the first nearest neighbor (self-matching)
                            if distances[1] > 0:  # Exclude the first nearest neighbor (which is the source point itself)
                                nearest_distances.append(distances[1])
                        else:
                            # If the source and target channels are different, take the first nearest neighbor
                            nearest_distances.append(distances[0])

                    bins = np.arange(0, 200, 2.5) 
                    
                    # Compute histogram
                    hist, bins = np.histogram(nearest_distances, bins=bins)
                    percentage_freq = (hist / source_positions_scaled.shape[0]) * 100

                
                    # Save histogram data
                    hist_save_path = os.path.join(distance_files, f'{source_channel} to marker', f'{source_channel}_vs_{target_channel}_nearest_distances_{gc_label}_{density_label}_{density_label_20}.csv')
                    hist_df = pd.DataFrame({'Bin Start': bins[:-1], 'Bin End': bins[1:], 'Percentage': percentage_freq})
                    hist_df.to_csv(hist_save_path, index=False)
                    
                    # Plot histogram
                    plt.figure(figsize=(6, 6))
                    plt.bar(bins[:-1], percentage_freq, width=np.diff(bins), edgecolor='black', align='edge')
                    plt.xlabel(f'Distance to Nearest {target_channel} (µm)', fontsize=14)
                    plt.ylabel('Percentage Frequency (%)', fontsize=14)
                    plt.title(f'Nearest Neighbor Distance ({source_channel} to {target_channel})', fontsize=16)
                    plt.xticks(fontsize=12)
                    plt.yticks(fontsize=12)
                    plt.tight_layout()

                    # Save plot
                    plot_save_path = os.path.join(graph_files, f'{source_channel} to marker', f'{source_channel}_vs_{target_channel}_nearest_distance_histogram_{gc_label}_{density_label}_{density_label_20}.png')
                    plt.savefig(plot_save_path, dpi=1200)
                    plt.close()


                    ############## Nearest neighbourhood analysis - cross G function - just a cumulativ freq distibutrion - with theoretical distance map from poisson distribution)

                    g_values = cross_g_function(source_positions_scaled.values, target_positions_scaled.values, radii, source_channel, target_channel)

                    

                    plt.figure(figsize=(8, 6))
                    plt.plot(radii, g_values, label='Cross-G Function')
                    plt.xlabel('Radius (µm)')
                    plt.ylabel('G(r)')
                    plt.title('Cross-G Function')
                    plt.legend()
                    plt.tight_layout()

                    area = 15493173  # um^2, observation window size
                    lambda_B = len(target_positions_scaled) / area  # Density of target points
                    theoretical_g = theoretical_cross_g(radii, lambda_B)

                    # lower_g, upper_g = monte_carlo_simulation(source_positions_scaled.values, target_positions_scaled.values, radii, area)

                    # Plot Cross-G Function with Theoretical Curve and Envelopes
                    plt.figure(figsize=(8, 6))
                    plt.plot(radii, g_values, label='Empirical Cross-G', color='blue')
                    plt.plot(radii, theoretical_g, label='Theoretical Cross-G (CSR)', linestyle='--', color='green')

                    # Plot Cross-G Function with Theoretical Curve and Envelopes
                    plt.figure(figsize=(8, 6))
                    plt.plot(radii, g_values, label='Empirical Cross-G', color='blue', linewidth=2)
                    plt.plot(radii, theoretical_g, label='Theoretical Cross-G (CSR)', linestyle='--', color='green', linewidth=2)
                    # plt.fill_between(radii, lower_g, upper_g, color='gray', alpha=0.3, label='95% Confidence Envelope')

                    # Plot labels and legend
                    plt.xlabel('Radius (µm)', fontsize=14)
                    plt.ylabel('G(r)', fontsize=14)
                    plt.title(f'Cross-G Function Comparison\n{source_channel} vs {target_channel}', fontsize=16)
                    plt.legend(fontsize=12)
                    plt.tight_layout()


                    plot_save_path = os.path.join(graph_files, f'{source_channel} to marker', f'{source_channel}_vs_{target_channel}_g_value_{gc_label}_{density_label}_{density_label_20}.png')
                    # Save and show the plot
                    # plot_save_path = os.path.join(graph_files, f'{source_channel} to marker', f'{source_channel}_vs_{target_channel}_comparison_{gc_label}.png')
                    plt.savefig(plot_save_path, dpi=1200)

                    g_values_df = pd.DataFrame({
                        'Radius (µm)': radii,
                        'Empirical G(r)': g_values,
                        'Theoretical G(r)': theoretical_g
                    })

                    g_values_save_path = os.path.join(g_files, f'{source_channel} to marker', f'{source_channel}_vs_{target_channel}_g_value_{gc_label}_{density_label}_{density_label_20}.xlsx')
                    g_values_df.to_excel(g_values_save_path, index=False)



                    #####################################################################################################
                    ###################### distance analysis #################################################
                    #####################################################################################################

                    # Analyze the number of neighboring cells within each radius
                    for radius in radii:
                        print('radius', radius)
                        num_targets_within_radius = (dist_matrix_um <= radius).sum(axis=1)
                        if source_channel == target_channel:
                            num_targets_within_radius -= 1
                        print(num_targets_within_radius)
                        source_data[f'{source_channel} in {radius}'] = num_targets_within_radius

                    print(f"Current memory usage after matrix: {tracemalloc.get_traced_memory()}")  # Prints memory usage

                    # Plot the percentage of cells with neighbors within the radius
                    total_cells_per_source = source_data.shape[0]
                    print(f"Total {source_channel} cells: {total_cells_per_source}")
                    
                    from matplotlib.colors import Normalize
                    percentages_source = []
                    

                    # Define the range for neighbors (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 100)
                    neighbor_counts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 100]
                
                    # Loop through the radii and compute the percentages for each neighbor count
                    for radius in radii:
                        print ('radius', radius)
                        for n_neighbors in neighbor_counts:
                            has_neighbor_source = source_data[f'{source_channel} in {radius}'] >= n_neighbors
                            percentage_source = (has_neighbor_source.sum() / total_cells_per_source) * 100
                            percentages_source.append((radius, n_neighbors, percentage_source))

                    # Interpolate radii for 25% and 50% for each neighbor count
                    radius_25_percent_source = {}
                    radius_50_percent_source = {}

                    for n_neighbors in neighbor_counts:
                        percentages_for_neighbors = [perc for r, n, perc in percentages_source if n == n_neighbors]
                        radii_for_neighbors = [r for r, n, perc in percentages_source if n == n_neighbors]

                        radius_25_percent_source[n_neighbors] = np.interp(25, percentages_for_neighbors, radii_for_neighbors)
                        radius_50_percent_source[n_neighbors] = np.interp(50, percentages_for_neighbors, radii_for_neighbors)

                    # Data save path
                    data_save_path = os.path.join(excel_files, f'{source_channel} to marker', f'{source_channel}_vs_{target_channel}_radius_analysis_data_{gc_label}_{density_label}__{density_label_20}.csv')

                    # Combine the data into a DataFrame for easy export
                    plot_data = pd.DataFrame({
                        'Radius (µm)': [r for r, n, perc in percentages_source],
                        'Neighbors': [n for r, n, perc in percentages_source],
                        f'Percentage of {source_channel} cells with at least X {target_channel} neighbors (%)': [perc for r, n, perc in percentages_source]
                    })

                    # Save to CSV
                    plot_data.to_csv(data_save_path, index=False)

                    # Plot the results
                    plt.figure(figsize=(6, 6))

                    cmap = cm.Blues_r  # Use the reversed "Blues" colormap for the inverted gradient
                    norm = Normalize(vmin=0, vmax=len(neighbor_counts) - 1)

                    plt.rcParams['font.family'] = 'Arial'
                    plt.rcParams['axes.labelsize'] = 14  # Set the axis labels size
                    plt.rcParams['xtick.labelsize'] = 12  # Set the x-axis ticks size
                    plt.rcParams['ytick.labelsize'] = 12  # Set the y-axis ticks size
                    
                    for idx, n_neighbors in enumerate(neighbor_counts):
                        percentages_for_neighbors = [perc for r, n, perc in percentages_source if n == n_neighbors]
                        radii_for_neighbors = [r for r, n, perc in percentages_source if n == n_neighbors]
                        
                        # Get the color for this neighbor count based on the list of normalized colormap values
                        color = cmap(norm(idx))  # Get color from colormap based on the normalized index
                        
                        plt.plot(radii_for_neighbors, percentages_for_neighbors, label=f'{n_neighbors} Neighbors', color=color)

                

                    # # Plot data for each neighbor count
                    # for n_neighbors in neighbor_counts:
                    #     percentages_for_neighbors = [perc for r, n, perc in percentages_source if n == n_neighbors]
                    #     radii_for_neighbors = [r for r, n, perc in percentages_source if n == n_neighbors]
                    #     color = cm.Blues_r
                    #     plt.plot(radii_for_neighbors, percentages_for_neighbors, label=f'{n_neighbors} Neighbors', color=color)

                    #     # Mark the 25% and 50% threshold lines
                    #     plt.axvline(x=radius_50_percent_source[n_neighbors], color='gray', linestyle='--', label=f'50% at {radius_50_percent_source[n_neighbors]:.2f} µm (for {n_neighbors} neighbors)')
                    #     plt.axvline(x=radius_25_percent_source[n_neighbors], color='black', linestyle='--', label=f'25% at {radius_25_percent_source[n_neighbors]:.2f} µm (for {n_neighbors} neighbors)')

                    # Add labels and legend
                    plt.xlabel('Radius (µm)', fontsize=14)
                    plt.ylabel(f'Percentage of {source_channel} cells with at least X {target_channel} neighbors (%)',  fontsize=14)
                    plt.legend()
                    # plt.grid(True)
                    plt.tight_layout()

                    # Save the plot
                    plot_save_path = os.path.join(graph_files, f'{source_channel} to marker', f'{source_channel}_vs_{target_channel}_radius_analysis_{gc_label}_{density_label}__{density_label_20}.png')
                    plt.savefig(plot_save_path, dpi=1200)
                    plt.close()

                    # Clean up memory
                    gc.collect()
                    tracemalloc.stop()

                    # Print memory usage
                    print(f"Current memory usage: {tracemalloc.get_traced_memory()}")  # Prints memory usage

