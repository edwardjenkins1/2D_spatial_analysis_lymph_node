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


pixel_value_x = 0.325
pixel_value_y = 0.325



###### Load and produce dilation of segmented nuclei for measuring signal around edge vs. centre to identify marker associated to a given nuclei #####################

all_folders = 'C:/Users/edwardj/Desktop/Data to analyse for others/Jo'
segment_and_filtred = os.path.join(all_folders,'segmented and filtered')
fio.mkdir(segment_and_filtred)
nuclei_folder = os.path.join(all_folders,'Channel_2')
segmented_nuclei_folder = os.path.join(nuclei_folder,'segmentation')
segmented_nuclei = os.path.join(segmented_nuclei_folder, 'segmented_dilation_stardist.tif')
segmented_nuclei = skio.imread(segmented_nuclei)
print(np.max(np.unique(segmented_nuclei)))


###### Here you can assign those channels that will need edge signal measurements vs. cytoplasmic - this will use different measurements for detecting which labels associated to which channel ##########
channels_with_cytoplasmic = None
channels_on_membrane = [7, 8, 10, 11,15]

all_data = os.path.join(segmented_nuclei_folder, 'Centroids_intensity_all_channels.csv')
all_data = pd.read_csv(all_data)
all_data_df = pd.DataFrame(all_data)
all_data_updated = all_data_df.copy()

channels_with_cytoplasmic = None
channels_on_membrane = [7, 8, 10, 11,15]

channel_name_mapping = {
    7: 'CD20',
    8: 'CD3',
    10: 'CD56',
    11: 'CD68',
    15: 'tyrptase'
}

for channel_num in channels_on_membrane:
    # Get the channel name
    channel_name = channel_name_mapping.get(channel_num, f'Channel_{channel_num}')
    
    if channel_name in all_data_df.columns:
        print(f"Processing channel: {channel_name}")
        # Calculate the threshold for the current channel
        intensity_values = all_data_df[channel_name]
        median_intensity = intensity_values.median()
        min_intensity = intensity_values.min()
        max_intensity = intensity_values.max()
        percentile_5 = np.percentile(intensity_values, 1)
        percentile_95 = np.percentile(intensity_values, 99)

        filtered_intensity_values = intensity_values[(intensity_values > percentile_5) & (intensity_values < percentile_95)]
        filtered_median = filtered_intensity_values.median()
        filtered_std_dev = filtered_intensity_values.std()

        if channel_name in ['CD56', 'tyrptase', 'CD68']:
            # For CD56 or tryptase, use median + 1 * std deviation
            intensity_threshold = filtered_median + (3.5 * filtered_std_dev)

        else:
            # For other channels, use median + 1 * std deviation
            intensity_threshold = filtered_median + (1 * filtered_std_dev)

        percentile_filtered_labels = all_data_df[all_data_df[channel_name] > intensity_threshold]['label'].values


        # std_dev_intensity = all_data_df[channel_name].std()
        # intensity_threshold = median_intensity + (1*std_dev_intensity)
        # filtered_labels = all_data_df[all_data_df[channel_name] > intensity_threshold]['label'].values

        print (channel_name)
        print ('median and std', filtered_median, filtered_std_dev)
        print(f"Min Intensity: {min_intensity}")
        print(f"Max Intensity: {max_intensity}")
        print(f"5th Percentile: {percentile_5}")
        print(f"95th Percentile: {percentile_95}")
        print ('ratio 95 to 5th', percentile_95/percentile_5)


        # # Filter labels based on the threshold
        # filtered_labels = all_data_df[all_data_df[channel_name] > intensity_threshold]['label'].values
        filtered_mask = np.isin(segmented_nuclei, percentile_filtered_labels).astype(np.uint32)

        # Save the filtered mask as a new .tif file
        filtered_mask_path = os.path.join(segment_and_filtred, f'filtered_labels_{channel_name}.tif')
        skio.imsave(filtered_mask_path, filtered_mask)

        # Create a new column for positivity/negativity (1 for positive, 0 for negative)
        all_data_updated[f'{channel_name}_positive'] = (intensity_values > intensity_threshold).astype(int)
        
        print(f"Saved filtered mask for {channel_name} at: {filtered_mask_path}")
    else:
        print(f"Channel {channel_name} not found in the data.")

output_excel_path = os.path.join(segment_and_filtred, 'Updated_Centroids_Intensity_with_Positive_Labels.xlsx')
all_data_updated.to_excel(output_excel_path, index=False)
print(f"Saved updated data with positivity columns to: {output_excel_path}")


# median_intensity = channel_props_df['mean_intensity'].median()
# std_dev_intensity = channel_props_df['mean_intensity'].std()

# # Define the threshold (median + 1 * std_dev)
# intensity_threshold = median_intensity + (2*std_dev_intensity)

# # Filter the props_df based on the threshold
# filtered_props_df = channel_props_df[channel_props_df['mean_intensity'] > intensity_threshold]

# filtered_labels = filtered_props_df['label'].values
# filtered_mask = np.isin(segmented_nuclei, filtered_labels).astype(np.uint32)

# # Save the filtered mask as a new .tif file
# filtered_mask_path = os.path.join(segmented_nuclei_folder, f'filtered_labels{channels}.tif')
# skio.imsave(filtered_mask_path, filtered_mask)