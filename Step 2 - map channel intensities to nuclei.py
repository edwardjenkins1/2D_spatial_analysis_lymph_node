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


pixel_value_x = 0.325
pixel_value_y = 0.325



###### Load and produce dilation of segmented nuclei for measuring signal around edge vs. centre to identify marker associated to a given nuclei #####################

all_folders = 'C:/Users/edwardj/Desktop/Data to analyse for others/Jo'
nuclei_folder = os.path.join(all_folders,'Channel_2')
segmented_nuclei_folder = os.path.join(nuclei_folder,'segmentation')
segmented_nuclei = os.path.join(segmented_nuclei_folder, 'segmented_dilation_stardist.tif')
segmented_nuclei = skio.imread(segmented_nuclei)
print(np.max(np.unique(segmented_nuclei)))


###### Here you can assign those channels that will need edge signal measurements vs. cytoplasmic - work in progress still this will use different measurements for detecting which labels associated to which channel ##########
channels_with_cytoplasmic = None
channels_on_membrane = [7, 8, 10, 11,15]

props_df = pd.DataFrame()

for idx, channels in enumerate(channels_on_membrane): 

    channel_folder = os.path.join(all_folders, f'Channel_{channels}')
    print (channel_folder)
    tif = [file for file in os.listdir(channel_folder) if file.endswith('.tif')]
    print(tif)
    for tif_file in tif:
        print(tif_file)
        tif_file_path = os.path.join(channel_folder, tif_file)
        channel_img = skio.imread(tif_file_path)

        print ('segmented nuclei', segmented_nuclei.shape)
        print ('channel_img', channel_img.shape)


        # Use regionprops to calculate properties
        if idx == 0:  # For the first channel
            props = skmeasure.regionprops_table(
                segmented_nuclei, 
                intensity_image=channel_img, 
                properties=["label", 'area', 'centroid', "mean_intensity"]
            )
            # Initialize the DataFrame
            props_df = pd.DataFrame(props)
            props_df = props_df.rename(columns={'mean_intensity': f'Channel_{channels}'})

        else:  # For subsequent channels
            props = skmeasure.regionprops_table(
                segmented_nuclei, 
                intensity_image=channel_img, 
                properties=["label", "mean_intensity"]
            )
            # Create a DataFrame for the current channel
            channel_props_df = pd.DataFrame(props)
            ### Threshold and mask on cell populations for publication images ###########

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

            channel_props_df = channel_props_df.rename(columns={'mean_intensity': f'Channel_{channels}'})
            props_df = props_df.merge(channel_props_df[['label', f'Channel_{channels}']], on='label', how='left')


channel_name_mapping = {
    'Channel_1': 'DAPI_int',
    'Channel_2': 'DAPI_final',
    'Channel_3': 'blank',
    'Channel_4': 'CD14',
    'Channel_5': 'CD16',
    'Channel_6': 'CD169',
    'Channel_7': 'CD20',
    'Channel_8': 'CD3',
    'Channel_9': 'CD45',
    'Channel_10': 'CD56',
    'Channel_11': 'CD68',
    'Channel_12': 'HLA-ABC',
    'Channel_13': 'LYVE-1',
    'Channel_14': 'PNAD',
    'Channel_15': 'tryptase'
}

props_df.rename(columns=channel_name_mapping, inplace=True)

# Save to CSV
individual_cell_stats = os.path.join(segmented_nuclei_folder, f'Centroids_intensity_all_channels.csv')
props_df.to_csv(individual_cell_stats, index=False)



        




#################
### Add intensity across all channels to one excel 
#### Filter on those channels to now assign labels 
#### e.g., if above threshold for CD20 - then these nuclei belong to B cells - assign label and remove from pool for next label 
#### e.g., now filter on not B cells, and then strong filter for CD56 - this assigns NK cells 





### Dilate the nuclei segmeted #############
# dilated_nuclei = os.path.join(segmented_nuclei_folder, 'dilated_nuclei_stardist.tif')

# if os.path.exists(dilated_nuclei):
#     print(f"Loading dilated nuclei from stardist: {dilated_nuclei}")
#     dilated_nuclei = skio.imread(dilated_nuclei)

# else:
#     print("dilated nuclei.tif not found. Performing dilation.")

#     # Get the unique labels (excluding the background, label 0)
#     unique_labels = np.unique(segmented_nuclei)
#     unique_labels = unique_labels[unique_labels != 0]  # Remove background
#     dilated_nuclei = np.zeros_like(segmented_nuclei, dtype=segmented_nuclei.dtype)
#     structuring_element = skmorph.disk(2)

#     # Apply binary dilation to each nucleus
#     for label_value in unique_labels:
#         # Create a binary mask for the current nucleus
#         print(label_value)
#         binary_mask = segmented_nuclei == label_value
        
#         # Dilate the binary mask
#         dilated_mask = skmorph.dilation(binary_mask, structuring_element)
        
#         # Add the dilated mask back to the labeled output
#         dilated_nuclei[dilated_mask] = label_value

#     segmented_image_path = os.path.join(segmented_nuclei_folder, 'dilated_nuclei_stardist.tif')
#     skio.imsave(segmented_image_path, dilated_nuclei.astype(np.uint32))
#     print ('Dilation done')


# ##### Load in the nuclei centroids for each given label assigned to nuclei - will need later on #########################
# label_db = os.path.join(segmented_nuclei_folder,'Centroids.csv')
# label_db = pd.read_csv(label_db)
# print(label_db, label_db.shape)









#### Dilate the nuclei segmeted #############
# dilated_nuclei = os.path.join(segmented_nuclei_folder, 'dilated_nuclei_stardist.tif')

# if os.path.exists(dilated_nuclei):
#     print(f"Loading dilated nuclei from stardist: {dilated_nuclei}")
#     dilated_nuclei = skio.imread(dilated_nuclei)

# else:
#     print("dilated nuclei.tif not found. Performing dilation.")

#     # Get the unique labels (excluding the background, label 0)
#     unique_labels = np.unique(segmented_nuclei)
#     unique_labels = unique_labels[unique_labels != 0]  # Remove background
#     dilated_nuclei = np.zeros_like(segmented_nuclei, dtype=segmented_nuclei.dtype)
#     structuring_element = skmorph.disk(2)

#     # Apply binary dilation to each nucleus
#     for label_value in unique_labels:
#         # Create a binary mask for the current nucleus
#         binary_mask = segmented_nuclei == label_value
        
#         # Dilate the binary mask
#         dilated_mask = skmorph.dilation(binary_mask, structuring_element)
        
#         # Add the dilated mask back to the labeled output
#         dilated_nuclei[dilated_mask] = label_value

#     segmented_image_path = os.path.join(segmented_nuclei_folder, 'dilated_nuclei_stardist.tif')
#     skio.imsave(segmented_image_path, dilated_nuclei.astype(np.uint32))
#     print ('Dilation done')
