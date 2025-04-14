import os
import re
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D, art3d
import matplotlib.patches as patches

# ---------------- Scikit-Image ---------------- #
from skimage import io, filters, morphology, segmentation, measure, exposure, feature
from skimage.io import imread
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import (
    binary_erosion, remove_small_objects, disk, white_tophat, black_tophat, ball
)
from skimage.measure import mesh_surface_area
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.exposure import rescale_intensity, equalize_hist

# ---------------- SciPy & NumPy ---------------- #
import scipy.io as spio
import scipy.ndimage as ndi
from scipy.ndimage import zoom
from scipy.stats import gaussian_kde

# ---------------- 3D & Mesh ---------------- #
import trimesh
import igl
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ---------------- Image IO ---------------- #
import tifffile as tiff
from tifffile import imread

# ---------------- Machine Learning ---------------- #
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
import sklearn.cluster as cluster
from sklearn.datasets import fetch_openml
import hdbscan
import umap
from skimage import measure
from stardist.models import StarDist2D
from csbdeep.utils import normalize, normalize_mi_ma
from stardist.plot import render_label
from csbdeep.utils import Path

# ---------------- Numba ---------------- #
from numba import njit

# ---------------- Pyclesperanto ---------------- #
import pyclesperanto_prototype as cle
from pyclesperanto_prototype import imshow

# ---------------- Stardist (commented) ---------------- #
# from stardist import fill_label_holes, relabel_image_stardist, random_label_cmap
# from stardist.matching import matching_dataset

# ---------------- unwrap3D ---------------- #
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

# ---------------- Misc ---------------- #
import openpyxl as px
from skimage.data import cells3d
from csbdeep.utils import Path, download_and_extract_zip_file
import cv2

                


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


pixel_value_x = 0.325
pixel_value_y = 0.325

#### Code assumes that you have seperated out each intensity channel from tissue image into seperate folders
### FOlder structure goes: Main folder -- channel folders --- then .tif file with your image channel 

all_folders = 'C:/Users/edwardj/Desktop/Data to analyse for others/Jo'

for folder in os.listdir(all_folders):
    print ('all folders', os.listdir(all_folders))
    folder_path = os.path.join(all_folders, folder)

    if "Channel_1" in folder:
            continue  # Skip folders with the name "deconvolved data"
    
    # Check if the item in the directory is a folder
    if os.path.isdir(folder_path):
        print(f"Processing folder: {folder}")
        segmentation_folder = os.path.join(folder_path, 'segmentation')
        fio.mkdir(segmentation_folder)
        graphing_folder = os.path.join(folder_path, 'graphs')
        fio.mkdir(graphing_folder)

        for imgfile in os.listdir(folder_path):
            print(f"Processing image: {imgfile}")
            img_path = os.path.join(folder_path, imgfile)
            
            # Check if the item is a file and ends with '.tif'
            if os.path.isfile(img_path) and imgfile.lower().endswith('.tif'):
                print(f"Reading image: {imgfile}")
                img = skio.imread(img_path)
                print('Image shape:', img.shape)

                bg_removed_path = os.path.join(segmentation_folder, 'bg_removed.tif')

                if os.path.exists(bg_removed_path):
                    print(f"Loading existing background-removed image: {bg_removed_path}")
                    img_no_bg = skio.imread(bg_removed_path)

                else:
                    print("bg_removed.tif not found. Performing background subtraction.")
                    img_no_bg = subtract_background_2d(img, radius=50)
                    skio.imsave(bg_removed_path, img_no_bg.astype(np.uint32))
                    print(f"Background-removed image saved to: {bg_removed_path}")


                # img_no_bg = skfilters.gaussian(img_no_bg, 0.5)
                # img_no_bg = img_no_bg >= skfilters.threshold_otsu(img_no_bg)
                # img_no_bg = (img_no_bg > 0).astype(np.uint8)

                print (img_no_bg.shape)

                # Load the pre-trained StarDist model (assuming a 2D model)
                model = StarDist2D.from_pretrained('2D_versatile_fluo')
                # input_image = np.expand_dims(img_no_bg, axis=-1)
                input_image = img_no_bg
                input_image = normalize(input_image, pmin=1, pmax=99.8)
                labels, _ = model.predict_instances_big(input_image, prob_thresh=0.5, nms_thresh=0.4, n_tiles = (3,3), axes = 'YX', block_size = (4568, 5952), min_overlap = 128)


                        # tiled_image = os.path.join(segmentation_folder, 'normalized.tif')
                # skio.imsave(tiled_image, input_image.astype(np.uint32))
                # input_image = normalize(input_image)

                # tile_size = (4568, 5952)  # Example tile size
                # tiles = extract_tiles(input_image, tile_size)
                # labels = []
                # max_label = 0
                # for tile in tiles:
                #     tile_labels, _ = model.predict_instances(tile, prob_thresh=0.5, nms_thresh=0.8)
                #     tile_labels[tile_labels > 0] += max_label
                #     max_label = tile_labels.max() 
                #     labels.append(tile_labels)
                #     torch.cuda.empty_cache()
                #     print (tile.shape, max_label)
                #     print(labels.shape)
                # # final_labels = stitch_tiles(labels, image_shape = input_image.shape, tile_size=tile_size) # Combine tiles
                # final_labels = stitch_tiles(labels, image_shape=input_image.shape[:2], tile_size=tile_size)
                # print(final_labels.shape)

                # tiled_image = os.path.join(segmentation_folder, 'tiled.tif')
                # skio.imsave(tiled_image, final_labels.astype(np.uint32))

                # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                # ax[0].imshow(img_no_bg, cmap='gray')
                # ax[0].set_title('Original Binary Image')
                # ax[0].axis('off')
                # ax[1].imshow(render_label(labels))  # Color map for labeled components
                # ax[1].set_title('Segmented Nuclei')
                # ax[1].axis('off')
                # plt.tight_layout()
                # plt.show()

                segmented_image_path = os.path.join(segmentation_folder, 'segmented_dilation_stardist.tif')
                skio.imsave(segmented_image_path, labels.astype(np.uint32))


                properties = skmeasure.regionprops_table(labels, properties=['label', 
                                                                            'area',
                                                                            'axis_major_length',
                                                                            'axis_minor_length',
                                                                            'centroid',
                                                                            'eccentricity',
                                                                            'equivalent_diameter_area',
                                                                            'bbox',
                                                                            'perimeter'
                                                                            ])
                properties = pd.DataFrame(properties)
                properties['area'] = properties['area'] * (pixel_value_x*pixel_value_y)

                # filtered_properties = properties[properties['area'] > area_threshold]

                individual_cell_stats = os.path.join(segmentation_folder, 'Centroids.csv')
                properties.to_csv(individual_cell_stats, index=False)

                # # Create a list to store centroid coordinates
                # centroid_list = []

                # print ('creating data cloud')

                # # Populate the centroid_list with (x, y) coordinates
                # for region in regions:
                #     centroid = region.centroid
                #     label = region.label
                #     centroid_list.append((centroid[1], centroid[0]))
                #     print ('region #', label, centroid)

                # # Create a DataFrame from the centroid_list
                # df = pd.DataFrame(centroid_list, columns=['X', 'Y'])

                # # Save the DataFrame to an Excel file
                # df.to_excel(os.path.join(segmentation_folder, 'Centroids.xlsx'), index=False)
                # x = df['X']
                # y = df['Y']


                # # Create a 3D histogram
                # fig = plt.figure()
                # ax = fig.add_subplot(111, projection='3d')

                # x_max = np.max(x)
                # y_max = np.max(y)


                # # Perform 3D kernel density estimation
                # kde = gaussian_kde(np.vstack([x, y]), bw_method=0.05)
                # print (np.vstack([x,y]).shape)

                # fig = plt.figure(figsize=(10, 10))
                # ax = fig.add_subplot(111, projection='2d')
                # positions = np.vstack([x, y])
                # density = kde(positions)
                # # v_min = np.min(density)
                # # v_max = np.max(density)

                # # Scatter plot with color-coded density
                # scatter = ax.scatter(x, y, c=density, cmap='viridis', s=0.1, alpha=0.8, vmin=np.percentile(density,10), vmax=np.percentile(density,90))
                
                # ax.set_xlim([0, x_max])
                # ax.set_ylim([0, y_max])  

                # # Add colorbar
                # cbar = plt.colorbar(scatter, shrink=0.8, aspect=20)
                # cbar.set_label('Probability Density')

                # # Set axis labels
                # ax.set_xlabel('X Coordinates')
                # ax.set_ylabel('Y Coordinates')

                # # Save images from different angles
                # for angle in range(0, 360, 120):  # Change the range and step size based on your preferences
                #     # Set the view angles
                #     ax.view_init(elev=90, azim=angle)  # Adjust the elevation (vertical) and azimuth (horizontal) angles

                #     # Save the plot with the current view angle
                #     plot_save_path = os.path.join(graphing_folder, f'3D_positions_angle_{angle}_{folder}.svg')
                #     plt.savefig(plot_save_path)

                # # Save the plot
                # density_plot_save_path = os.path.join(graphing_folder, f'3D_density_scatter_plot_{folder}.svg')
                # plt.savefig(density_plot_save_path)
                # plt.close()






                # for percentile in range(0, 101, 15):
                # # Set the threshold as the current percentile
                #     threshold = np.percentile(density, percentile)

                #     # Filter points based on the threshold
                #     filtered_points = positions[:, density > threshold]

                #     # Create a 3D histogram
                #     fig = plt.figure(figsize=(10,10))
                #     ax = fig.add_subplot(111, projection='3d')
                  

                #     # Scatter plot with color-coded density for points above the threshold
                #     scatter = ax.scatter(filtered_points[0], filtered_points[1], filtered_points[2], c=density[density > threshold], cmap='viridis', s=0.1, alpha=0.8, vmin=np.percentile(density,10), vmax=np.percentile(density,90))

                #     # Set axis limits
                #     ax.set_xlim([0, x_max])
                #     ax.set_ylim([0, y_max])

                #     # Add colorbar
                #     cbar = plt.colorbar(scatter, shrink=0.8, aspect=20)
                #     cbar.set_label('Probability Density')

                #     # Set axis labels
                #     ax.set_xlabel('X Coordinates')
                #     ax.set_ylabel('Y Coordinates')
                #     ax.set_zlabel('Z Coordinates')

                #     # Save images from different angles
                #     for angle in range(0, 360, 180):
                #         ax.view_init(elev=90, azim=angle)
                #         plot_save_path = os.path.join(graphing_folder, f'{percentile}th_3D_positions_angle_{angle}.svg')
                #         plt.savefig(plot_save_path)

                #     # # Save the plot
                #     # density_plot_save_path = os.path.join(spatial_data_folder, f'{percentile}th_3D_density_scatter_plot.svg')
                #     # plt.savefig(density_plot_save_path)
                #     plt.close()































                # img_no_bg = skfilters.gaussian(img_no_bg, 0.5)
                # # img_no_bg = img_no_bg >= skfilters.threshold_otsu(img_no_bg)
                # segmented = cle.voronoi_otsu_labeling(img_no_bg, spot_sigma=2, outline_sigma=3)

                # unique_labels = np.unique(segmented)
                # print ('unique lbels', np.unique(unique_labels))

                # random_cmap = random_colormap(len(unique_labels), seed=42)


                # fig, ax = plt.subplots(1, 3, figsize=(12, 4))
                # ax[0].imshow(img, cmap='gray')
                # ax[0].set_title("Binary Image")
                # ax[1].imshow(img_no_bg, cmap='gray')
                # ax[1].set_title("Binary Image BG removed")
                # ax[2].imshow(segmented, cmap=random_cmap)
                # ax[2].set_title("Voronoi Labels")
                # plt.show()


                # # sauvola_segmented = cle.voronoi_otsu_labeling(thresholded_img, spot_sigma=1, outline_sigma=1)
                # print ('no erode')
                # # segmented_erosion = skmorph.erosion(segmented, footprint = skmorph.disk(1))
                # # segmented_erosion = remove_large_objects(segmented_erosion, min_size=300)

                # segmented_image_path = os.path.join(segmentation_folder, 'segmented.tif')
                # skio.imsave(segmented_image_path, segmented.astype(np.uint32))

                # # segmented_image_path = os.path.join(segmentation_folder, 'segmented_dilation.tif')
                # # skio.imsave(segmented_image_path, segmented_erosion.astype(np.uint16))

                # # segmented_image_path = os.path.join(segmentation_folder, 'sauvola_segmented.tif')
                # # skio.imsave(segmented_image_path, sauvola_segmented.astype(np.uint16))

                # # print ('voronoi label', np.unique(segmented).shape)



                ################################################
                #### now extract information from each label using regionprops 


                # #### centroid, area, volume, sphericity, circularity, complexity, axis lengths (x,y,z), how clustered the cell is (i.e. how many surrounding cells), how surrounded by red cancer, green cancer, 
                # segmented_dilation = np.array(segmented)
                # # Calculate the region properties including centroid for each labeled region
                # regions = measure.regionprops(segmented)

                # # Create a list to store centroid coordinates
                # centroid_list = []

                # print ('creating data cloud')

                # # Populate the centroid_list with (x, y) coordinates
                # for region in regions:
                #     centroid = region.centroid
                #     label = region.label
                #     centroid_list.append((centroid[1], centroid[0]))
                #     print ('region #', label, centroid)

                # # Create a DataFrame from the centroid_list
                # df = pd.DataFrame(centroid_list, columns=['X', 'Y'])

                # # Save the DataFrame to an Excel file
                # df.to_excel(os.path.join(segmentation_folder, 'Centroids.xlsx'), index=False)
                # x = df['X']
                # y = df['Y']


                # # Create a 3D histogram
                # fig = plt.figure()
                # ax = fig.add_subplot(111, projection='3d')

                # x_max = np.max(x)
                # y_max = np.max(y)


                # # Perform 3D kernel density estimation
                # kde = gaussian_kde(np.vstack([x, y]), bw_method=0.05)
                # print (np.vstack([x,y]).shape)

                # fig = plt.figure(figsize=(10, 10))
                # ax = fig.add_subplot(111, projection='2d')
                # positions = np.vstack([x, y])
                # density = kde(positions)
                # # v_min = np.min(density)
                # # v_max = np.max(density)

                # # Scatter plot with color-coded density
                # scatter = ax.scatter(x, y, c=density, cmap='viridis', s=0.1, alpha=0.8, vmin=np.percentile(density,10), vmax=np.percentile(density,90))
                
                # ax.set_xlim([0, x_max])
                # ax.set_ylim([0, y_max])  

                # # Add colorbar
                # cbar = plt.colorbar(scatter, shrink=0.8, aspect=20)
                # cbar.set_label('Probability Density')

                # # Set axis labels
                # ax.set_xlabel('X Coordinates')
                # ax.set_ylabel('Y Coordinates')

                # # Save images from different angles
                # for angle in range(0, 360, 120):  # Change the range and step size based on your preferences
                #     # Set the view angles
                #     ax.view_init(elev=90, azim=angle)  # Adjust the elevation (vertical) and azimuth (horizontal) angles

                #     # Save the plot with the current view angle
                #     plot_save_path = os.path.join(graphing_folder, f'3D_positions_angle_{angle}_{folder}.svg')
                #     plt.savefig(plot_save_path)

                # # Save the plot
                # density_plot_save_path = os.path.join(graphing_folder, f'3D_density_scatter_plot_{folder}.svg')
                # plt.savefig(density_plot_save_path)
                # plt.close()






                # for percentile in range(0, 101, 15):
                # # Set the threshold as the current percentile
                #     threshold = np.percentile(density, percentile)

                #     # Filter points based on the threshold
                #     filtered_points = positions[:, density > threshold]

                #     # Create a 3D histogram
                #     fig = plt.figure(figsize=(10,10))
                #     ax = fig.add_subplot(111, projection='3d')
                  

                #     # Scatter plot with color-coded density for points above the threshold
                #     scatter = ax.scatter(filtered_points[0], filtered_points[1], filtered_points[2], c=density[density > threshold], cmap='viridis', s=0.1, alpha=0.8, vmin=np.percentile(density,10), vmax=np.percentile(density,90))

                #     # Set axis limits
                #     ax.set_xlim([0, x_max])
                #     ax.set_ylim([0, y_max])
                #     ax.set_zlim([0, z_max])

                #     # Add colorbar
                #     cbar = plt.colorbar(scatter, shrink=0.8, aspect=20)
                #     cbar.set_label('Probability Density')

                #     # Set axis labels
                #     ax.set_xlabel('X Coordinates')
                #     ax.set_ylabel('Y Coordinates')
                #     ax.set_zlabel('Z Coordinates')

                #     # Save images from different angles
                #     for angle in range(0, 360, 180):
                #         ax.view_init(elev=90, azim=angle)
                #         plot_save_path = os.path.join(graphing_folder, f'{percentile}th_3D_positions_angle_{angle}.svg')
                #         plt.savefig(plot_save_path)

                #     # # Save the plot
                #     # density_plot_save_path = os.path.join(spatial_data_folder, f'{percentile}th_3D_density_scatter_plot.svg')
                #     # plt.savefig(density_plot_save_path)
                #     plt.close()








                # unique_labels = np.unique(segmented)
                # num_labels = len(unique_labels)
                # label_colors = generate_random_colors(num_labels)
                # colored_image = label_to_color(segmented, label_colors)

                # # Save the colored segmented image
                # colored_segmented_image_path = os.path.join(segmentation_folder, 'colour_segmented.tif')
                # skio.imsave(colored_segmented_image_path, colored_image.astype(np.uint16))

                # # Display the result
                # plt.imshow(colored_image)
                # plt.axis('off')
                # plt.show()




                # # Get unique labels
                # unique_labels = np.unique(segmented)

                # # Call the Numba-compiled function
                # colored_array = color_labels(segmented, unique_labels)

                # # Scale the RGB values to the 16-bit range (0 to 65535)
                # scaled_array = (colored_array * 65535).astype(np.uint16)

                









            #     # Get unique labels from the segmented image
            #     labels = np.unique(segmented)

            #     print (labels.shape)

            #    # Generate random colors for each unique label
            #     random_colors = np.random.randint(0, 65535, size=(len(labels), 3))
            #     print(random_colors.shape)

            #     # Create a mpping from label to random color
            #     label_color_map = dict(zip(labels, random_colors))

            #     # Apply the random colors to the segmented image
            #     colored_segmented = np.zeros((segmented.shape[0], segmented.shape[1], segmented.shape[2], 3), dtype=np.uint16)

            #     for label in labels:
            #         mask = segmented == label
            #         colored_segmented[mask] = label_color_map[label]


                

                # # Save the colored segmented image
                # colored_segmented_image_path = os.path.join(segmentation_folder, 'colour_segmented.tif')
                # skio.imsave(colored_segmented_image_path, (colored_segmented[:, :, :, :3] * 65535).astype(np.uint16))

                # img = ndimage.zoom(img,
                #             zoom=[z_res/factor, 1./factor, 1./factor],
                #             order=0,
                #             mode='reflect')
                
               
                # # Enhance contrast
                # contrast_img = skfilters.rank.enhance_contrast(img, skmorph.ball(4))
                # lowest_value_points = ndi.label(contrast_img < skfilters.threshold_otsu(contrast_img))[0]
                # segmented_image_path = os.path.join(segmentation_folder, 'lowest value.tif')
                # skio.imsave(segmented_image_path, lowest_value_points.astype(np.uint16))


                # labels = voronoi_otsu(contrast_img, skmorph.ball(1))
                

               
                # # # # Find markers for watershed segmentation
                # markers = ndi.label(contrast_img > skfilters.threshold_otsu(contrast_img), skmorph.ball(1))[0]
                # print ('unique labels', np.unique(markers).shape)
                # segmented_image_path = os.path.join(segmentation_folder, 'markers.tif')
                # skio.imsave(segmented_image_path, markers.astype(np.uint16))

                
                # distance = ndi.distance_transform_edt(markers, sampling = np.array([20, 1.657, 1.657]))
                # print (distance)
                # coords = peak_local_max(distance, labels=markers, footprint=skmorph.ball(1), num_peaks_per_label = 3, exclude_border=1 ) ## generates seeds for the watershed
                # print (np.unique(coords).shape)

                
                # # make image to check seed points ######### 

                # # Create a new image with the same shape as contrast_img
                # marked_image = np.zeros_like(contrast_img, dtype=np.uint16)
                # # Mark the coordinates in the new image
                # for index, coord in enumerate(coords):
                #     marked_image[coord[0], coord[1], coord[2]] = index + 1

                # # Save the marked image as a TIFF file
                # marked_image_path = os.path.join(segmentation_folder, 'watershed_seedpoints.tif')
                # skio.imsave(marked_image_path, marked_image.astype(np.uint16))

               
                # mask = lowest_value_points
                # mask[tuple(coords.T)] = False
                # markers, _ = ndi.label(mask)
                # labels = watershed(-distance, marked_image, mask=img > skfilters.threshold_otsu(img))
                
                # labels = ndi.label(labels, skmorph.ball(1))[0]
                # print ('labels', np.unique(labels).shape)                

                # segmented_image_path = os.path.join(segmentation_folder, 'segmentation.tif')
                # skio.imsave(segmented_image_path, labels.astype(np.uint16))



                ########### Produce 3D plot showing cnetroid of markers vs. local maxima of markers ########################### 

                # # Calculate centroids of labeled regions
                # markers_props = skmeasure.regionprops(markers)

                # # Plot the original image with 3D dots
                # fig = plt.figure(figsize=(10, 10))
                # ax = fig.add_subplot(111, projection='3d')

                # # Extract x, y, and z coordinates from the local maxima
                # coords_x, coords_y, coords_z = coords.T

                # # Plot 3D green dots
                # ax.scatter(coords_y, coords_x, coords_z, c='g', marker='o', label='Peak Local Maxima')

                # # Plot 3D red dots for markers with calculated centroids
                # for prop in markers_props:
                #     centroid = prop.centroid
                #     ax.scatter(centroid[1], centroid[0], centroid[2], c='r', marker='o', label='Marker Centroids')

                # ax.set_xlabel('X')
                # ax.set_ylabel('Y')
                # ax.set_zlabel('Z')

                # plt.show()


               


                
                # # Plotting the results
                # fig, axes = plt.subplots(1, 4, figsize=(12, 4))
                # ax = axes.ravel()

                # ax[0].imshow(img, cmap='gray')
                # ax[0].set_title('Original Image')

                # ax[1].imshow(denoised_img, cmap='gray')
                # ax[1].set_title('Denoised Image')

                # ax[2].imshow(contrast_img, cmap='gray')
                # ax[2].set_title('Contrast Enhanced')

                # ax[3].imshow(segmentation, cmap='nipy_spectral')
                # ax[3].set_title('Segmentation')

                # for a in ax:
                #     a.axis('off')

                # plt.tight_layout()
                # plt.show()





                # ########### equalize stack intensities ############ 

                # # Equalize mean intensities of slices transformation
                # equalized_intensities_stack = np.zeros_like(img)

                # num_slices = img.shape[0]
                # mean_intensity_stack = np.mean(img)

                # for z in range(0, num_slices):
                #     # get a single slice out of the stack
                #     a_slice = img[z, :, :]
                #     # measure its intensity
                #     mean_intensity_slice = np.mean(a_slice)
                #     # correct the intensity
                #     correction_factor = mean_intensity_slice / mean_intensity_stack
                #     corrected_slice = a_slice / correction_factor
                #     # copy slice back into a stack
                #     equalized_intensities_stack[z, :, :] = corrected_slice

                # img = equalized_intensities_stack
                # segmented_image_path = os.path.join(segmentation_folder, 'equalized stack.tif')
                # skio.imsave(segmented_image_path, img.astype(np.uint16))


    #             # Reorder axis order from (z, y, x) to (x, y, z)
    #             im_orig = img.transpose()

    #             # # Rescale image data to range [0, 1]
    #             # im_orig = np.clip(im_orig,
    #             #                 np.percentile(im_orig, 5),
    #             #                 np.percentile(im_orig, 95))
    #             # im_orig = (im_orig - im_orig.min()) / (im_orig.max() - im_orig.min())

    #             # # Degrade image by applying exponential intensity decay along x
    #             # sigmoid = np.exp(-3 * np.linspace(0, 1, im_orig.shape[0]))
    #             # im_degraded = (im_orig.T * sigmoid).T

    #             # Set parameters for AHE
    #             # Determine kernel sizes in each dim relative to image shape
    #             # kernel_size = (im_orig.shape[0] // 5,
    #             #             im_orig.shape[1] // 5,
    #             #             im_orig.shape[2] // 2)
    #             # kernel_size = np.array(kernel_size)
    #             # clip_limit = 0.9

    #            # Perform histogram equalization without normalization
    #             im_orig_he = np.empty_like(im_orig, dtype=np.float64)
    #             for i in range(im_orig.shape[2]):
    #                 im_orig_he[:, :, i] = exposure.equalize_hist(im_orig[:, :, i])

    #             # Perform adaptive histogram equalization without normalization
    #             im_orig_ahe = np.empty_like(im_orig, dtype=np.float64)
    #             for i in range(im_orig.shape[2]):
    #                 im_orig_ahe[:, :, i] = exposure.equalize_adapthist(
    #                     im_orig[:, :, i],
    #                     kernel_size=kernel_size,
    #                     clip_limit=clip_limit
    # )
                
    #             im_orig_he = im_orig_he.transpose((2, 1, 0))
    #             im_orig_ahe = im_orig_ahe.transpose((2, 1, 0))
    #             print ('histo', im_orig_he.shape, im_orig_ahe.shape)
    #             segmented_image_path = os.path.join(segmentation_folder, 'img_orig_he.tif')
    #             skio.imsave(segmented_image_path, (im_orig_he * 65535).astype(np.uint16))
    #             segmented_image_path = os.path.join(segmentation_folder, 'img_orig_adaptive_he.tif')
    #             skio.imsave(segmented_image_path, (im_orig_ahe * 65535).astype(np.uint16))
                
                ################## tophat background subtraction #####################
                
                

                # segmented_image_path = os.path.join(segmentation_folder, 'BG removed.tif')
                # skio.imsave(segmented_image_path, img.astype(np.uint16))

                # thresholded_img = img > skfilters.threshold_sauvola(img)
                # thresholded_img = skmorph.remove_small_objects(thresholded_img, min_size = 10)
                # thresholded_img = ndi.label(thresholded_img)[0]
                # segmented_image_path = os.path.join(segmentation_folder, 'sauvola.tif')
                # skio.imsave(segmented_image_path, thresholded_img.astype(np.uint16))


                ################## segmentation ###############################
