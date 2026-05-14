import numpy as np
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
from skimage.morphology import binary_dilation,reconstruction,closing, square,binary_erosion,rectangle
import numpy.ma as ma
import skimage
from skimage.measure import label, regionprops
from skimage.color import label2rgb
import math
import pandas as pd
import shutil
from tqdm import tqdm
import sys
import os
import pickle


from Double_thresh_function import parameters_correction,find_original_position,find_center_square,find_edges, npy_correction_v3, sam_info



#######################################################################################################################################################
####################################### STEP 1 : Generate images information ##########################################################################
#######################################################################################################################################################

##############################################################################################
##############                       version time  20240806                     ##############
##############################################################################################
dest_path = "Output/SEGMENT_ANYTHING"    # path to save the choosed informations from the sam segmentations
npy_path = "Output/PREPROCESS_DATA/Resize"                       #  Path where the resized .npy files are stored
csv_path = "Output/PREPROCESS_DATA/Csv"                #  Path where the original .csv files are stored
img_path = "Output/PREPROCESS_DATA/Mask"        #  Path where the resized and corrected images are stored
mapping_path = os.path.join(npy_path,"mapping_info.pkl")      #  the mapping_info.pkl file is stored in the same path as the resized npy files
sam_result_path = "Output/SEGMENT_ANYTHING/sam_results.pkl"    #  Path where the segmentations by sam is stored

if not os.path.exists(dest_path):
    os.makedirs(dest_path)

with open(mapping_path, 'rb') as f:
    loaded_mapping_info = pickle.load(f)

# Load the results from the SAM
with open(sam_result_path, 'rb') as f:
    loaded_segment_sam_results = pickle.load(f)

img_all = os.listdir(img_path)   
img_all_png = [image for image in img_all if image.endswith('.png')]
# Step 2:  Initialize the variables
segment_sam_info = {}
segment_sam_table = pd.DataFrame(columns=['file', 'nbr_school','all_size','mean_intensity_school','mean_depth','mean_intensity_imgwithout0'])

# Step 3 : segment_sam_infoSAM methode
for file in tqdm(img_all_png, desc="Processing matrix"):
    file_info,new_row,center_file = sam_info(file = file ,sam_result = loaded_segment_sam_results,npy_path=npy_path,csv_path=csv_path,dest_path = dest_path)

    if file_info is not None and new_row is not None and center_file is not None:
        # Step 4 : From the pixels in center_file, to get the original corresponding pixel position for school center ( npy resized ---> npy )
        pixel_original = []
        depth_sea = []
        gps_lon_lat = []
        time = []
        csv_name = file.replace('_new.png','.csv')
        csv_table = pd.read_csv(os.path.join(csv_path,csv_name))
        # Step 4-1 : From center_file to get the school center in resized file
        cut_intervals = loaded_mapping_info[csv_name] 
        pixel_positions = [row[1] for row in center_file]
        
        # Step 4-2 : From mapping table to get the school position in original file
        for pixel_position in pixel_positions:
            original_position = find_original_position(pixel_position, cut_intervals)
            pixel_original.append(original_position)
            depth_sea.append(csv_table.iloc[original_position-1]['depth'])
            gps_lon_lat.append([csv_table.iloc[original_position-1]['Long'],csv_table.iloc[original_position-1]['Lat']])
            time.append(csv_table.iloc[original_position-1]['UTC_time'])

        # Save all the output by double threshold 
        file_info['sea_depth'] = depth_sea
        file_info['school_seabed_distance'] = [a - b for a, b in zip(file_info['sea_depth'], file_info['depth'])]
        file_info['gps_lon_lat'] = gps_lon_lat
        file_info['time'] = time
        file_info['depth_ratio'] = [d / s for d, s in zip(file_info['depth'], file_info['school_seabed_distance'])]
        segment_sam_info[file] = file_info
        segment_sam_table = pd.concat([segment_sam_table, pd.DataFrame([new_row])], ignore_index=True)

# Step 5 : Save all output
segment_sam_table.to_csv(f'{dest_path}/sam_image_masktb200_afterv0.csv', index=False)   
# Save the results to the specified file using pickle
save_path = os.path.join(dest_path, 'sam_info.pkl')
with open(save_path, 'wb') as f:
    pickle.dump(segment_sam_info, f)  





# read the output file by GPU
dis_limit = 10
dis_check = 50

with open(os.path.join(dest_path,'sam_info.pkl'), 'rb') as file:
    segment_sam_info = pickle.load(file)
##########################################################################################
# Combine the school informations from pkl file by sam  
##########################################################################################
file_list = list(segment_sam_info.keys())
seg_new = {}
for file in file_list:
    seg_new[file] = segment_sam_info[file]

# label all the school    
j = 1
# sam_school_table = pd.DataFrame(columns=['number', 'label','file_name','size','depth','intensity_school','center','sea_depth','school_seabed_distance','gps_lon_lat','time'])
sam_school_table = pd.DataFrame(columns=['number', 'label','file_name','size','bbox','center','depth','sea_depth',\
                                     'school_seabed_distance','gps_lon_lat','time',
                                     "width_bbox","length_box","is_very_wide","is_very_tall","dis_to_surface",\
                                     'dis_level',"axis_major_length","axis_minor_length","perimeter_school",\
                                    'intensity_school',"intensity_img","dif_intensity_school_image","center_square_intensity","edges_school_intensity",\
                                    "dif_intensity_center_edges","std_intensity_school","gradient_school",\
                                    "gradient_school_center","gradient_school_edges","dif_gradient_center_edges",\
                                    "width_length_ratio","axis_ellipse_ratio","solidity","compactness",\
                                    "inertia_tensor_eigvals_ratio","perimeter_area_ratio"])

for file in file_list:
    nbr_school = seg_new[file]['nbr_school']
    index_nbr = 1
    for i in range(len(seg_new[file]['label'])):
    # for label in seg_new[file]['label']:
        new_row = {'number':j, 
                   'label':seg_new[file]['label'][i],
                    'file_name':file,
                    'size':seg_new[file]['size'][i],
                    'bbox':seg_new[file]['bbox'][i],
                    'depth':seg_new[file]['depth'][i],
                    'center':seg_new[file]['center'][i],
                    'sea_depth':seg_new[file]['sea_depth'][i],
                    'school_seabed_distance':seg_new[file]['school_seabed_distance'][i],
                    'gps_lon_lat':seg_new[file]['gps_lon_lat'][i],
                    'time':seg_new[file]['time'][i],
                    "width_bbox":seg_new[file]['width_length'][i][0],
                    "length_box":seg_new[file]['width_length'][i][1],
                    "is_very_wide":seg_new[file]['is_very_wide'][i],
                    "is_very_tall":seg_new[file]['is_very_tall'][i],
                    'dis_to_surface':seg_new[file]['dis_to_surface'][i],

                    "axis_major_length":seg_new[file]['axis_ellipse'][i][0],
                    "axis_minor_length":seg_new[file]['axis_ellipse'][i][1],
                    "perimeter_school":seg_new[file]['perimeter_school'][i],
                    'intensity_school':seg_new[file]['intensity_school'][i][0],
                    "intensity_img":seg_new[file]['intensity_img'][0],
                    "center_square_intensity":seg_new[file]['center_square_intensity'][i],
                    "edges_school_intensity":seg_new[file]['edges_school_intensity'][i],

                    "dif_intensity_center_edges":seg_new[file]['dif_intensity_center_edges'][i],
                    "std_intensity_school":seg_new[file]['std_intensity_school'][i],
                    "gradient_school":seg_new[file]['gradient_school'][i],
                    "gradient_school_center":seg_new[file]['gradient_school_center'][i],
                    "gradient_school_edges":seg_new[file]['gradient_school_edges'][i],
                    "dif_gradient_center_edges":seg_new[file]['dif_gradient_center_edges'][i],\
                    "width_length_ratio":seg_new[file]['width_length_ratio'][i],
                    "axis_ellipse_ratio":seg_new[file]['axis_ellipse_ratio'][i],
                    "solidity":seg_new[file]['solidity'][i],
                    "compactness":seg_new[file]['compactness'][i],
                    "inertia_tensor_eigvals_ratio":seg_new[file]['inertia_tensor_eigvals_ratio'][i],
                    "perimeter_area_ratio":seg_new[file]['perimeter_area_ratio'][i]}
        new_row['dis_to_bottom'] = new_row['sea_depth']*10 - new_row['length_box']/2 - new_row['depth']*10
        if new_row['dis_to_surface'] > dis_check and new_row['dis_to_bottom'] > dis_check:
            dis_level = 2
        elif new_row['dis_to_surface'] < dis_limit or new_row['dis_to_bottom'] < dis_limit:
            dis_level = 0
        else:
            dis_level = 1
        new_row['dif_intensity_school_image'] = new_row['intensity_school'] - new_row['intensity_img']
        new_row['dis_level'] = dis_level
        index_nbr = index_nbr + 1
        sam_school_table = pd.concat([sam_school_table, pd.DataFrame([new_row])], ignore_index=True)

        j = j + 1


sam_school_table.to_csv(f'{dest_path}/sam_school_table.csv', index=False)   