import numpy as np 
import matplotlib.pyplot as plt
from skimage.morphology import binary_dilation,reconstruction,dilation, square,rectangle
import numpy.ma as ma
import skimage
from scipy import stats
import os
import pandas as pd
import sys
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from tqdm import tqdm
import math
import pickle
from skimage.transform import rescale
from skimage.filters import threshold_multiotsu,threshold_mean,threshold_otsu

from Double_thresh_function import npy_correction_v3,convergence_test_new20240730,find_original_position,find_edges,find_center_square,parameters_correction,thresh_info_new20240806


#######################################################################################################################################################
#################################### STEP 1 : Get thresh_max and thresh_min ###########################################################################
#######################################################################################################################################################

#####  version time 20240805
import warnings
# Suppress RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

###################################               path part                   ###################################  
npy_path = "F:/SURVEY2025/PREPROCESS_DATA/Resize"               #  Path where the resized .npy files are stored
dest_path_max = "F:/SURVEY2025/DOUBLE THRESHOLDING/Thresh_max"    # Path to save the thresh_max curve
dest_path_min = "F:/SURVEY2025/DOUBLE THRESHOLDING/Thresh_min"    # Path to save the thresh_min curve
csv_path = "F:/SURVEY2025/PREPROCESS_DATA/Csv"                   #  Path where the original .csv files are stored
dest_path_dt = "F:/SURVEY2025/DOUBLE THRESHOLDING/Schools"   # path to save the double threshold output images
mapping_path = npy_path                 #  Path where the mapping_info.pkl file is stored
###################################            end of path part               ###################################  

if not os.path.exists(dest_path_max):
    # Create the directory if it doesn't exist
    os.makedirs(dest_path_max)
if not os.path.exists(dest_path_min):
    # Create the directory if it doesn't exist
    os.makedirs(dest_path_min)
if not os.path.exists(dest_path_dt):
    os.makedirs(dest_path_dt)
    
files = os.listdir(npy_path)
files_npy = [file for file in files if file.endswith('npy')]

criteria_table = pd.DataFrame(columns=['file', 'convergence_all','nbr_school',
                                       'mean_c1','mean_c2','mean_c3',
                                       'min_c1','min_c2','min_c3',
                                       'thresh_max','min_pixel_intensity','total_rows',
                                       'shape_top_desc','shape_bottom_desc'])

shallow_depth = 50              # sea botttom depth
medium_depth = 70

for file in tqdm(files_npy):
    file_path = os.path.join(npy_path,file)
    img = np.load(file_path)
    min_pixel_intensity = np.min(img)
    if min_pixel_intensity > -150:                           # pass the images with echo parts        
        plt.imsave(f'{npy_path}/{file}_original.png',img)

        # Step 0 : find the corrections parameters
        csv_name = file.replace('_new.npy','.csv')
        csv_table = pd.read_csv(os.path.join(csv_path,csv_name))
        median_sea_depth = np.median(csv_table['depth'])

        shape_top,shape_bottom,total_rows,shape_top_desc,shape_bottom_desc = parameters_correction(img,median_sea_depth,threshold = -30)
            
        # Step 1 : mask the image 
        im = npy_correction_v3(img,total_rows,shape_top,shape_bottom)         # 1 change

        # Step 2 : find threshold_max 
        thresh_min = 0.04
        thresh_maxs = np.arange(10, 25, 1)*0.01     # Avoid floating-point precision errors
        regions_min = np.digitize(im, bins=[thresh_min])
        regions_min[im.mask] = 0
        nbr_school_all = []
        nbr_school = []

        for thresh_max in thresh_maxs:
            regions_max = np.digitize(im, bins=[thresh_max])
            regions_max[im.mask] = 0
            im_recon = reconstruction(regions_max, regions_min,method='dilation' )
          
            label_img = label(im_recon)
            regions = regionprops(label_img)
            big_regions = [region for region in regions if region.area > 20]
            nbr_school_all.append(len(regions))
            nbr_school.append(len(big_regions))

        # Analyse the nbr of school sequence
        max_start,max_end, nbr_estimate,convergence_value  = convergence_test_new20240730(nbr_school_all)
        # Plotting nbr_school vs thresh_max
        plt.figure()
        plt.plot(thresh_maxs, nbr_school, marker='o',label = "nbr_school")
        plt.plot(thresh_maxs, nbr_school_all, marker='o',label = "nbr_school_all")
        plt.title('Number of Detected Fish Schools vs thresh_max')
        plt.xlabel('thresh_max')
        plt.ylabel('Number of Fish Schools')
        x_ticks = thresh_maxs
        plt.xticks(x_ticks, rotation=45)  # Adjust rotation if needed
        plt.grid(True)
        plt.savefig(f'{dest_path_max}/{file}_thresh_max.png')
        plt.close()

        if convergence_value > 0:                          # 1 : change
            series = thresh_maxs[max_start:max_end+1]
            if len(series)%2 == 0:
                max_median = series[len(series)//2]
            else:
                max_median = series[(len(series)-1)//2]
            
            # Step 3 : find threshold_min 
            new_max = max_median
            new_mins = np.arange(1,int(100*(new_max+0.01)),1)*0.01
            regions_max2 = np.digitize(im, bins=np.array([new_max]))
            regions_max2[im.mask] = 0
            c1_mean = []
            c1_min = []
            c2_mean = []
            c2_min = []
            c3_mean = []
            c3_min = []
            nbr_school2_all = []

            for thresh_min in new_mins:
                regions_min2 = np.digitize(im, bins=np.array([thresh_min]))
                regions_min2[im.mask] = 0
                
                # double thresholding
                im_recon2 = reconstruction(regions_max2, regions_min2,method='dilation' )   

                # laber then caculate the pixels detected
                label_img2 = label(im_recon2)
                regions2 = regionprops(label_img2)
                big_regions2 = [region for region in regions2 if region.area > 20]
                nbr_school2_all.append(len(regions2))

                # creat the interest interval for threshold_max
                # for region in big_regions:    
                if len(big_regions2)>0:
                    solidity_mean = np.mean([region.solidity for region in big_regions2])
                    solidity_min = np.min([region.solidity for region in big_regions2])

                    peri_area_mean = np.mean([4*math.pi*(region.area)/(region.perimeter**2) for region in big_regions2])
                    peri_area_min = np.min([4*math.pi*(region.area)/(region.perimeter**2) for region in big_regions2])
                    
                    inertia_mean = np.mean([region.inertia_tensor_eigvals[1]/region.inertia_tensor_eigvals[0] for region in big_regions2])
                    inertia_min = np.min([region.inertia_tensor_eigvals[1]/region.inertia_tensor_eigvals[0] for region in big_regions2])
               
                else:
                    solidity_mean = 0
                    solidity_min = 0
                    peri_area_mean = 0
                    peri_area_min = 0
                    inertia_mean = 0
                    inertia_min = 0
                ################################################
                ################################################
                c1_mean.append(solidity_mean)
                c1_min.append(solidity_min)
                c2_mean.append(peri_area_mean)
                c2_min.append(peri_area_min)
                c3_mean.append(inertia_mean)
                c3_min.append(inertia_min)
                
            # Select the thresh_min interval corresponding to the same number of school as found before by 'nbr_estimate'
            nbr_school2_all = np.array(nbr_school2_all)
            selected_new_mins = new_mins[nbr_school2_all == nbr_estimate]
            # Plot Criteria 1 
            fig, ax1 = plt.subplots()
            ax1.plot(new_mins, c1_mean, label='c1_mean', marker='o')
            ax1.plot(new_mins, c1_min, label='c1_min', marker='o')
            ax1.plot(new_mins, c2_mean, label='c2_mean', marker='o')
            ax1.plot(new_mins, c2_min, label='c2_min', marker='o')
            ax1.plot(new_mins, c3_mean, label='c3_mean', marker='o')
            ax1.plot(new_mins, c3_min, label='c3_min', marker='o')
            ax1.set_xlabel('thresh_min')
            ax1.set_ylabel('Criteria 1')
            ax1.grid(True)
            ax1.legend()
            ax2 = ax1.twinx()
            ax2.plot(new_mins, nbr_school2_all, color='gray', label='nbr of all fish school', linestyle='--')
            ax2.set_ylabel('Number of Fish School')
            ax2.legend()
            if selected_new_mins.size > 0:
                ax1.axvline(x=np.min(selected_new_mins), color='red', linestyle='--')
                ax1.axvline(x=np.max(selected_new_mins), color='red', linestyle='--')
            plt.savefig(f'{dest_path_min}/{file}_thresh_min.png')
            plt.close()
            
            # get the choosen criteria values
            selected_c1_mean = [c1 for c1, new_min in zip(c1_mean, new_mins) if new_min in selected_new_mins]
            selected_c1_min = [c1 for c1, new_min in zip(c1_min, new_mins) if new_min in selected_new_mins]
            selected_c2_mean = [c1 for c1, new_min in zip(c2_mean, new_mins) if new_min in selected_new_mins]
            selected_c2_min = [c1 for c1, new_min in zip(c2_min, new_mins) if new_min in selected_new_mins]
            selected_c3_mean = [c1 for c1, new_min in zip(c3_mean, new_mins) if new_min in selected_new_mins]
            selected_c3_min = [c1 for c1, new_min in zip(c3_min, new_mins) if new_min in selected_new_mins]
        
            # Criteria 1    
            if selected_new_mins.size > 0:
                # plt.imsave(f'{dest_path_min}/{file}_original.png',img)
                # find the best solidity
                best_solidity_mean = np.argmax(selected_c1_mean)
                best_solidity_min = np.argmax(selected_c1_min)
                best_threshmin_mean1 = selected_new_mins[best_solidity_mean]
                best_threshmin_min1 = selected_new_mins[best_solidity_min]
                # print(f"the couple of threshs is by mean with criteria 1: ({thresh_max},{best_threshmin_mean1})")
                # print(f"the couple of threshs is by min with criteria 1: ({thresh_max},{best_threshmin_min1})")

                # Criteria 2    
                # find the best peri_area
                best_peri_area_mean = np.argmax(selected_c2_mean)
                best_peri_area_min = np.argmax(selected_c2_min)
                best_threshmin_mean2 = selected_new_mins[best_peri_area_mean]
                best_threshmin_min2 = selected_new_mins[best_peri_area_min]
                # print(f"the couple of threshs is by mean with criteria 2: ({thresh_max},{best_threshmin_mean2})")
                # print(f"the couple of threshs is by min with criteria 2: ({thresh_max},{best_threshmin_min2})")

                # Criteria 3   
                # find the best peri_area
                best_inertia_mean = np.argmax(selected_c3_mean)
                best_inertia_min = np.argmax(selected_c3_min)
                best_threshmin_mean3 = selected_new_mins[best_inertia_mean]
                best_threshmin_min3 = selected_new_mins[best_inertia_min]

                new_row = {'file':file, 
                           'convergence_all':convergence_value,
                           'nbr_school':nbr_estimate,
                            'mean_c1':best_threshmin_mean1, 
                            'mean_c2':best_threshmin_mean2, 
                            'mean_c3':best_threshmin_mean3,
                            'min_c1':best_threshmin_min1,
                            'min_c2':best_threshmin_min2,
                            'min_c3':best_threshmin_min3,
                            'thresh_max':new_max,
                            'min_pixel_intensity':min_pixel_intensity,
                            'total_rows':total_rows,
                            'shape_top_desc':shape_top_desc,
                            'shape_bottom_desc':shape_bottom_desc} 
                
            else:
                new_row = {'file':file, 
                           'convergence_all':convergence_value,
                           'nbr_school':nbr_estimate,
                            'mean_c1':None, 
                            'mean_c2':None, 
                            'mean_c3':None,
                            'min_c1':None,
                            'min_c2':None,
                            'min_c3':None,
                            'thresh_max':new_max,
                            'min_pixel_intensity':min_pixel_intensity,
                            'total_rows':total_rows,
                            'shape_top_desc':shape_top_desc,
                            'shape_bottom_desc':shape_bottom_desc} 
            criteria_table = pd.concat([criteria_table, pd.DataFrame([new_row])], ignore_index=True)


criteria_table.to_csv(f'{dest_path_min}/criteria_table.csv', index=False)



#######################################################################################################################################################
#################################### STEP 2 : Generate the double threshold images ####################################################################
#######################################################################################################################################################


###################################################
# new version time : 20240806 
###################################################
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

criteria_table = pd.read_csv(os.path.join(dest_path_min,"criteria_table.csv"))

with open(os.path.join(mapping_path,"mapping_info.pkl"), 'rb') as f:
    loaded_mapping_info = pickle.load(f)

files = [file for file in criteria_table['file']]
segment_threshold_results = {}
segment_threshold_table = pd.DataFrame(columns=['file', 'nbr_school','all_size','mean_intensity_school','mean_depth','mean_intensity_imgwithout0'])

for file in tqdm(files, desc="Processing matrix"):
   
    file_info ,new_row,center_file = thresh_info_new20240806(dest_path = dest_path_dt,criteria_table = criteria_table,file = file ,thresh_index = 1,npy_path=npy_path)     # 2 : new threshold function
    
    # Step 4 : From the pixels in center_file, to get the original corresponding pixel position for school center ( npy resized ---> npy )
    pixel_original = []
    depth_sea = []
    gps_lon_lat = []
    time = []
    csv_name = file.replace('_new.npy','.csv')
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
    segment_threshold_results[file] = file_info
    segment_threshold_table = pd.concat([segment_threshold_table, pd.DataFrame([new_row])], ignore_index=True)

# Step 5 : Save all output
segment_threshold_table.to_csv(f'{dest_path_dt}/segment_threshold_table.csv', index=False)   
# Save the results to the specified file using pickle
save_path = os.path.join(dest_path_dt, 'segment_threshold_results.pkl')
with open(save_path, 'wb') as f:
    pickle.dump(segment_threshold_results, f)  




#######################################################################################################################################################
#################################### STEP 3 : Generate fish schools ####################################################################
#######################################################################################################################################################


with open(f"{dest_path_dt}/segment_threshold_results.pkl", 'rb') as f:
    segment_threshold_results = pickle.load(f)


dis_limit = 10
dis_check = 50
file_list = list(segment_threshold_results.keys())   # check it 
seg_new = {}
for file in file_list:
    seg_new[file] = segment_threshold_results[file]

# label all the school    
j = 1
school_table = pd.DataFrame(columns=['number', 'label','file_name','size','bbox','center','depth','sea_depth',\
                                     'school_seabed_distance','gps_lon_lat','time',
                                     "width_bbox","length_box","is_very_wide","is_very_tall","dis_to_surface",\
                                    'dis_level',"axis_major_length","axis_minor_length","perimeter_school",\
                                    'intensity_school',"intensity_img","dif_intensity_school_image","center_square_intensity","edges_school_intensity",\
                                    "dif_intensity_center_edges","std_intensity_school","gradient_school",\
                                    "gradient_school_center","gradient_school_edges","dif_gradient_center_edges",\
                                    "width_length_ratio","axis_ellipse_ratio","solidity","compactness",\
                                    "inertia_tensor_eigvals_ratio","perimeter_area_ratio","thresh_max","thresh_min", "min_intensity"])

for file in file_list:
    nbr_school = seg_new[file]['nbr_school']
    for i in range(len(seg_new[file]['label'])):
        # print("label",label)
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
                    "perimeter_area_ratio":seg_new[file]['perimeter_area_ratio'][i],
                    "thresh_max":seg_new[file]['thresh_max'],
                    "thresh_min":seg_new[file]['thresh_min'],
                    "min_intensity":seg_new[file]['min_intensity']}
        new_row['dis_to_bottom'] = new_row['sea_depth']*10 - new_row['length_box']/2 - new_row['depth']*10
        if new_row['dis_to_surface'] > dis_check and new_row['dis_to_bottom'] > dis_check:
            dis_level = 2
        elif new_row['dis_to_surface'] < dis_limit or new_row['dis_to_bottom'] < dis_limit:
            dis_level = 0
        else:
            dis_level = 1
        new_row['dif_intensity_school_image'] = new_row['intensity_school'] - new_row['intensity_img']
        new_row['dis_level'] = dis_level

        school_table = pd.concat([school_table, pd.DataFrame([new_row])], ignore_index=True)

        j = j + 1
        
school_table.to_csv(f'{dest_path_dt}/school_table_thresh.csv', index=False)   