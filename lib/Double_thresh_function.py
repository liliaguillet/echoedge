import numpy as np 
import matplotlib.pyplot as plt
from skimage.morphology import binary_dilation,reconstruction,rectangle
import numpy.ma as ma
from skimage.transform import rescale
import math
import skimage
import os
from skimage.measure import label, regionprops
import pandas as pd
from dateutil import parser
from astral import LocationInfo
from astral.sun import sun
import datetime

########################################   list of functions  ########################################
#  line : 32        apply_day_or_night(row)               (not used in the key code part)
#  line : 36        convergence_test_new20240730(arr)      (to test the convergence of the images)
#  line : 84        day_or_night(coords,timezone,time)      (not used in the key code part)
#  line : 122       find_center_square(centroid, coords)      (get the square 3*3 centered by a choosen pixel)  
#  line : 139       find_edges(image, row, col)
#  line : 178       find_first_incolumns(matrix,element)       (not used in the key code part)
#  line : 195       find_minimum_dis(matrix2,matrix1)        
#  line : 214       find_original_position(resized_pixel, intervals)
#  line : 227       inter_positions_new(arr)
#  line : 244       npy_correction_v3(img,total_rows,shape_top,shape_bottom)
#  line : 280       parameters_correction(img,median_sea_depth,threshold = -30)          
#  line : 335       resize_matrix_31052024(matrix,velocity,meanvelocity)
########################################   end of list        ########################################

# # Define a helper function to apply the function day_or_night(coords,timezone,time) to a dataframe
# def apply_day_or_night(row):
#     return day_or_night(row['gps_lon_lat'], 'Australia/Melbourne', row['time'])


def convergence_test_new20240730(arr):
    """
    Determine convergence type and intervals in a sequence.

    Args:
        arr (list): Input sequence.

    Returns:
        tuple: A tuple containing:
            - max_start_index (int): Start index of the longest convergence interval.
            - max_end_index (int): End index of the longest convergence interval.
            - convergence (int): Type of convergence (0, 1, or 2). 
    """

    # Initialize list to store segment indices
    segments = []
    # Find segment indices where the value changes
    for i in range(1,len(arr)):
        if arr[i] != arr[i-1]:
            segments.append(i)
    segments.insert(0, 0)

    temp = segments.copy()
    temp.append(len(arr))
    temp = temp[1:]
    # Compute the length of each segment
    seq_length = [j - i for i, j in zip(segments, temp)] 
    # Find the maximum segment length and its index
    max_length = np.max(seq_length)
    max_p = len(seq_length) - 1 - seq_length[::-1].index(max_length)
    # max_p = seq_length.index(max_length)

    max_start_index = segments[max_p]
    max_end_index = temp[max_p]-1

    if arr[max_start_index] == 0:
        convergence = 0
    else:
        if max_length >= 4:
            convergence = 2
        else:
            convergence = 1
       
    

    # Convert max_start_index and max_end_index to integers
    max_start_index = int(max_start_index)
    max_end_index = int(max_end_index)
    return max_start_index, max_end_index, arr[max_start_index],convergence


# def day_or_night(coords,timezone,time):
#     # Remove the square brackets
#     coordinates_str = coords.strip("[]")

#     # Split the string by comma to get a list of strings
#     coordinates_list_str = coordinates_str.split(",")

#     # Convert the list of strings to a list of floats
#     coordinates = [float(coord) for coord in coordinates_list_str]

#     # Access the first and second numbers
#     lon = coordinates[0]
#     lat = coordinates[1]

#     loc = LocationInfo(name='Melbourne', region='Australia', timezone=timezone,
#                    latitude=lat, longitude=lon)

#     # Convert the string to a datetime object
#     date_datetime = parser.isoparse(time)
#     year = date_datetime.year
#     month = date_datetime.month
#     day = date_datetime.day
  
#     s = sun(loc.observer, date=datetime.date(year,month,day), tzinfo=loc.timezone)
#     sunrise_time = s['sunrise']
#     sunset_time = s['sunset']

#     # Format the sunrise and sunset times
#     sunrise_str = sunrise_time.strftime('%Y-%m-%d %H:%M:%S %Z')
#     sunset_str = sunset_time.strftime('%Y-%m-%d %H:%M:%S %Z')

#     if sunrise_time < date_datetime < sunset_time :
#         return 1,sunrise_str,sunset_str
#     else:
#         return 0,sunrise_str,sunset_str



def find_center_square(centroid, coords):
    """
    Get the square of the center in the image: get the square 3*3 centered by a choosen pixel
    
    """
    int_centroid = np.round(centroid).astype(int)
    square_matrice = [[int_centroid[0]-1,int_centroid[1]-1],[int_centroid[0]-1,int_centroid[1]],[int_centroid[0]-1,int_centroid[1]+1],\
                               [int_centroid[0],int_centroid[1]-1],[int_centroid[0],int_centroid[1]],[int_centroid[0],int_centroid[1]+1],\
                                [int_centroid[0]+1,int_centroid[1]-1],[int_centroid[0]+1,int_centroid[1]],[int_centroid[0]+1,int_centroid[1]+1]]
    square_matrice_set = set(map(tuple, square_matrice))
    coords_set = set(map(tuple, coords))
    common_coords = square_matrice_set.intersection(coords_set)
    
    # Convert the set back to a list of lists
    return list(map(list, common_coords))


def find_edges(image, row, col):
    """
    Get the four connected neighbors of a given position in the image.

    Parameters:
    - image: numpy array representing the image or matrix
    - row, col: coordinates of the position in the image

    Returns:
    - List of tuples containing coordinates of the four connected neighbors
    """
    edges_object = []
    neighbors_pixel_sum_c = []
    for i, j in zip(row,col):
        neighbors = []
        # Check north neighbor (above)
        if i > 0:
            neighbors.append((i - 1, j))

        # Check south neighbor (below)
        if i < image.shape[0] - 1:
            neighbors.append((i + 1, j))

        # Check west neighbor (left)
        if j > 0:
            neighbors.append((i, j - 1))

        # Check east neighbor (right)
        if j < image.shape[1] - 1:
            neighbors.append((i, j + 1))

        sum_neighbors_pixels = np.sum([image[n[0], n[1]] for n in neighbors])
        
        if sum_neighbors_pixels<4:
            edges_object.append([i,j])
    return edges_object



# def find_first_incolumns(matrix,element): 
#     # find the element position in each columne
#     coords = np.argwhere(matrix == element)
#     coords_r = coords[:, 0]
#     coords_c = coords[:, 1]
#     column_count = []
#     first_postion = []
#     last_postion = []
#     for i in np.unique(coords_c):
#         coords_f = [coord[0] for coord in coords if coord[1] == i]
#         first_postion.append(np.min(coords_f))
#         last_postion.append(np.max(coords_f))
#         column_count.append(i)
#     return column_count,first_postion,last_postion



# def find_minimum_dis(matrix2,matrix1):
#     # matrix2 to matrix1 
#     column_count1,first_postion1,last_postion1 = find_first_incolumns(matrix1,False)
#     column_count2,first_postion2,last_postion2 = find_first_incolumns(matrix2,True)
    
#     distance_top = []
#     distance_bottom = []
#     for i in range(0,len(column_count2)):
#         index1 = np.where(np.array(column_count1) == column_count2[i])[0]
#         index1 = index1[0]
#         # print('i,index1',i,index1,'top_f,m_top_t',first_postion1[index1],first_postion2[i],first_postion2[i]-first_postion1[index1])
#         # print('i,index1',i,index1,'bot_f,m_bot_t',last_postion1[index1],last_postion2[i],last_postion1[index1]-last_postion2[i])
#         distance_top.append(first_postion2[i]-first_postion1[index1])
#         distance_bottom.append(last_postion1[index1]-last_postion2[i])
#     # print(np.min(distance_bottom),np.min(distance_top))
#     return np.min(distance_top),np.min(distance_bottom)



def find_original_position(resized_pixel, intervals):
    for (orig_start, orig_end, resized_start, resized_end) in intervals:
        if resized_start <= resized_pixel < resized_end:
            # Calculate the scaling factor for this segment
            scaling_factor = (orig_end - orig_start) / (resized_end - resized_start)
            # Calculate the original pixel position
            orig_pixel_float = orig_start + (resized_pixel - resized_start) * scaling_factor
            orig_pixel = int(max(orig_start, min(orig_pixel_float, orig_end - 1)))  # Ensure orig_pixel is within the interval
            return orig_pixel # Return original pixel and interval index
    return None  # Pixel not found in any segment



def inter_positions_new(arr):
    segments = []
    arr_unique = [arr[0]]
    for i in range(1,len(arr)):
        if arr[i] != arr[i-1]:
            segments.append(i)
            arr_unique.append(arr[i])

    segments.insert(0, 0)
    result_dict = {
        "segments": segments,
        "unique_values": arr_unique
    }
    return result_dict



def npy_correction_v3(img,total_rows,shape_top,shape_bottom):
    """
    Image corrections.
    
    Parameters:
    img: np array loaded from npy file.

    Returns:
    np.ma.core.MaskedArray: The corrected and masked image array.
    """
    dilation_top = 85
    # Step 1 : mask the image 
    mask = np.zeros(img.shape)
    mask[img >-30] = 1 
    top_mask = mask[:total_rows, :]
    bottom_mask = mask[total_rows:, :]
    dilated_bottom_mask = binary_dilation(bottom_mask, footprint = shape_bottom)
    dilated_top_mask = binary_dilation(top_mask, footprint = shape_top)
    mask[:total_rows, :] = dilated_top_mask
    mask[total_rows:, :] = dilated_bottom_mask
    masked_img = ma.array(img, mask = mask)

    # Step 2 : Calculate the median at the same depth level
    fond = np.ma.median(masked_img, axis = 1).reshape((masked_img.shape[0], 1))
    im_fond = np.repeat(fond, masked_img.shape[1], axis=1)

    # Step 3 : Get the corrected image
    corrected_img = 1 - masked_img/im_fond
    
    # Step 4 : Gaussian filtering
    tmp = skimage.filters.gaussian(corrected_img, sigma=0.3) 
    im = ma.array(tmp, mask = corrected_img.mask)
    return(im)



def parameters_correction(img,median_sea_depth,threshold = -30):
    sharp_wave = 160
    sharp_bottom = 20

    rows_matrix, columns_matrix = img.shape
    max_wave_height_top = []
    max_wave_height_bot = []
    for i in range(columns_matrix):
        arr = (img[:,i]>threshold).astype(int)
        # for the top
        min_value = np.min(arr)
        first_min = np.where(arr == min_value)[0][0]
        max_wave_height_top.append(first_min)

        # for the bottom
        reversed_index = np.where(arr[::-1] == min_value)[0][0]
        last_min = len(arr)- reversed_index
        max_wave_height_bot.append(last_min)
    
    if median_sea_depth < 25:
        total_lines = int(median_sea_depth*10-50)
    elif median_sea_depth < 35:
        total_lines = 200
    elif median_sea_depth < 40:
        if np.max(max_wave_height_top)<50:
            total_lines = 200
        else:
            total_lines = 300
    else:
        if np.max(max_wave_height_top)<50:
            total_lines = 200
        elif np.max(max_wave_height_top)<100:
            total_lines = 300
        else:
            total_lines = 350

    if np.max(max_wave_height_top) > sharp_wave:
        shape_top = rectangle(130,20)
        shape_top_desc = "rectangle(130, 20)"
    else:
        shape_top = rectangle(80,20)
        shape_top_desc = "rectangle(80,20)"

        
    if np.var(max_wave_height_bot) > sharp_bottom:
        shape_bottom = rectangle(20,10)
        shape_bottom_desc = "rectangle(20,10)"
    else:
        shape_bottom = rectangle(10,20)
        shape_bottom_desc = "rectangle(10,20)"

    return shape_top,shape_bottom,total_lines,shape_top_desc,shape_bottom_desc



def resize_matrix_31052024(matrix,velocity,meanvelocity):
    """
    matrice : matrix for a data.npy file
    velocity : the number of lines = the pixel count vertically in the matrice
    """

    # get the division positions and factors
    div = inter_positions_new(velocity)
    div_pix = div['segments']
    div_fac = div['unique_values']

    # Initialize an empty image with the same dimensions as the input image
    matrix_groupe = matrix[:, :0].copy()
    # matrix_groupe = np.zeros((matrix.shape[0], 0), dtype=matrix.dtype)
    mapping = []
    for i in range(len(div_pix)):
        # get the division position of pixels
        p_start = div_pix[i]
        p_end = div_pix[i + 1] if i + 1 < len(div_pix) else matrix.shape[1]
        
        # get the zoom factor fx, (fy=1)
        f_x = (div_fac[i] / meanvelocity)

        # chop the image divided by pixels
        left_matrix = matrix[:,p_start:p_end]
        if f_x > 1:
            matrix_chop =  rescale(left_matrix, (1,f_x), anti_aliasing=True, anti_aliasing_sigma=1)
        else:
            matrix_chop =  rescale(left_matrix, (1,f_x), anti_aliasing=True, anti_aliasing_sigma=1)
      
        mapping.append((p_start, p_end, matrix_groupe.shape[1],matrix_groupe.shape[1]+matrix_chop.shape[1]))
        # Group the image with the images gotten before
        matrix_groupe = np.concatenate((matrix_groupe, matrix_chop), axis=1)
    return matrix_groupe, mapping



def thresh_info_new20240806(dest_path,criteria_table,file,thresh_index,npy_path):  
    """
    Parameters:
    1 - dest_path: Path to save the output images and tables.
    2 - criteria_table: The table containing the information (thresh_max, thresh_min, and the mask used for the images, etc.) of all the images with 
        convergence = 1, 2.
    3 - file: The resized file name, ending with the format '.npy'.
    4 - thresh_index: Helps choose the thresh_min by different criteria. Here, (1: by criterion mean_c1, 2: mean_c2, 3: min_c1, 4: min_c2).

    Returns:
    1 - file_info: Information about fish schools (to be saved in a pkl file).
    2 - new_row_table: Information by files or by images (to be generated as a CSV table).
    3 - centroid_file: All the (weighted) centroids of fish schools in the image.
    """

    if file not in criteria_table['file'].values:
        print(f"File {file} not found in the criteria table.")
        return
    ## thresh_index : choosing different thresh_min by different critere, 1:mean_c1, 2:mean_c2 , 3:min_c1, 4:min_c2

    # Read values from criteria_table
    matched_row = criteria_table.loc[criteria_table['file'] == file]
    thresh_max = matched_row['thresh_max'].values[0]
    nbr_school = matched_row['nbr_school'].values[0]
    shape_top_desc = matched_row['shape_top_desc'].values[0]        # 1 : new mask
    shape_bottom_desc = matched_row['shape_bottom_desc'].values[0]
    first_lines = matched_row['total_rows'].values[0]
    
    if shape_top_desc == "rectangle(80,20)":
        shape_top = rectangle(80,20)
    else:
        shape_top= rectangle(130,20)

    if shape_bottom_desc == "rectangle(10,20)":
        shape_bottom = rectangle(10,20)
    else:
        shape_bottom = rectangle(20,10)

    if thresh_index == 1:
        thresh_min = matched_row['mean_c1'].values[0]
    elif thresh_index == 2:
        thresh_min = matched_row['mean_c2'].values[0]
    elif thresh_index == 3:
        thresh_min = matched_row['min_c1'].values[0]
    elif thresh_index == 4:
        thresh_min = matched_row['min_c2'].values[0]
    else :
        print("Choose the right critere: 1 for mean_c1, 2 for mean_c2, 3 for min_c1,4 for min_c2")
        return
    
    img = np.load(os.path.join(npy_path,file))    
    width_image = img.shape[1]
    height_image = img.shape[0]
    
    # Get the corrected image  
    im = npy_correction_v3(img,first_lines,shape_top,shape_bottom)    # 1 : top_shape + bottom_shape

    regions_max = np.digitize(im, bins=[thresh_max])
    regions_max[im.mask] = 0
    region_min = np.digitize(im, bins=[thresh_min])
    region_min[im.mask] = 0
        
    im_recon = reconstruction(regions_max, region_min, method='dilation')
    label_img = label(im_recon)
    regions_all = regionprops(label_img,intensity_image=img)
    # regions_all = regionprops(label_img)                       #            change!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    regions = [region for region in regions_all if region.area > 1]

    # Initialising the output variables
    label_file = []     # class 1
    bbox_file = []       
    width_length_file = []  
    axis_ellipse_file = []  #  major , minor
    perimeter_file = []
    size_file = []      
    is_very_wide_s = []
    is_very_tall_s = []
    dis_to_surface_s = []   

    intensity_school = []    # class 2 
    intensity_img = [np.mean(img[img!=0]),np.min(img),np.max(img[img!=0])]
    dif_intensity_s_i = []
    min_intensity = []

    center_square_intensity_file = []    
    edges_school_intensity_file = [] 
    dif_intensity_center_edges_file  = []
    std_intensity_school_file  = []
    gradient_school_file = []    
    gradient_school_center_file = []
    gradient_school_edges_file = []
    dif_gradient_center_edges_file = []

    width_length_ratio_file = []    # class 3   elongation 
    axis_ellipse_ratio_file = []  #  short / long   eccentricity 
    solidity_file = []
    compactness_file = []
    inertia_tensor_eigvals_ratio_file = []   
    perimeter_area_ratio_file = []

    centroid_file = []    # file class 4
    centroid_weighted_file = []
    depth_file = []     
    coords_file = []  

    edges_school_file = []  # file class 5
    center_square_file = []
    min_intensity_file = []

    plt.imsave(f'{dest_path}/{file[:-4]}_double.png',im_recon)

    for region in  regions:
        coords = region.coords
        
        y_coords = coords[:, 0]
        x_coords = coords[:, 1]
        bbox = region.bbox

        dis_to_surface = bbox[0]
        is_very_wide = 1 if (bbox[3]-bbox[1])>width_image/2 else 0
        is_very_tall = 1 if (bbox[2]-bbox[0])>height_image/2 else 0

        # Calculate the average coordinates
        avg_0 = sum(coord[0] for coord in coords) / len(coords)
        avg_1 = sum(coord[1] for coord in coords) / len(coords)
        # center_file.append([avg_0,avg_1])

        intensities = [img[coord[0], coord[1]] for coord in coords]
        max_index = np.argmax(intensities)
        max_intensity_coords = coords[max_index]
        
        min_intensity = np.min(intensities)

        edges_school = find_edges(im_recon, y_coords, x_coords)
        center_square = find_center_square(max_intensity_coords, region.coords)  # use max_intensity_coords not center
        center_square_intensity =  np.mean([img[coord[0], coord[1]] for coord in center_square])
        edges_school_intensity =  np.mean([img[coord[0], coord[1]] for coord in edges_school])
        dif_intensity_center_edges = center_square_intensity - edges_school_intensity
        std_intensity_school = np.std([img[coord[0], coord[1]] for coord in coords])

        gradient_y, gradient_x = np.gradient(coords)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        gradient_school = np.mean(gradient_magnitude)
        gradient_y_center, gradient_x_center = np.gradient(center_square)
        gradient_magnitude_center = np.sqrt(gradient_x_center**2 + gradient_y_center**2)
        gradient_school_center = np.mean(gradient_magnitude_center)
        gradient_y_edges, gradient_x_edges = np.gradient(edges_school)
        gradient_magnitude_edges = np.sqrt(gradient_x_edges**2 + gradient_y_edges**2)
        gradient_school_edges = np.mean(gradient_magnitude_edges)
        dif_gradient_center_edges = abs(gradient_school_center - gradient_school_edges)

        centroid_file.append(region.centroid)
        centroid_weighted_file.append(region.centroid_weighted)
        label_file.append(region.label)
        bbox_file.append(region.bbox)
        width_length_file.append([region.bbox[3]-region.bbox[1],region.bbox[2]-region.bbox[0]])
        axis_ellipse_file.append([region.axis_major_length,region.axis_minor_length])
        perimeter_file.append(region.perimeter)
        size_file.append(region.area)
        is_very_wide_s.append(is_very_wide)
        is_very_tall_s.append(is_very_tall)
        dis_to_surface_s.append(dis_to_surface)
        intensity_school.append([np.mean(img[y_coords, x_coords]),np.min(img[y_coords, x_coords]),np.max(img[y_coords, x_coords])])
        dif_intensity_s_i.append(np.mean(intensities) - intensity_img[0])

        center_square_intensity_file.append(center_square_intensity)
        edges_school_intensity_file.append(edges_school_intensity)
        dif_intensity_center_edges_file.append(dif_intensity_center_edges)
        std_intensity_school_file.append(std_intensity_school)
        gradient_school_file.append(gradient_school)    
        gradient_school_center_file.append(gradient_school_center)
        gradient_school_edges_file.append(gradient_school_edges)
        dif_gradient_center_edges_file.append(dif_gradient_center_edges)
        width_length_ratio_file.append((region.bbox[3]-region.bbox[1])/(region.bbox[2]-region.bbox[0]))
    
        axis_ellipse_ratio = region.axis_minor_length / region.axis_major_length
      
        
        axis_ellipse_ratio_file.append(axis_ellipse_ratio)
        solidity_file.append(region.solidity)
        compactness_file.append(4*math.pi*(region.area)/(region.perimeter**2))
        inertia_tensor_eigvals_ratio_file.append(region.inertia_tensor_eigvals[1]/region.inertia_tensor_eigvals[0])
        perimeter_area_ratio_file.append(region.perimeter/region.area)

        coords_file.append(region.coords)
        depth_file.append(avg_0/10)
        edges_school_file.append(edges_school)
        center_square_file.append(center_square)
        min_intensity_file.append(min_intensity)

    y_all = np.dot(size_file, [coord[0] for coord in centroid_file]) / np.sum(size_file)
    x_all = np.dot(size_file, [coord[1] for coord in centroid_file]) / np.sum(size_file)
    
    if not intensity_school:
        print("intensity school :",intensity_school)
        print(file)                       
    intensity_school_array = np.vstack(intensity_school)
    mean_intensity_school = np.dot(size_file, intensity_school_array[:,0])/np.sum(size_file)
    
    
    file_info = {"label":label_file,
                "size":size_file,
                "bbox":bbox_file,
                "depth":depth_file,
                "center":centroid_file,
                "width_length":width_length_file,
                "is_very_wide":is_very_wide_s,
                "is_very_tall":is_very_tall_s,
                'dis_to_surface':dis_to_surface_s,
                "axis_ellipse": axis_ellipse_file,
                "perimeter_school": perimeter_file,
                "intensity_school":intensity_school,
                "dif_intensity_school_image": dif_intensity_s_i,
                "intensity_img":intensity_img,
                "center_square_intensity":center_square_intensity_file, 
                "edges_school_intensity":edges_school_intensity_file,
                "dif_intensity_center_edges":dif_intensity_center_edges_file,
                "std_intensity_school":std_intensity_school_file,
                "gradient_school":gradient_school_file,
                "gradient_school_center":gradient_school_center_file,
                "gradient_school_edges":gradient_school_edges_file,
                "dif_gradient_center_edges":dif_gradient_center_edges_file,
                "width_length_ratio":width_length_ratio_file,
                "axis_ellipse_ratio":axis_ellipse_ratio_file,
                "solidity":solidity_file,
                "compactness":compactness_file,
                "inertia_tensor_eigvals_ratio":inertia_tensor_eigvals_ratio_file,
                "perimeter_area_ratio":perimeter_area_ratio_file,
                "coords":coords_file,
                "nbr_school":len(label_file),
                "total_area":sum(size_file),
                "mean_intensity_school":mean_intensity_school,
                "center_all":[y_all,x_all],     #[row,column]
                "mean_depth":y_all/10,
                "thresh_max":thresh_max,
                "thresh_min":thresh_min,
                'min_intensity': min_intensity_file}
    new_row_table = {'file':file, 
                'nbr_school':len(label_file), 
                'thresh_max':thresh_max, 
                'thresh_min':thresh_min,
                'all_size':sum(size_file),
                'mean_intensity_school':mean_intensity_school,
                'mean_depth':y_all/10,
                'mean_intensity_imgwithout0':np.mean(img[img!=0]),
                'min_intensity': np.min(min_intensity_file)} 
    return file_info ,new_row_table,centroid_file


##############################################################################################
##############                       version time  20250702                     ##############
##############################################################################################
def sam_info(file,sam_result,npy_path,csv_path,dest_path):    # version time : 20240806
    # Step 1 : change the file name to get the corresponding name in sam result
    file_npy =file[:-4]+'.npy'

    # Step 2 : load the image from original directionary + mask
    img = np.load(os.path.join(npy_path,file_npy))
    csv_name = file.replace('_new.png','.csv')
    csv_table = pd.read_csv(os.path.join(csv_path,csv_name))
    bottom = int(csv_table['depth'].median()*10)
   
    length_limit = bottom *0.62
    size_limit = bottom*img.shape[1]*0.5            #  2 : bigger than before = 0.42
    nomal_limit = bottom*img.shape[1]*0.1

    # Step 3 : find the corrections parameters
    shape_top,shape_bottom,first_lines,shape_top_desc,shape_bottom_desc = parameters_correction(img,bottom,threshold = -30)
   
    # Step 4 : get the mask 

    im = npy_correction_v3(img,first_lines,shape_top,shape_bottom)           # 1 : new mask
    mask = im.mask
    
    dis_limit = 10
    dis_check = 50 # width limit
    width_image = img.shape[1]
    height_image = img.shape[0]

    # Step 5 : Initialising the output variables
    label_file = []     # class 1
    bbox_file = []       
    width_length_file = []  
    axis_ellipse_file = []  #  major , minor
    perimeter_file = []
    size_file = []      
    is_very_wide_s = []
    is_very_tall_s = [] 
    dis_level_s = []  
    dis_to_surface_s = []  
    dis_to_bottom_s = []   

    intensity_school = []    # class 2 
    intensity_img = [np.mean(img[img!=0]),np.min(img),np.max(img[img!=0])]
    intensity_c_img_mask = np.mean(im[~mask])                                       # 2: use corrected img intensity

    dif_intensity_s_i = []
    center_square_intensity_file = []    
    edges_school_intensity_file = [] 
    dif_intensity_center_edges_file  = []
    std_intensity_school_file  = []
    gradient_school_file = []    
    gradient_school_center_file = []
    gradient_school_edges_file = []
    dif_gradient_center_edges_file = []
    
    width_length_ratio_file = []    # class 3   elongation 
    axis_ellipse_ratio_file = []  #  short / long   eccentricity 
    solidity_file = []
    compactness_file = []
    inertia_tensor_eigvals_ratio_file = []    
    perimeter_area_ratio_file = []

    centroid_file = []    # file class 4
    depth_file = []     
    coords_file = []  

    edges_school_file = []  # file class 5
    center_square_file = []

    # Step 6 : Read the info from the sam pkl file
    seg_new = sam_result[file]
    segmentations = seg_new['segmentation']
    bboxs = seg_new['bbox']
    areas = seg_new['area']
    if len(segmentations)>0 :
        binary_matrice = np.zeros((np.array(segmentations[0])).shape)

        # Step 5 : Get the union of segmentations of fish school
        for j in range(len(segmentations)):
            seg = segmentations[j]
            intensity_c_seg = np.mean(im[seg])                                      # 2: use corrected seg intensity

            bbox = bboxs[j]
            # Check if there are any common positions
            common_positions = np.logical_and(mask, seg)
            intersections_mask = np.any(common_positions)

            # Criterion 1
            if (not intersections_mask) and (intensity_c_seg>intensity_c_img_mask) and (areas[j]<size_limit) and (bbox[3]<length_limit):   # 2 : big or small both use the same intensity limit
                    binary_matrice[seg] = 1
                                  
        label_img = label(binary_matrice)
        regions = regionprops(label_img)
        
        # Step 6 : Get the informations from the fish school detected
        if len(regions) > 0 : 
            plt.imsave(f'{dest_path}\{file[:-4]}.png', binary_matrice, cmap='gray')
            for region in  regions:
                if region.area > 1 :
                    coords = region.coords
                    y_coords = coords[:, 0]
                    x_coords = coords[:, 1]
                    bbox = region.bbox
                    dis_to_bottom = bottom - bbox[1]-bbox[3]/2
                    dis_to_surface = bbox[1]
                    is_very_wide = 1 if (bbox[3]-bbox[1])>width_image/2 else 0
                    is_very_tall = 1 if (bbox[2]-bbox[0])>height_image/2 else 0
                    
                    if dis_to_surface > dis_check and dis_to_bottom > dis_check:
                        dis_level = 2
                    elif dis_to_bottom < dis_limit or dis_to_surface <dis_limit:
                        dis_level = 0
                    else:
                        dis_level = 1
                    # Calculate the average coordinates
                    avg_0 = sum(coord[0] for coord in coords) / len(coords)
                    avg_1 = sum(coord[1] for coord in coords) / len(coords)

                    intensities = [img[coord[0], coord[1]] for coord in coords]
                    max_index = np.argmax(intensities)
                    max_intensity_coords = coords[max_index]
                
                    edges_school = find_edges(binary_matrice, y_coords, x_coords)
                    center_square = find_center_square(max_intensity_coords, region.coords)  # use max_intensity_coords not center
                    center_square_intensity =  np.mean([img[coord[0], coord[1]] for coord in center_square])
                    edges_school_intensity =  np.mean([img[coord[0], coord[1]] for coord in edges_school])
                    dif_intensity_center_edges = center_square_intensity - edges_school_intensity
                    std_intensity_school = np.std([img[coord[0], coord[1]] for coord in coords])
                
                    gradient_y, gradient_x = np.gradient(coords)
                    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
                    gradient_school = np.mean(gradient_magnitude)
                    gradient_y_center, gradient_x_center = np.gradient(center_square)
                    gradient_magnitude_center = np.sqrt(gradient_x_center**2 + gradient_y_center**2)
                    gradient_school_center = np.mean(gradient_magnitude_center)
                    gradient_y_edges, gradient_x_edges = np.gradient(edges_school)
                    gradient_magnitude_edges = np.sqrt(gradient_x_edges**2 + gradient_y_edges**2)
                    gradient_school_edges = np.mean(gradient_magnitude_edges)
                    dif_gradient_center_edges = abs(gradient_school_center - gradient_school_edges)
        
                    centroid_file.append(region.centroid)
                    label_file.append(region.label)
                    bbox_file.append(region.bbox)
                    width_length_file.append([region.bbox[3]-region.bbox[1],region.bbox[2]-region.bbox[0]])
                    axis_ellipse_file.append([region.axis_major_length,region.axis_minor_length])
                    perimeter_file.append(region.perimeter)
                    size_file.append(region.area)
                    is_very_wide_s.append(is_very_wide)
                    is_very_tall_s.append(is_very_tall)
                    dis_level_s.append(dis_level)
                    dis_to_surface_s.append(dis_to_surface)
                    dis_to_bottom_s.append(dis_to_bottom)

                    intensity_school.append([np.mean(img[y_coords, x_coords]),np.min(img[y_coords, x_coords]),np.max(img[y_coords, x_coords])])
                    dif_intensity_s_i.append(np.mean(intensities) - intensity_img[0])
                    
                    center_square_intensity_file.append(center_square_intensity)
                    edges_school_intensity_file.append(edges_school_intensity)
                    dif_intensity_center_edges_file.append(dif_intensity_center_edges)
                    std_intensity_school_file.append(std_intensity_school)
                    gradient_school_file.append(gradient_school)    
                    gradient_school_center_file.append(gradient_school_center)
                    gradient_school_edges_file.append(gradient_school_edges)
                    dif_gradient_center_edges_file.append(dif_gradient_center_edges)

                    width_length_ratio_file.append((region.bbox[3]-region.bbox[1])/(region.bbox[2]-region.bbox[0]))
                    axis_ellipse_ratio_file.append(region.axis_minor_length/region.axis_major_length)
                    solidity_file.append(region.solidity)
                    compactness_file.append(4*math.pi*(region.area)/(region.perimeter**2))
                    inertia_tensor_eigvals_ratio_file.append(region.inertia_tensor_eigvals[1]/region.inertia_tensor_eigvals[0])
                    perimeter_area_ratio_file.append(region.perimeter/region.area)
            
                    coords_file.append(region.coords)
                    depth_file.append(avg_0/10)
                    edges_school_file.append(edges_school)
                    center_square_file.append(center_square)

            y_all = np.dot(size_file, [coord[0] for coord in centroid_file]) / np.sum(size_file)
            x_all = np.dot(size_file, [coord[1] for coord in centroid_file]) / np.sum(size_file)
            intensity_school_array = np.vstack(intensity_school)
            mean_intensity_school = np.dot(size_file, intensity_school_array[:,0])/np.sum(size_file)
            
            file_info = {"label":label_file,
                        "size":size_file,
                        "bbox":bbox_file,
                        "depth":depth_file,
                        "center":centroid_file,
                        "width_length":width_length_file,
                        "is_very_wide":is_very_wide_s,
                        "is_very_tall":is_very_tall_s,
                        'dis_to_surface':dis_to_surface_s,
                        'dis_to_bottom':dis_to_bottom_s,
                        'dis_level':dis_level_s,
                        "axis_ellipse": axis_ellipse_file,
                        "perimeter_school": perimeter_file,
                        "intensity_school":intensity_school,
                        "dif_intensity_school_image": dif_intensity_s_i,
                        "intensity_img":intensity_img,
                        "center_square_intensity":center_square_intensity_file, 
                        "edges_school_intensity":edges_school_intensity_file,
                        "dif_intensity_center_edges":dif_intensity_center_edges_file,
                        "std_intensity_school":std_intensity_school_file,
                        "gradient_school":gradient_school_file,
                        "gradient_school_center":gradient_school_center_file,
                        "gradient_school_edges":gradient_school_edges_file,
                        "dif_gradient_center_edges":dif_gradient_center_edges_file,
                        "width_length_ratio":width_length_ratio_file,
                        "axis_ellipse_ratio":axis_ellipse_ratio_file,
                        "solidity":solidity_file,
                        "compactness":compactness_file,
                        "inertia_tensor_eigvals_ratio":inertia_tensor_eigvals_ratio_file,
                        "perimeter_area_ratio":perimeter_area_ratio_file,
                        "coords":coords_file,
                        "nbr_school":len(label_file),
                        "total_area":sum(size_file),
                        "mean_intensity_school":mean_intensity_school,
                        "center_all":[y_all,x_all],     #[row,column]
                        "mean_depth":y_all/10}
            new_row_table = {'file':file, 
                        'nbr_school':len(label_file), 
                        'all_size':sum(size_file),
                        'mean_intensity_school':mean_intensity_school,
                        'mean_depth':y_all/10,
                        'mean_intensity_imgwithout0':np.mean(img[img!=0])} 

            return file_info ,new_row_table,centroid_file
        else:
            return None, None, None
    else:
        return None, None, None


##############################################################################################
##############                       version time  20250702                     ##############
##############################################################################################
def sam_info(file,sam_result,npy_path,csv_path,dest_path):    # version time : 20240806
    # Step 1 : change the file name to get the corresponding name in sam result
    file_npy =file[:-4]+'.npy'

    # Step 2 : load the image from original directionary + mask
    img = np.load(os.path.join(npy_path,file_npy))
    csv_name = file.replace('_new.png','.csv')
    csv_table = pd.read_csv(os.path.join(csv_path,csv_name))
    bottom = int(csv_table['depth'].median()*10)
   
    length_limit = bottom *0.62
    size_limit = bottom*img.shape[1]*0.5            #  2 : bigger than before = 0.42
    nomal_limit = bottom*img.shape[1]*0.1

    # Step 3 : find the corrections parameters
    shape_top,shape_bottom,first_lines,shape_top_desc,shape_bottom_desc = parameters_correction(img,bottom,threshold = -30)
   
    # Step 4 : get the mask 

    im = npy_correction_v3(img,first_lines,shape_top,shape_bottom)           # 1 : new mask
    mask = im.mask
    
    dis_limit = 10
    dis_check = 50 # width limit
    width_image = img.shape[1]
    height_image = img.shape[0]

    # Step 5 : Initialising the output variables
    label_file = []     # class 1
    bbox_file = []       
    width_length_file = []  
    axis_ellipse_file = []  #  major , minor
    perimeter_file = []
    size_file = []      
    is_very_wide_s = []
    is_very_tall_s = [] 
    dis_level_s = []  
    dis_to_surface_s = []  
    dis_to_bottom_s = []   

    intensity_school = []    # class 2 
    intensity_img = [np.mean(img[img!=0]),np.min(img),np.max(img[img!=0])]
    intensity_c_img_mask = np.mean(im[~mask])                                       # 2: use corrected img intensity

    dif_intensity_s_i = []
    center_square_intensity_file = []    
    edges_school_intensity_file = [] 
    dif_intensity_center_edges_file  = []
    std_intensity_school_file  = []
    gradient_school_file = []    
    gradient_school_center_file = []
    gradient_school_edges_file = []
    dif_gradient_center_edges_file = []
    
    width_length_ratio_file = []    # class 3   elongation 
    axis_ellipse_ratio_file = []  #  short / long   eccentricity 
    solidity_file = []
    compactness_file = []
    inertia_tensor_eigvals_ratio_file = []    
    perimeter_area_ratio_file = []

    centroid_file = []    # file class 4
    depth_file = []     
    coords_file = []  

    edges_school_file = []  # file class 5
    center_square_file = []

    # Step 6 : Read the info from the sam pkl file
    seg_new = sam_result[file]
    segmentations = seg_new['segmentation']
    bboxs = seg_new['bbox']
    areas = seg_new['area']
    if len(segmentations)>0 :
        binary_matrice = np.zeros((np.array(segmentations[0])).shape)

        # Step 5 : Get the union of segmentations of fish school
        for j in range(len(segmentations)):
            seg = segmentations[j]
            intensity_c_seg = np.mean(im[seg])                                      # 2: use corrected seg intensity

            bbox = bboxs[j]
            # Check if there are any common positions
            common_positions = np.logical_and(mask, seg)
            intersections_mask = np.any(common_positions)

            # Criterion 1
            if (not intersections_mask) and (intensity_c_seg>intensity_c_img_mask) and (areas[j]<size_limit) and (bbox[3]<length_limit):   # 2 : big or small both use the same intensity limit
                    binary_matrice[seg] = 1
                                  
        label_img = label(binary_matrice)
        regions = regionprops(label_img)
        
        # Step 6 : Get the informations from the fish school detected
        if len(regions) > 0 : 
            plt.imsave(f'{dest_path}\{file[:-4]}.png', binary_matrice, cmap='gray')
            for region in  regions:
                if region.area > 1 :
                    coords = region.coords
                    y_coords = coords[:, 0]
                    x_coords = coords[:, 1]
                    bbox = region.bbox
                    dis_to_bottom = bottom - bbox[1]-bbox[3]/2
                    dis_to_surface = bbox[1]
                    is_very_wide = 1 if (bbox[3]-bbox[1])>width_image/2 else 0
                    is_very_tall = 1 if (bbox[2]-bbox[0])>height_image/2 else 0
                    
                    if dis_to_surface > dis_check and dis_to_bottom > dis_check:
                        dis_level = 2
                    elif dis_to_bottom < dis_limit or dis_to_surface <dis_limit:
                        dis_level = 0
                    else:
                        dis_level = 1
                    # Calculate the average coordinates
                    avg_0 = sum(coord[0] for coord in coords) / len(coords)
                    avg_1 = sum(coord[1] for coord in coords) / len(coords)

                    intensities = [img[coord[0], coord[1]] for coord in coords]
                    max_index = np.argmax(intensities)
                    max_intensity_coords = coords[max_index]
                
                    edges_school = find_edges(binary_matrice, y_coords, x_coords)
                    center_square = find_center_square(max_intensity_coords, region.coords)  # use max_intensity_coords not center
                    center_square_intensity =  np.mean([img[coord[0], coord[1]] for coord in center_square])
                    edges_school_intensity =  np.mean([img[coord[0], coord[1]] for coord in edges_school])
                    dif_intensity_center_edges = center_square_intensity - edges_school_intensity
                    std_intensity_school = np.std([img[coord[0], coord[1]] for coord in coords])
                
                    gradient_y, gradient_x = np.gradient(coords)
                    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
                    gradient_school = np.mean(gradient_magnitude)
                    gradient_y_center, gradient_x_center = np.gradient(center_square)
                    gradient_magnitude_center = np.sqrt(gradient_x_center**2 + gradient_y_center**2)
                    gradient_school_center = np.mean(gradient_magnitude_center)
                    gradient_y_edges, gradient_x_edges = np.gradient(edges_school)
                    gradient_magnitude_edges = np.sqrt(gradient_x_edges**2 + gradient_y_edges**2)
                    gradient_school_edges = np.mean(gradient_magnitude_edges)
                    dif_gradient_center_edges = abs(gradient_school_center - gradient_school_edges)
        
                    centroid_file.append(region.centroid)
                    label_file.append(region.label)
                    bbox_file.append(region.bbox)
                    width_length_file.append([region.bbox[3]-region.bbox[1],region.bbox[2]-region.bbox[0]])
                    axis_ellipse_file.append([region.axis_major_length,region.axis_minor_length])
                    perimeter_file.append(region.perimeter)
                    size_file.append(region.area)
                    is_very_wide_s.append(is_very_wide)
                    is_very_tall_s.append(is_very_tall)
                    dis_level_s.append(dis_level)
                    dis_to_surface_s.append(dis_to_surface)
                    dis_to_bottom_s.append(dis_to_bottom)

                    intensity_school.append([np.mean(img[y_coords, x_coords]),np.min(img[y_coords, x_coords]),np.max(img[y_coords, x_coords])])
                    dif_intensity_s_i.append(np.mean(intensities) - intensity_img[0])
                    
                    center_square_intensity_file.append(center_square_intensity)
                    edges_school_intensity_file.append(edges_school_intensity)
                    dif_intensity_center_edges_file.append(dif_intensity_center_edges)
                    std_intensity_school_file.append(std_intensity_school)
                    gradient_school_file.append(gradient_school)    
                    gradient_school_center_file.append(gradient_school_center)
                    gradient_school_edges_file.append(gradient_school_edges)
                    dif_gradient_center_edges_file.append(dif_gradient_center_edges)

                    width_length_ratio_file.append((region.bbox[3]-region.bbox[1])/(region.bbox[2]-region.bbox[0]))
                    axis_ellipse_ratio_file.append(region.axis_minor_length/region.axis_major_length)
                    solidity_file.append(region.solidity)
                    compactness_file.append(4*math.pi*(region.area)/(region.perimeter**2))
                    inertia_tensor_eigvals_ratio_file.append(region.inertia_tensor_eigvals[1]/region.inertia_tensor_eigvals[0])
                    perimeter_area_ratio_file.append(region.perimeter/region.area)
            
                    coords_file.append(region.coords)
                    depth_file.append(avg_0/10)
                    edges_school_file.append(edges_school)
                    center_square_file.append(center_square)

            y_all = np.dot(size_file, [coord[0] for coord in centroid_file]) / np.sum(size_file)
            x_all = np.dot(size_file, [coord[1] for coord in centroid_file]) / np.sum(size_file)
            intensity_school_array = np.vstack(intensity_school)
            mean_intensity_school = np.dot(size_file, intensity_school_array[:,0])/np.sum(size_file)
            
            file_info = {"label":label_file,
                        "size":size_file,
                        "bbox":bbox_file,
                        "depth":depth_file,
                        "center":centroid_file,
                        "width_length":width_length_file,
                        "is_very_wide":is_very_wide_s,
                        "is_very_tall":is_very_tall_s,
                        'dis_to_surface':dis_to_surface_s,
                        'dis_to_bottom':dis_to_bottom_s,
                        'dis_level':dis_level_s,
                        "axis_ellipse": axis_ellipse_file,
                        "perimeter_school": perimeter_file,
                        "intensity_school":intensity_school,
                        "dif_intensity_school_image": dif_intensity_s_i,
                        "intensity_img":intensity_img,
                        "center_square_intensity":center_square_intensity_file, 
                        "edges_school_intensity":edges_school_intensity_file,
                        "dif_intensity_center_edges":dif_intensity_center_edges_file,
                        "std_intensity_school":std_intensity_school_file,
                        "gradient_school":gradient_school_file,
                        "gradient_school_center":gradient_school_center_file,
                        "gradient_school_edges":gradient_school_edges_file,
                        "dif_gradient_center_edges":dif_gradient_center_edges_file,
                        "width_length_ratio":width_length_ratio_file,
                        "axis_ellipse_ratio":axis_ellipse_ratio_file,
                        "solidity":solidity_file,
                        "compactness":compactness_file,
                        "inertia_tensor_eigvals_ratio":inertia_tensor_eigvals_ratio_file,
                        "perimeter_area_ratio":perimeter_area_ratio_file,
                        "coords":coords_file,
                        "nbr_school":len(label_file),
                        "total_area":sum(size_file),
                        "mean_intensity_school":mean_intensity_school,
                        "center_all":[y_all,x_all],     #[row,column]
                        "mean_depth":y_all/10}
            new_row_table = {'file':file, 
                        'nbr_school':len(label_file), 
                        'all_size':sum(size_file),
                        'mean_intensity_school':mean_intensity_school,
                        'mean_depth':y_all/10,
                        'mean_intensity_imgwithout0':np.mean(img[img!=0])} 

            return file_info ,new_row_table,centroid_file
        else:
            return None, None, None
    else:
        return None, None, None
