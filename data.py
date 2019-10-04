import numpy as np
import matplotlib.pyplot as plt
import os,glob

import cv2
import skimage

import pydicom, nrrd, nibabel
import SimpleITK as sitk

import scipy.ndimage
from skimage import measure, morphology
from scipy.ndimage.morphology import binary_dilation,generate_binary_structure
from skimage.morphology import convex_hull_image

import cv2


def load_dicom(path):
    '''
        prend le path du repertoire contenant les images dicom et renvoie :
            . le scan en unité HU numpy array de shape (Z, Y, X)
            . l'espacement en mm entre pixels dans les 3 axes array([Z, Y, X])
    '''
    '''
    dicoms = os.listdir(path)

    
    # Get Metadata
    dcm = pydicom.dcmread(path + '/' + dicoms[10])
    intercept = np.int16(dcm.RescaleIntercept)
    slope = np.float64(dcm.RescaleSlope)
    
    
    try:
        
        try:
            
            image = []
            for d in dicoms:
                try:
                    s = pydicom.dcmread(path + '/' + d)
                    test = s.ImagePositionPatient[2]
                    image.append(s)
                except:
                    continue
    
            
            image.sort(key = lambda x: int(x.ImagePositionPatient[2]))
            
            st = np.abs(image[0].ImagePositionPatient[2] - image[1].ImagePositionPatient[2])
          
            try:
                ps = dcm.PixelSpacing
            except:
                print("rare error")

            spacing = np.array((np.float64(st),np.float64(ps[1]),np.float64(ps[0]))) 

            image = [s.pixel_array for s in image]
            image = np.array(image).astype(np.int16)
            
            # Convertion en HU
            for i in range(image.shape[0]):
                image[i] = slope * image[i].astype(np.float64)
                image[i] = image[i].astype(np.int16)    
                image[i] += intercept

        except:  
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(path)
            reader.SetFileNames(dicom_names)
            image = reader.Execute()

            sp = image.GetSpacing()
            st = sp[2]
            ps = [sp[1],sp[0]]

            spacing = np.array((np.float64(st),np.float64(ps[1]),np.float64(ps[0]))) 
            image = sitk.GetArrayFromImage(image).astype(np.int16)
    except:
        print(path)

         '''
    
    dicoms = os.listdir(path)
    
    if dicoms[0][0] == '.':
        os.system("rm ./temp/a*")
        for d in dicoms:
            os.system("cp "+path+d+" ./temp/a"+d)
        path = "./temp/"

    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()

    sp = image.GetSpacing()
    st = sp[2]
    ps = [sp[1],sp[0]]

    spacing = np.array((np.float64(st),np.float64(ps[1]),np.float64(ps[0]))) 
    image = sitk.GetArrayFromImage(image).astype(np.int16)
       
    
    return image, spacing
    

def resample(image, spacing, new_spacing):
    '''
        image : 3D np.array (Z,Y,X)    
        
        spacing et new_spacing : 1D np.array [Z,Y,X]
        
    '''
    
    # on calcule le nouveau shape de l'image pour avoir le nouveau spacing
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape).astype(np.int16)

    
    # Avec skimage 
    image =  skimage.transform.resize(image, new_shape, order=1, clip=True, mode='edge')
    
    return image





################################################################# Optimize or change below : 
def preprocess(img, spacing):
    bw = binarize_per_slice(img, spacing)
    flag = 0
    cut_num = 0
    cut_step = 2
    bw0 = np.copy(bw)
    while flag == 0 and cut_num < bw.shape[0] and np.max(bw) > 0:
        bw = np.copy(bw0)
        bw, flag = all_slice_analysis(bw, spacing, cut_num=cut_num, vol_limit=[0.68,16])
        if np.max(bw) == 0 and cut_num == 0:
            return 1
        cut_num = cut_num + cut_step
        
    
    bw = fill_hole(bw)
    bw1, bw2, bw = two_lung_only(bw, spacing)

    Mask = bw1+bw2
    

    dm1 = process_mask(bw1)
    dm2 = process_mask(bw2)
    dilatedMask = dm1+dm2
    return dilatedMask

    extramask = dilatedMask ^ Mask
    bone_thresh = 210
    pad_value = 170

    img[np.isnan(img)]=-2000
    sliceim = lumTrans(img)
    sliceim = sliceim*dilatedMask +pad_value*(1-dilatedMask).astype('uint8')
    bones = sliceim*extramask>bone_thresh
    sliceim[bones] = pad_value
    
    return sliceim

def binarize_per_slice(image, spacing, intensity_th=-600, sigma=1, area_th=30, eccen_th=0.99, bg_patch_size=10):
    bw = np.zeros(image.shape, dtype=bool)
    
    # prepare a mask, with all corner values set to nan
    image_size = image.shape[1]
    grid_axis = np.linspace(-image_size/2+0.5, image_size/2-0.5, image_size)
    x, y = np.meshgrid(grid_axis, grid_axis)
    d = (x**2+y**2)**0.5
    nan_mask = (d<image_size/2).astype(float)
    nan_mask[nan_mask == 0] = np.nan
    for i in range(image.shape[0]):
        # Check if corner pixels are identical, if so the slice  before Gaussian filtering
        if len(np.unique(image[i, 0:bg_patch_size, 0:bg_patch_size])) == 1:
            current_bw = scipy.ndimage.filters.gaussian_filter(np.multiply(image[i].astype('float32'), nan_mask), sigma, truncate=2.0) < intensity_th
            #current_bw = cv2.GaussianBlur(np.multiply(image[i].astype('float32'), nan_mask), (0,0),sigmaX = sigma) < intensity_th
        else:
            current_bw = scipy.ndimage.filters.gaussian_filter(image[i].astype('float32'), sigma, truncate=2.0) < intensity_th
            #current_bw = cv2.GaussianBlur(image[i].astype('float32'), (0,0),sigmaX = sigma) < intensity_th
        
        # select proper components
        label = measure.label(current_bw)
        properties = measure.regionprops(label)
        valid_label = set()
        for prop in properties:
            if prop.area * spacing[1] * spacing[2] > area_th and prop.eccentricity < eccen_th:
                valid_label.add(prop.label)
        current_bw = np.in1d(label, list(valid_label)).reshape(label.shape)
        bw[i] = current_bw
        
    return bw


def all_slice_analysis(bw, spacing, cut_num=0, vol_limit=[0.5, 13], area_th=6e3, dist_th=62): #vol lim 0.68 and 8.2
    
    # Essai : ########################
    ''' 
    lab, n  = measure.label(bw[-1-cut_num], connectivity=1,return_num=True)
    

    for i in range(1,n+1):
        x = np.copy(lab)
        x[x != i] = 0
        x[x == i] = 1
        if np.sum(x) < 1000:
            lab[lab == i] = 0
    lab[lab > 0] = 1
            
    bw[-1-cut_num] = np.multiply(bw[-1-cut_num],lab)
    '''
    ##################################
    
    # in some cases, several top layers need to be removed first    
    if cut_num > 0:
        bw0 = np.copy(bw)
        bw[-cut_num:] = False
    label = measure.label(bw, connectivity=1)
    # remove components access to corners
    mid = int(label.shape[2] / 2)
    bg_label = set([label[0, 0, 0], label[0, 0, -1], label[0, -1, 0], label[0, -1, -1], \
                    label[-1-cut_num, 0, 0], label[-1-cut_num, 0, -1], label[-1-cut_num, -1, 0], label[-1-cut_num, -1, -1], \
                    label[0, 0, mid], label[0, -1, mid], label[-1-cut_num, 0, mid], label[-1-cut_num, -1, mid]])
    

    for l in bg_label:
        label[label == l] = 0
    
    #print("########### 1st : ",np.max(label))
    
    # select components based on volume
    properties = measure.regionprops(label)
    for prop in properties:
        if prop.area * spacing.prod() < vol_limit[0] * 1e6 or prop.area * spacing.prod() > vol_limit[1] * 1e6:
            label[label == prop.label] = 0
            
    #print(spacing.prod())        
    #print("########### 2nd : ",np.max(label))     
    #return label        
            
    # prepare a distance map for further analysis
    x_axis = np.linspace(-label.shape[1]/2+0.5, label.shape[1]/2-0.5, label.shape[1]) * spacing[1]
    y_axis = np.linspace(-label.shape[2]/2+0.5, label.shape[2]/2-0.5, label.shape[2]) * spacing[2]
    x, y = np.meshgrid(x_axis, y_axis)
    d = (x**2+y**2)**0.5
    vols = measure.regionprops(label)
    valid_label = set()
    # select components based on their area and distance to center axis on all slices
    for vol in vols:
        single_vol = label == vol.label
        slice_area = np.zeros(label.shape[0])
        min_distance = np.zeros(label.shape[0])
        for i in range(label.shape[0]):
            slice_area[i] = np.sum(single_vol[i]) * np.prod(spacing[1:3])
            min_distance[i] = np.min(single_vol[i] * d + (1 - single_vol[i]) * np.max(d))
        
        if np.average([min_distance[i] for i in range(label.shape[0]) if slice_area[i] > area_th]) < dist_th:
            valid_label.add(vol.label)
            
    bw = np.in1d(label, list(valid_label)).reshape(label.shape)
    
    # fill back the parts removed earlier
    if cut_num > 0:
        # bw1 is bw with removed slices, bw2 is a dilated version of bw, part of their intersection is returned as final mask
        bw1 = np.copy(bw)
        bw1[-cut_num:] = bw0[-cut_num:]
        bw2 = np.copy(bw)
        bw2 = scipy.ndimage.binary_dilation(bw2, iterations=cut_num)
        bw3 = bw1 & bw2
        label = measure.label(bw, connectivity=1)
        label3 = measure.label(bw3, connectivity=1)
        l_list = list(set(np.unique(label)) - {0})
        valid_l3 = set()
        for l in l_list:
            indices = np.nonzero(label==l)
            l3 = label3[indices[0][0], indices[1][0], indices[2][0]]
            if l3 > 0:
                valid_l3.add(l3)
        bw = np.in1d(label3, list(valid_l3)).reshape(label3.shape)
    
    return bw, len(valid_label)


def fill_hole(bw):
    # fill 3d holes
    label = measure.label(~bw)
    # idendify corner components
    bg_label = set([label[0, 0, 0], label[0, 0, -1], label[0, -1, 0], label[0, -1, -1], \
                    label[-1, 0, 0], label[-1, 0, -1], label[-1, -1, 0], label[-1, -1, -1]])
    bw = ~np.in1d(label, list(bg_label)).reshape(label.shape)
    
    return bw


def two_lung_only(bw, spacing, max_iter=22, max_ratio=4.8):    
    def extract_main(bw, cover=0.95):
        for i in range(bw.shape[0]):
            current_slice = bw[i]
            label = measure.label(current_slice)
            properties = measure.regionprops(label)
            properties.sort(key=lambda x: x.area, reverse=True)
            area = [prop.area for prop in properties]
            count = 0
            sum = 0
            while sum < np.sum(area)*cover:
                sum = sum+area[count]
                count = count+1
            filter = np.zeros(current_slice.shape, dtype=bool)
            for j in range(count):
                bb = properties[j].bbox
                filter[bb[0]:bb[2], bb[1]:bb[3]] = filter[bb[0]:bb[2], bb[1]:bb[3]] | properties[j].convex_image
            bw[i] = bw[i] & filter
           
        label = measure.label(bw)
        properties = measure.regionprops(label)
        properties.sort(key=lambda x: x.area, reverse=True)
        bw = label==properties[0].label

        return bw
    
    def fill_2d_hole(bw):
        for i in range(bw.shape[0]):
            current_slice = bw[i]
            label = measure.label(current_slice)
            properties = measure.regionprops(label)
            for prop in properties:
                bb = prop.bbox
                current_slice[bb[0]:bb[2], bb[1]:bb[3]] = current_slice[bb[0]:bb[2], bb[1]:bb[3]] | prop.filled_image
            bw[i] = current_slice

        return bw
    
    found_flag = False
    iter_count = 0
    bw0 = np.copy(bw)
    while not found_flag and iter_count < max_iter:
        label = measure.label(bw, connectivity=2)
        properties = measure.regionprops(label)
        properties.sort(key=lambda x: x.area, reverse=True)
        if len(properties) > 1 and properties[0].area/properties[1].area < max_ratio:
            found_flag = True
            bw1 = label == properties[0].label
            bw2 = label == properties[1].label
        else:
            bw = scipy.ndimage.binary_erosion(bw)
            iter_count = iter_count + 1
    
    if found_flag:
        d1 = scipy.ndimage.morphology.distance_transform_edt(bw1 == False, sampling=spacing)
        d2 = scipy.ndimage.morphology.distance_transform_edt(bw2 == False, sampling=spacing)
        bw1 = bw0 & (d1 < d2)
        bw2 = bw0 & (d1 > d2)
                
        bw1 = extract_main(bw1)
        bw2 = extract_main(bw2)
        
    else:
        bw1 = bw0
        bw2 = np.zeros(bw.shape).astype('bool')
        
    bw1 = fill_2d_hole(bw1)
    bw2 = fill_2d_hole(bw2)
    bw = bw1 | bw2

    return bw1, bw2, bw


def process_mask(mask):
    convex_mask = np.copy(mask)
    for i_layer in range(convex_mask.shape[0]):
        mask1  = np.ascontiguousarray(mask[i_layer])
        if np.sum(mask1)>0:
            mask2 = convex_hull_image(mask1)
            if np.sum(mask2)>2*np.sum(mask1):
                mask2 = mask1
        else:
            mask2 = mask1
        convex_mask[i_layer] = mask2
    struct = generate_binary_structure(3,1)  
    dilatedMask = binary_dilation(convex_mask,structure=struct,iterations=10) 
    return dilatedMask

def lumTrans(img):
    lungwin = np.array([-1200.,600.])
    newimg = (img-lungwin[0])/(lungwin[1]-lungwin[0])
    newimg[newimg<0]=0
    newimg[newimg>1]=1
    newimg = (newimg*255).astype('uint8')
    return newimg