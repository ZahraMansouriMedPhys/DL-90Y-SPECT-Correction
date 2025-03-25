
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom
from termcolor import cprint
import numpy as np
import monai
import cv2
import nibabel as nib

def common_segment_area(segment_1_url, segment_2_url, match_space = True):

    if isinstance(segment_1_url, str):
        segment_1_image = sitk.ReadImage(segment_1_url, sitk.sitkUInt8)
    else:
        segment_1_image = sitk.Cast(segment_1_url, sitk.sitkUInt8)
   
    if isinstance(segment_2_url, str):
        segment_2_image = sitk.ReadImage(segment_2_url, sitk.sitkUInt8)
    else:
        segment_2_image = sitk.Cast(segment_2_url, sitk.sitkUInt8)
        
    if match_space:
        segment_2_image = match_space(input_image = segment_2_image, reference_image = segment_1_image)
        
    common_segment = segment_1_image + segment_2_image
    common_segment[(segment_1_image == 0) | (segment_2_image == 0)] = 0
    common_segment[common_segment != 0] = 1
    common_area_volume = segment_volume(common_segment)[1]

    segment_1_volume = segment_volume(segment_1_image)[1]  
    segment_2_volume = segment_volume(segment_2_image)[1]  
    
    return common_area_volume/segment_1_volume, common_area_volume/segment_2_volume, common_area_volume



def padd_image_to_shape(image_url, target_shape=(96, 96, 96), replacing_value=0, orientation="LPS", pad_z_axis = True):

    if isinstance(image_url, str):
        image = sitk.ReadImage(image_url)
    else:
        image = image_url
    image = sitk.DICOMOrient(image, orientation)

    if any(x > y for x, y in zip(image.GetSize(), target_shape)):
        cprint("image is bigger than selected crop size returned None", "white", "on_red")
        return None

    pad_amount = [(t - s) for t, s in zip(target_shape, image.GetSize())]
    pad_lower = [(pa // 2) for pa in pad_amount]
    pad_upper = [pad_amount[i] - pad_lower[i] for i in range(len(pad_amount))]
    if not pad_z_axis:
        pad_lower[-1] = 0
        pad_upper[-1] = 0
    padded_image = sitk.ConstantPad(image, pad_lower, pad_upper, constant=replacing_value)
    return padded_image

def pad_array_to_shape(input_array, target_shape, replacing_value = 0):
    return np.pad(
        input_array,
        [(replacing_value, target_shape[i] - input_array.shape[i]) for i in range(len(input_array.shape))],
        "constant",
    )


def sitk_random(reference_image, minimum = -100, maximum = 100):
    if isinstance(reference_image, str):
        reference_image = sitk.ReadImage(reference_image)
        
    refernce_array = sitk.GetArrayFromImage(reference_image)
    random_array = np.random.uniform(minimum, maximum, refernce_array.shape)
    random_image = sitk.GetImageFromArray(random_array)
    random_image.CopyInformation(reference_image)
    return random_image

def sitk_rescale(image, input_min = "image-min", input_max = "image-max", output_min = 0, output_max = 1):
    if isinstance(image, str):
        image = sitk.ReadImage(image)
    image_array = sitk.GetArrayFromImage(image)
    if input_min == "image-min":
        input_min = np.min(image_array)
    if input_max == "image-max":
        input_max = np.max(image_array)
        
    array_no_clip = (image_array - input_min) / (input_max - input_min)
    array_no_clip = output_min + (array_no_clip * (output_max - output_min))
    image_no_clip = sitk.GetImageFromArray(array_no_clip)
    image_no_clip = CopyInfo(ReferenceImage = image, UpdatingImage = image_no_clip)
    
    array_clip = np.interp(image_array, (input_min, input_max), (output_min, output_max))
    image_clip = sitk.GetImageFromArray(array_clip)
    image_clip = CopyInfo(ReferenceImage = image, UpdatingImage = image_clip)
    return image_clip, image_no_clip

        
def sitk_resample(input_image, target_spacing, interpolation_method = "BSpline"):
    if isinstance(input_image, str):
        input_image = sitk.ReadImage(input_image, sitk.sitkFloat64)
    original_size = input_image.GetSize()
    original_spacing = input_image.GetSpacing()
    # Calculate the new size based on the desired spacing
    new_size = [int(round(original_size[i] * original_spacing[i] / target_spacing[i])) for i in range(input_image.GetDimension())]
    # Create the resampling transform
    resample_transform = sitk.Transform()
    # Perform the resampling
    if interpolation_method == "BSpline":
        resampled_image = sitk.Resample(input_image, new_size, resample_transform, 
                                        sitk.sitkBSpline, input_image.GetOrigin(), target_spacing)
    else:
        resampled_image = sitk.Resample(input_image, new_size, resample_transform, 
                                        interpolation_method, input_image.GetOrigin(), target_spacing, input_image.GetDirection())
    return resampled_image



def sitk_resize(input_image, target_size, interpolation_method="BSpline"):
    if isinstance(input_image, str):
        input_image = sitk.ReadImage(input_image)

    # Calculate the spacing for the resized image
    input_size = input_image.GetSize()
    input_spacing = input_image.GetSpacing()
    target_spacing = [sz * spc / trg_sz for sz, spc, trg_sz in zip(input_size, input_spacing, target_size)]

    # Set up the resampling filter
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(target_size)
    resampler.SetOutputOrigin(input_image.GetOrigin())
    resampler.SetOutputSpacing(target_spacing)  # Adjust spacing for resizing
    resampler.SetOutputDirection(input_image.GetDirection())

    # Set the interpolation method
    if interpolation_method == "BSpline":
        resampler.SetInterpolator(sitk.sitkBSpline)
    else:
        resampler.SetInterpolator(interpolation_method)

    # Execute the resampling
    resized_image = resampler.Execute(input_image)
    return resized_image

def sitk_rotate(image, degrees = (0,0,0)):
    if isinstance(image, str):
        image = sitk.ReadImage(image)
    # Assuming rotation_transform and LV_segment are defined before this code block
    # Set rotation parameters
    rotation_transform = sitk.Euler3DTransform()
    rotation_transform.SetRotation(np.deg2rad(degrees[0]), 
                                   np.deg2rad(degrees[1]),
                                   np.deg2rad(degrees[2]),
                                    )
    # Get the image center in physical coordinates
    image_center = np.array(image.TransformContinuousIndexToPhysicalPoint([(sz-1)/2.0 for sz in image.GetSize()]))
    # Set the rotation center
    rotation_transform.SetCenter(image_center)
    # Get the size of the original image
    original_size = image.GetSize()
    # Compute the bounding box of the rotated image
    rotated_corners = [rotation_transform.TransformPoint(image.TransformIndexToPhysicalPoint(idx)) for idx in
                       [(0, 0, 0), (original_size[0] - 1, 0, 0), (0, original_size[1] - 1, 0), (0, 0, original_size[2] - 1),
                        (original_size[0] - 1, original_size[1] - 1, original_size[2] - 1)]]
    min_coords = np.min(rotated_corners, axis=0)
    max_coords = np.max(rotated_corners, axis=0)
    # Compute the size of the bounding box
    bounding_box_size = np.ceil(max_coords - min_coords + 1).astype(int)
    # Create a new image with the size of the bounding box
    output_size = tuple(bounding_box_size.tolist())
    rotated_image = sitk.Image(output_size, image.GetPixelID())
    # Set the new image origin to the minimum coordinates of the bounding box
    rotated_image.SetOrigin(min_coords)
    # Perform the rotation without cropping
    rotated_image = sitk.Resample(image1=image,
                                    # size=output_size,
                                    transform=rotation_transform,
                                    referenceImage=rotated_image,
                                    useNearestNeighborExtrapolator=True,
                                    interpolator=sitk.sitkNearestNeighbor, 
                                    )
    return rotated_image, rotation_transform
    
def sitk_squeeze(image):
    if isinstance(image, str):
        image = sitk.ReadImage(image)
    image_size = image.GetSize()
    
    selct_range = [0] * len(image_size)
    for dim in range(len(image_size)):
        if image_size[dim] == 1:
            selct_range[dim] = 0
        else:
            selct_range[dim] = slice(0, image_size[dim])
    
    squeezed_image = image[selct_range]
    return squeezed_image



def sitk_percentile(image_url, percentile, segment_url = "none"):
    if isinstance(image_url, str):
        image_array = sitk.GetArrayFromImage(sitk.ReadImage(image_url))
    elif isinstance(image_url, sitk.Image):
        image_array = sitk.GetArrayFromImage(image_url)
    elif isinstance(image_url, np.ndarray):
        image_array = image_url
    if segment_url != "none":
        if isinstance(segment_url, str):
            segment_array = sitk.GetArrayFromImage(sitk.ReadImage(segment_url))
        elif isinstance(segment_url, sitk.Image):
            segment_array = sitk.GetArrayFromImage(segment_url)
        elif isinstance(segment_url, np.ndarray):
            segment_array = segment_url
            
        image_array = image_array[segment_array != 0]
    percentile_value_non_zeros = np.percentile(image_array[image_array != 0], percentile)
    percentile_value_with_zeros = np.percentile(image_array, percentile)
    return percentile_value_non_zeros, percentile_value_with_zeros


def sitk_gauusian(image_url, sigma):

    if isinstance(image_url, str):
        image = sitk.ReadImage(image_url)
        array = sitk.GetArrayFromImage(image)
    elif isinstance(image_url, sitk.Image):
        image = image_url
        array = sitk.GetArrayFromImage(image)
        
    smoothed_array = gaussian_filter(array, sigma)
    smoothed_image = sitk.GetImageFromArray(smoothed_array)
    smoothed_image = CopyInfo(ReferenceImage = image, UpdatingImage = smoothed_image)
    return smoothed_array, smoothed_image


def numpy_resize(input_image, target_size, interpolation_method = "BSpline"):

    if isinstance(input_image, str):
        input_image = sitk.ReadImage(input_image)
        input_array = sitk.GetArrayFromImage(input_image)
    elif isinstance(input_image, sitk.Image):
        input_array = sitk.GetArrayFromImage(input_image)
    elif isinstance(input_image, np.ndarray):
        input_array = input_image
    target_array_size = (target_size[2], target_size[1], target_size[0])
    resized_array = zoom(input_array, 
                         [x/y for x,y in zip(target_array_size,input_array.shape )],
                         )
    
    resized_image = sitk.GetImageFromArray(resized_array)
    resized_image_corrected = CopyInfo(input_image, resized_image)
    return resized_image_corrected
def pad_image(image, target_size):
    
    height, width = image.shape[:2]
    # Calculate required padding
    pad_height = max(target_size[0] - height, 0)
    pad_width = max(target_size[1] - width, 0)
    
    # Calculate padding on each side
    top_pad = pad_height // 2
    bottom_pad = pad_height - top_pad
    left_pad = pad_width // 2
    right_pad = pad_width - left_pad
 
    padded_image = cv2.copyMakeBorder(image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    return padded_image


def crop_image_to_segment(image, segment, crop_dims = "all", margin_mm = 0, 
                          lowerThreshold = 0.1, upperThreshold = .9,
                          insideValue = 0, outsideValue =1, 
                          force_match = False
                          ):
    if isinstance(image, str):
        image = sitk.ReadImage(image)
        image = sitk.DICOMOrient(image, "LPS")
    if isinstance(segment, str):
        segment = sitk.ReadImage(segment)
        segment = sitk.DICOMOrient(segment, "LPS")
        # finding crop area
    if force_match:
        segment = match_space(input_image = segment, reference_image = image)
    segment = sitk.Cast(segment, sitk.sitkUInt8)
    segment_non_binary = segment
    segment = sitk.BinaryThreshold(segment, lowerThreshold=lowerThreshold, 
                                   upperThreshold=upperThreshold, 
                                   insideValue = insideValue, outsideValue = outsideValue)
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(segment)
    bounding_box = label_shape_filter.GetBoundingBox(1) 

    start_physical_point = segment.TransformIndexToPhysicalPoint(bounding_box[0 : int(len(bounding_box) / 2)])
    end_physical_point = segment.TransformIndexToPhysicalPoint([x+sz for x,sz in zip(bounding_box[0 : int(len(bounding_box) / 2)], bounding_box[int(len(bounding_box) / 2) :])])
    if any([start>end for start, end in zip(start_physical_point, end_physical_point)]):
        cprint("warning directions have issues, check the output !!!!!", "white", "on_red")
    
    start_physical_point = [x - margin_mm for x in start_physical_point]
    end_physical_point = [x + margin_mm for x in end_physical_point]
    # crop using the indexes
    image_crop_start_indices = image.TransformPhysicalPointToIndex(start_physical_point)
    image_crop_end_indices = image.TransformPhysicalPointToIndex(end_physical_point)
    
    segment_crop_start_indices = segment.TransformPhysicalPointToIndex(start_physical_point)
    segment_crop_end_indices = segment.TransformPhysicalPointToIndex(end_physical_point)
    
    
    image_crop_sizes = [a-b for a,b in zip(image_crop_end_indices , image_crop_start_indices)]
    segment_crop_sizes = [a-b for a,b in zip(segment_crop_end_indices , segment_crop_start_indices)]
            
    
    image_crop_start_indices = list(image_crop_start_indices)
    for dimension, image_crop_start_index in enumerate(image_crop_start_indices):
        if image_crop_start_index < 0 : 
            image_crop_start_indices[dimension] = 0

    image_crop_sizes = list(image_crop_sizes)
    for dimension, image_crop_size in enumerate(image_crop_sizes):
        if image_crop_size + image_crop_start_indices[dimension] > image.GetSize()[dimension]: 
            image_crop_sizes[dimension] = image.GetSize()[dimension] - image_crop_start_indices[dimension] -1
        
    segment_crop_start_indices = list(segment_crop_start_indices)
    for dimension, segment_crop_start_index in enumerate(segment_crop_start_indices):
        if segment_crop_start_index < 0 : 
            segment_crop_start_indices[dimension] = 0


    segment_crop_sizes = list(segment_crop_sizes)
    for dimension, segment_crop_size in enumerate(segment_crop_sizes):
        if segment_crop_size + segment_crop_start_indices[dimension] > segment.GetSize()[dimension]: 
            segment_crop_sizes[dimension] = segment.GetSize()[dimension] - segment_crop_start_indices[dimension] -1
            
    image_crop_start_indices = list(image_crop_start_indices)
    if crop_dims == "all":
        "do nothging -- crop in all dimension"
    else:
        no_crop_dims = [x for x in [0,1,2] if x not in crop_dims]
        for dimension in no_crop_dims:
            image_crop_start_indices[dimension] = 0
            image_crop_sizes[dimension] = image.GetSize()[dimension]
    
    image_cropped = sitk.RegionOfInterest(image, image_crop_sizes, image_crop_start_indices)
    segment_cropped = sitk.RegionOfInterest(segment, segment_crop_sizes, segment_crop_start_indices)
    segment_non_binary_cropped = sitk.RegionOfInterest(segment_non_binary, segment_crop_sizes, segment_crop_start_indices)
    crop_box_out = {}
    crop_box_out["start_physical_point"] = start_physical_point
    crop_box_out["end_physical_point"] = end_physical_point
    crop_box_out["crop_start_indices"] = image_crop_start_indices
    crop_box_out["crop_end_indices"] = image_crop_end_indices
    crop_box_out["crop_sizes"] = image_crop_sizes
    
    return image_cropped, segment_cropped, segment_non_binary_cropped, crop_box_out

def mask_image_to_seg(image, segment, segment_value = 0, replace_value = 0, force_space = True):
    # image
    if isinstance(image, str):
        image = sitk.ReadImage(image)
    if isinstance(image, sitk.Image):
        image_array = sitk.GetArrayFromImage(image)
    elif isinstance(image, np.ndarray):
        image_array = image
    # segment
    if isinstance(segment, str):
        segment = sitk.ReadImage(segment)
        if force_space:
            segment = match_space(input_image = segment, reference_image = image)
    if isinstance(segment, sitk.Image):
        if force_space:
            segment = match_space(input_image = segment, reference_image = image)
        segment_array = sitk.GetArrayFromImage(segment)
    elif isinstance(segment, np.ndarray):
        segment_array = segment
        
    image_array[segment_array == segment_value] = replace_value
    
    image_masked = sitk.GetImageFromArray(image_array)
    image_masked.CopyInformation(image)
    return image_masked


def CropBckg(image, treshold = "otsu"):
    '''
    threshold_based_crop_and_bg_median was original name
    link : https://simpleitk.org/SPIE2019_COURSE/03_data_augmentation.html
    Use Otsu's threshold estimator to separate background and foreground. In medical imaging the background is
    usually air. Then crop the image using the foreground's axis aligned bounding box and compute the background 
    median intensity.
    Args:
        image (SimpleITK image): An image where the anatomy and background intensities form a bi-modal distribution
                                 (the assumption underlying Otsu's method.)
    Return:
        Cropped image based on foreground's axis aligned bounding box.
        Background median intensity value.
    '''
    if isinstance(image, str):
        image = sitk.ReadImage(image)
    # Set pixels that are in [min_intensity,otsu_threshold] to inside_value, values above otsu_threshold are
    # set to outside_value. The anatomy has higher intensity values than the background, so it is outside.
    inside_value = 0
    outside_value = 255
    if treshold == "otsu":
        bin_image = sitk.OtsuThreshold(image, inside_value, outside_value)
    else:
        bin_image = sitk.BinaryThreshold(image, lowerThreshold=treshold[0], upperThreshold=treshold[1], insideValue=inside_value, outsideValue=outside_value)

    # Get the median background intensity
    label_intensity_stats_filter = sitk.LabelIntensityStatisticsImageFilter()
    label_intensity_stats_filter.SetBackgroundValue(outside_value)
    label_intensity_stats_filter.Execute(bin_image,image)
    bg_mean = label_intensity_stats_filter.GetMedian(inside_value)
    
    # Get the bounding box of the anatomy
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()    
    label_shape_filter.Execute(bin_image)
    bounding_box = label_shape_filter.GetBoundingBox(outside_value)
    # The bounding box's first "dim" entries are the starting index and last "dim" entries the size
    return bg_mean, bin_image>1, sitk.RegionOfInterest(image, bounding_box[int(len(bounding_box)/2):], bounding_box[0:int(len(bounding_box)/2)])
    



def match_space(input_image, reference_image, interpolate = "linear", DefaultPixelValue = 0):
    if isinstance(input_image, str):
        input_image = sitk.ReadImage(input_image)
    if isinstance(reference_image, str):
        reference_image = sitk.ReadImage(reference_image)
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(reference_image.GetSpacing())
  
    resampler.SetSize(reference_image.GetSize())
    resampler.SetOutputOrigin(reference_image.GetOrigin())
    resampler.SetOutputDirection(reference_image.GetDirection())
    # Set the default pixel value to -1000
    resampler.SetDefaultPixelValue(DefaultPixelValue)
    if interpolate == "linear":
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    elif interpolate == "nearest":
        resampler.SetInterpolator(sitk.sitkLinear)
    elif interpolate.lower() == "bspline":
        resampler.SetInterpolator(sitk.sitkBSpline)
    resampled_image = resampler.Execute(input_image)
    return resampled_image
def is_same_space(image1, image2):
    if isinstance(image1, str):
        image1 = sitk.ReadImage(image1)
    if isinstance(image2, str):
        image2 = sitk.ReadImage(image2)
    decision = image1.GetDimension() == image2.GetDimension() and image1.GetSize() == image2.GetSize() and image1.GetOrigin() == image2.GetOrigin() and image1.GetSpacing() == image2.GetSpacing()
    return decision


def body_segment(url, lower_hu = -300, metal_hu = 2000, object_min_size = 10, close_voxels = 1, keep_largest = True):
    if isinstance(url, str):
        image = sitk.ReadImage(url)
    elif isinstance(url, sitk.Image):
        image = url
    image = sitk.Cast(image, sitk.sitkFloat32)
    bcct = sitk.BinaryThreshold(image, lowerThreshold=lower_hu, upperThreshold=metal_hu, insideValue=1, outsideValue=0)
    bcct = sitk.BinaryMorphologicalClosing(bcct, [close_voxels, close_voxels, close_voxels])  # Closing operation to fill small gaps
    for slice in range(bcct.GetSize()[2]):
        bcct[:,:,slice] = sitk.BinaryFillhole(bcct[:,:,slice], fullyConnected = True)  # Fill any remaining holes inside the body
    bcct = sitk.ConnectedComponent(bcct)
    
    bcct = select_objects_larger_than(bcct, object_min_size)
    if keep_largest:
        bcct = keep_largest_segments(bcct)
    bcct = bcct > 0
    bcct = sitk.Cast(bcct, sitk.sitkUInt8)
    bed_no_use = bcct
    return bcct, bed_no_use

def skin_body_contour(body_contour, kernel_radius = (3,3,3)):
    if isinstance(body_contour, str):
        body_contour = sitk.ReadImage(body_contour)
    elif isinstance(body_contour, np.ndarray):
        body_contour = sitk.GetImageFromArray(body_contour)
      
    shrunken_segment = sitk.BinaryErode(body_contour, kernelRadius = kernel_radius)
    skin_segment = sitk.Subtract(body_contour, shrunken_segment)
    return skin_segment
    

def get_orientation(image):
    if isinstance(image, str):
        image = nib.load(image)
    orientation = nib.aff2axcodes(image.affine)
    return orientation 


def CopyInfo(ReferenceImage, UpdatingImage, origin = True, spacing = True, direction = True):
    if isinstance(ReferenceImage, str):
        ReferenceImage = sitk.ReadImage(ReferenceImage)
    if isinstance(UpdatingImage, str):
        UpdatingImage = sitk.ReadImage(UpdatingImage)
    UpdatedImage = UpdatingImage 
    if origin:
        UpdatedImage.SetOrigin(ReferenceImage.GetOrigin())
    if spacing:
        UpdatedImage.SetSpacing(ReferenceImage.GetSpacing())
    if direction:
        UpdatedImage.SetDirection(ReferenceImage.GetDirection())
    return UpdatedImage

def is_segment(image, limit = 10):

    if isinstance(image, str):
        image = sitk.ReadImage(image)
    unique_values = np.unique(sitk.GetArrayFromImage(image))
    is_segment = len(unique_values)<limit and all(int(x) == x for x in unique_values)
    return unique_values, is_segment