#Adapted from:
#https://github.com/mapbox/rio-hist/blob/master/docs/notebooks/Histogram_Matching_Numpy.ipynb
#Histogram matching algoritm with numpy

from PIL import Image
import numpy as np

def histogram_matching(source_img, reference_img):
    #Switch PIL images to numpy array
    source_img = np.array(source_img)
    
    reference_img = np.array(reference_img)
    
    orig_shape = source_img.shape
    
    source = source_img.ravel()
    
    #Horrible hack to remove all the black zones in the foreground images
    #Add only non-black (0,0,0) pixels
    filtered = []
    for x in range(0, source.size, 3):
        if source[x:x+2] != (0, 0, 0):
            filtered.append(source[x])
            filtered.append(source[x+1])
            filtered.append(source[x+2])
    
    reference = reference_img.ravel()
    
    s_values, s_idx, s_counts = np.unique(
        filtered, return_inverse=True, return_counts=True)
    
    s_counts[0] = 0
    
    #print(s_values, s_idx, s_counts)
    #print(s_values.shape, s_idx.shape, s_counts.shape)
    #print('')
    
    # the original source array
    np.array([s_values[idx] for idx in s_idx]).reshape(orig_shape)
    
    r_values, r_counts = np.unique(reference, return_counts=True)
    
    s_quantiles = np.cumsum(s_counts).astype(np.float64) / source.size
    
    r_quantiles = np.cumsum(r_counts).astype(np.float64) / reference.size
    
    interp_r_values = np.interp(s_quantiles, r_quantiles, r_values)
    
    target = interp_r_values[s_idx].reshape(orig_shape)

    # Return the adjusted image
    return Image.fromarray(target.astype('uint8'))