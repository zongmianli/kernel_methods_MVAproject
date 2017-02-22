import matplotlib
import numpy as np
from scipy.ndimage import uniform_filter
import os

def rgb2gray(rgb):
    """
    Convert RGB image to grayscale

    Parameters:
        rgb : RGB image
    
    Returns:
        gray : grayscale image
  
    """
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

def extract_hog(im):
    """
    Extract Histogram of Gradient (HOG) features for an image
    
    Inputs:
        im : an input grayscale or rgb image
    
    Returns:
        feat: Histogram of Gradient (HOG) feature
    """
    
    # convert rgb to grayscale
    image = rgb2gray(im)

    sx, sy = image.shape # image size
    orientations = 9 # number of gradient bins
    cx, cy = (8, 8) # pixels per cell

    gx = np.zeros(image.shape)
    gy = np.zeros(image.shape)
    gx[:, :-1] = np.diff(image, n=1, axis=1) # compute gradient on x-direction
    gy[:-1, :] = np.diff(image, n=1, axis=0) # compute gradient on y-direction
    grad_mag = np.sqrt(gx ** 2 + gy ** 2) # gradient magnitude
    grad_ori = np.arctan2(gy, (gx + 1e-15)) * (180 / np.pi) + 90 # gradient orientation

    n_cellsx = int(np.floor(sx / cx))  # number of cells in x
    n_cellsy = int(np.floor(sy / cy))  # number of cells in y
    
    # compute orientations integral images
    orientation_histogram = np.zeros((n_cellsx, n_cellsy, orientations))
    for i in range(orientations):
        # create new integral image for this orientation
        # isolate orientations in this range
        temp_ori = np.where(grad_ori < 180 / orientations * (i + 1),
                        grad_ori, 0)
        temp_ori = np.where(grad_ori >= 180 / orientations * i,
                        temp_ori, 0)
        # select magnitudes for those orientations
        cond2 = temp_ori > 0
        temp_mag = np.where(cond2, grad_mag, 0)
        orientation_histogram[:,:,i] = uniform_filter(temp_mag, size=(cx, cy))[cx/2::cx, cy/2::cy].T
    feat = orientation_histogram.ravel()
    dim = n_cellsx*n_cellsy*orientations
    return feat, dim

if __name__ == '__main__':
    input_path = 'data'
    input_file = os.path.join(input_path, 'Xtr.csv') # 'Xtr.csv' or 'Xte.csv'
    
    X = np.genfromtxt(input_file, delimiter=',')
    X = np.delete(X, np.s_[-1:], 1)
    n_img = X.shape[0]
    X = X.reshape(n_img, 3, 32, 32).transpose(0,2,3,1)
    
    # Use the first image to determine feature dimensions
    feat, dim = extract_hog(X[0])
    X_hog = np.zeros((n_img, dim))
    
    # Extract features for the rest of the images
    for i in range(n_img):
        feat, dim = extract_hog(X[i])
        X_hog[i, :] = feat
    
    # output setups
    output_path = 'data'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    output_file = os.path.join(output_path, 'Xtr_hog.txt') # 'Xtr_hog.txt' or 'Xte_hog.txt'
    np.savetxt(output_file, X_hog)
    
