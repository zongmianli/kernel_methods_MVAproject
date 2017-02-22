import matplotlib
import numpy as np
from scipy.ndimage import uniform_filter
import os
import sys

if __name__ == '__main__':
    input_file = os.path.join('data', 'Xtr_hog.txt') # 'Xtr_hog.txt' or 'Xte_hog.txt'
    Xtr_hog = np.loadtxt(input_file)
    
    
    
