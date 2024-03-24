from PIL import Image
import numpy as np
from skimage import io, filters, morphology
from skimage.measure import label, regionprops
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import closing, square
from skimage.segmentation import clear_border
from skimage.color import label2rgb
import matplotlib.pyplot as plt


def count_planes(image_path):
    image = io.imread(image_path)
    gray_image = rgb2gray(image)
    # show the image in grayscale
    plt.imshow(gray_image, cmap='gray')

    # apply threshold
    thresh = threshold_otsu(gray_image)
    binary = gray_image > thresh
    # show the binary image
    plt.imshow(binary, cmap='gray')

    # remove artifacts connected to image border
    cleared = clear_border(binary)
    # show the cleared image
    plt.imshow(cleared, cmap='gray')


image = 'images/1.jpg'
