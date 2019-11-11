import scipy.ndimage as ndi
from skimage import io, transform, img_as_float, filters, restoration, morphology
from skimage.color import rgb2hed, rgb2gray
from skimage.exposure import exposure
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


def main():
    image = io.imread('photos/dice4.jpg')
    height, width = int(500 / image.shape[1] * image.shape[0]), 500
    image_resize = transform.resize(image, (height, width), anti_aliasing=True)
    image = img_as_float(image_resize)

    # Immunohistochemical staining colors separation
    ihc_hed = rgb2hed(image)

    h = exposure.rescale_intensity(ihc_hed[:, :, 0], out_range=(0, 1))
    d = exposure.rescale_intensity(ihc_hed[:, :, 2], out_range=(0, 1))
    zdh = np.dstack((np.zeros_like(h), d, h))

    image = rgb2gray(zdh)
    # image = filters.prewitt(image)
    # image = restoration.denoise_nl_means(image, h=0.95)
    # threshold = filters.threshold_minimum(image)
    # image = image > threshold

    # image = filters.prewitt(image)

    # image = ndi.binary_fill_holes(image)
    # image = morphology.remove_small_objects(image)
    image = Image.fromarray((image * 255).astype(np.uint8))
    image.show()



if __name__ == "__main__":
    main()
