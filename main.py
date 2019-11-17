import scipy.ndimage as ndi
from skimage import io, transform, img_as_float, filters, restoration, morphology, feature
import skimage.morphology as mp
from skimage.color import rgb2hed, rgb2gray
from skimage.exposure import exposure
import numpy as np
import scipy.ndimage
from matplotlib import pyplot as plt
from PIL import Image
from skimage.measure import regionprops
import cv2
import imutils

selem = mp.selem.disk(2)


def to_uint8(arr):
    temp = arr.astype(int)
    temp[temp == 1] = 255
    temp = np.uint8(temp)
    return temp


def main():
    # image = io.imread('photos/dice4.jpg')
    image = cv2.imread('photos/dice4.jpg')
    rimage = imutils.resize(image, width=800)
    resized = imutils.resize(image, width=300)
    ratio = image.shape[0] / float(resized.shape[0])


    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)


    edges = feature.canny(gray, sigma=5)
    edges = mp.binary_dilation(edges, selem=selem)
    edges = to_uint8(edges)

    redges = imutils.resize(edges, width=800)


    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
    cnt, hr = cv2.findContours(redges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(rimage, cnt, -1, (0,255,0), 2)
    img = Image.fromarray(rimage)
    img.show()

    # height, width = int(500 / image.shape[1] * image.shape[0]), 500
    # image_resize = transform.resize(image, (height, width), anti_aliasing=True)

    # image = img_as_float(image_resize)

    # Immunohistochemical staining colors separation
    # ihc_hed = rgb2hed(image)

    # h = exposure.rescale_intensity(ihc_hed[:, :, 0], out_range=(0, 1))
    # d = exposure.rescale_intensity(ihc_hed[:, :, 2], out_range=(0, 1))
    # zdh = np.dstack((np.zeros_like(h), d, h))

    # image = rgb2gray(zdh)
    # image = Image.fromarray((image * 255).astype(np.uint8))
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # edges = rgb2gray(image)

    # Adding this leaves only the dice outline somehow
    # edges = restoration.denoise_nl_means(edges, h=0.95)

    # Option 1
    # edges = feature.canny(gray, sigma=5)
    # edges = mp.binary_dilation(edges, selem=selem)
    # edges = imutils.grab_contours(edges)

    # Option 2
    # edges = feature.canny(edges, sigma=5)
    # edges = filters.prewitt(edges)
    # edges = mp.binary.binary_closing(edges)

    # image = restoration.denoise_nl_means(image, h=0.95)
    # threshold = filters.threshold_minimum(image)
    # image = image > threshold

    # image = filters.prewitt(image)

    # labeled, objects = scipy.ndimage.label(edges)
    #
    # region = regionprops(labeled)
    #
    # for j in range(len(edges)):
    #     for k in range(len(edges[j])):
    #         if edges[j][k]:  # if appropriate label is present
    #             image[j][k] = [0, 0, 1]
    #
    # for cent in region:
    #     x, y = cent.centroid
    #     x, y = int(x), int(y)
    #     image[x][y] = [1, 0.55, 0]
    #     image[x+1][y] = image[x-1][y] = image[x][y+1] = image[x][y-1] = [1, 0.55, 0]
    #     image[x+2][y] = image[x-2][y] = image[x][y+2] = image[x][y-2] = [1, 0.55, 0]

    # image = ndi.binary_fill_holes(image)
    # image = morphology.remove_small_objects(image)
    # image = Image.fromarray((image * 255).astype(np.uint8))
    # peri = cv2.arcLength(edges, True)
    # approx = cv2.approxPolyDP(edges, 0.04 * peri, True)

    # image = Image.fromarray((image * 255).astype(np.uint8))
    # image.show()
    # print(objects)


if __name__ == "__main__":
    main()
