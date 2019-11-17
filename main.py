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

selem = mp.selem.disk(1)


def to_uint8(arr):
    temp = arr.astype(int)
    temp[temp == 1] = 255
    temp = np.uint8(temp)
    return temp


class Node:
    def __init__(self, name, parent=None, contour=None):
        self.name = name
        self.parent = parent
        self.child = []
        self.contour = contour
        if parent is not None:
            parent.child.append(self)
        self.level = 0

    def __str__(self):
        return str(self.name)

    def get_level(self):
        level = 0
        current = self
        while current.parent is not None:
            current = current.parent
            level += 1
        self.level = level


class Tree:
    def __init__(self, node):
        self.root = node
        self.root.level = 0

        current_node = self.root
        stack = []
        while True:
            current_node.get_level()
            if not current_node.child:
                if not stack:
                    break
                current_node = stack.pop()
                continue

            for i in current_node.child[1:]:
                stack.append(i)
            current_node = current_node.child[0]

    def __repr__(self):
        representation = ""
        current_node = self.root
        stack = []
        dices = []
        total = 0
        while True:
            if current_node.level == 1:
                dices.append(0)
            if current_node.level == 3:
                dices[-1] += 1
            if not current_node.child:
                if not stack:
                    for i, dice in enumerate(dices):
                        if not dice:
                            continue
                        representation += "dice no. " + str(i+1) + " : " + str(dice) + "\n"
                        total += dice
                    representation += "total: " + str(total)
                    return representation
                current_node = stack.pop()
                continue

            for i in current_node.child[1:]:
                stack.append(i)
            current_node = current_node.child[0]

    def die(self):
        current_node = self.root
        stack = []
        dices = {}
        total = 0
        current_contour = None
        while True:
            if current_node.level == 1:
                dices[current_node] = 0
                current_contour = current_node
            if current_node.level == 3:
                dices[current_contour] += 1
            if not current_node.child:
                if not stack:
                    return dices
                current_node = stack.pop()
                continue

            for i in current_node.child[1:]:
                stack.append(i)
            current_node = current_node.child[0]


def print_hierarchy(hierarchy, contours, ratio, image):
    nodes = {}
    seen = [-1]
    nodes[-1] = Node(-1)
    hierarchy = [[i, contour, parent] for i, (contour, [_, _, _, parent]) in enumerate(zip(contours, hierarchy))]
    while True:
        current = None
        for i in hierarchy:
            if i[0] in seen:
                continue
            current = i
            break
        if current is None:
            break
        i, contour, parent = current
        if parent in nodes:
            nodes[i] = Node(i, nodes[parent], contour)
            seen.append(i)
    tree = Tree(nodes[-1])
    dice = tree.die()
    print(dice)

    for c in dice:
        M = cv2.moments(c.contour)
        cX = int((M["m10"] / M["m00"]) * ratio)
        cY = int((M["m01"] / M["m00"]) * ratio)
        cY -= int(ratio * (np.sqrt(M["m00"]) * np.sqrt(2)) / 2)
        if dice[c] > 0:
            cv2.putText(image, str(dice[c]), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 2)


def main():
    image = cv2.imread('photos/dice4.jpg')
    resized = imutils.resize(image, width=300)
    ratio = image.shape[0] / float(resized.shape[0])

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # test = filters.prewitt(gray)
    denoised = restoration.denoise_nl_means(gray, h=0.95)
    threshold = filters.threshold_minimum(denoised)
    thres = denoised > threshold


    # ihc_hed = rgb2hed(image)

    # h = exposure.rescale_intensity(ihc_hed[:, :, 0], out_range=(0, 1))
    # d = exposure.rescale_intensity(ihc_hed[:, :, 2], out_range=(0, 1))
    # zdh = np.dstack((np.zeros_like(h), d, h))
    edges = feature.canny(thres, sigma=3)
    edges = mp.binary_dilation(edges, selem=selem)
    edges = to_uint8(edges)

    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
    contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Loop over the contours
    for c, h in zip(contours, hierarchy[0]):
        # Compute moments of the contours
        M = cv2.moments(c)
        cX = 0
        cY = 0
        c = c.astype("float")
        c *= ratio
        c = c.astype("int")
        cv2.drawContours(image, [c], -1, (0, 255, 0), 5)
        if h[-1] == -1:

            # Compute the centroids of the dies' outer contours
            cX = int((M["m10"] / M["m00"]) * ratio)
            cY = int((M["m01"] / M["m00"]) * ratio)
            # Move the Y coordinate to the bottom of a die
            cY += int(ratio*(np.sqrt(M["m00"])*np.sqrt(2))/2)
            cv2.putText(image, "Kostka", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 2)

    # cv2.drawContours(resized, cnt, -1, (0, 255, 0), 1)
    # image = imutils.resize(resized, width=800)
    # cnt = 0
    # for i in hierarchy[0]:
    #     if i[0] == -1 and i[1] == -1 and i[2] == -1:
    #         cnt += 1
    #
    # print(cnt)
    print_hierarchy(hierarchy[0], contours, ratio, image)

    img = Image.fromarray(image)
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
