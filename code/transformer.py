import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pycocotools import coco
import utils
import os

x_length = 1000
y_length = 500

def transform(x, y, image: 'np.ndarray'):

    h = z / 1000 * 1026
    xt = round(z*np.cos(theta))+1232
    yt = round(z*np.sin(theta))+1027
    return image[xt, yt]

def transform2(x, y, xc, yc, k, C):
    theta = x/1000 * 2*np.pi
    R = (y / 1000 * k) + C
    x2 = int(round(R*np.cos(theta) + xc))
    y2 = int(round(R*np.sin(theta) + yc))
    return x2, y2

def transform3(image, x, y):
    z2 = y
    theta = x/x_length * 2 * np.pi
    R = x_length/(2*np.pi)
    x2 = R*np.cos(theta)
    y2 = x_length/y_length * R*np.sin(theta)
    point = [x2, y2, z2, image[y][x]]
    return point

def transformImage(image, inner_bbox, outer_bbox):
    transformed = np.zeros((y_length, x_length), dtype=np.int64)
    xc = inner_bbox[0]+inner_bbox[2]/2
    yc = inner_bbox[1]+inner_bbox[3]/2
    C = min(inner_bbox[2], inner_bbox[3])/2
    k = max(outer_bbox[2], outer_bbox[3])/2-C
    for x in range(1000):
        for y in range(500):
            x2, y2 = transform2(x, y, xc, yc, k, C)
            transformed[y][x] = image[y2][x2]
    return transformed

def threeDPlot(file, inner_bbox, outer_bbox):
    image = np.array(Image.open(file).convert('L'))
    transformed = transformImage(image, inner_bbox, outer_bbox)
    pointcloud = []
    first = True
    for y in range(len(transformed)):
        for x in range(len(transformed[0])):
            if transformed[y][x] != 0:
                if first:
                    pointcloud = [transform3(transformed, x, y)]
                    first = False
                pointcloud.append(transform3(transformed, x, y))

    return pointcloud

def threeDPlot2(defects, inner_bbox, outer_bbox):
    for defect in defects:
        s= 0

"""image = np.array(Image.open("/home/antonio123/workspace/Github_projects/DefectAnalysis/data/img-10-_bmp.rf.d8de4750948704b5e615b9d21e3395d3.jpg").convert('L'))
data = coco.COCO("/home/antonio123/workspace/Github_projects/DefectAnalysis/code/_annotations.coco.json")
image_id = utils.findImageId(data.dataset, "img-10-_bmp.rf.d8de4750948704b5e615b9d21e3395d3.jpg")
ann_id = data.getAnnIds(imgIds=image_id, catIds=2)
inner_bbox = data.dataset["annotations"][ann_id[0]]["bbox"]
ann_id = data.getAnnIds(imgIds=image_id, catIds=3)
outer_bbox = data.dataset["annotations"][ann_id[0]]["bbox"]
plt.figure()
plt.imshow(transformImage(image, inner_bbox, outer_bbox))
plt.show()"""






















































