import json

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import utils
from PIL import Image

def computeCentroid(boxes: 'np.ndarray', ids: 'np.ndarray', files: 'list'):
    """computes coordinates of centroids and returns them as list"""
    centroids = []
    images = [np.array(Image.open(file).convert('L')) for file in files]
    for i in range(len(boxes)):
        box = boxes[i]
        index = ids[i]
        image = images[ids[i]]
        sum_x = 0
        sum_y = 0
        x0 = int(box[0])
        y0 = int(box[1])
        columns = int(box[2])
        rows = int(box[3])
        defect = image[y0:(y0+rows), x0:(x0+columns)]
        for i in range(columns):
            for j in range(rows):
                sum_x += x0 + i*defect[j][i]/255
                sum_y += y0 + j*defect[j][i]/255
        centroids.append([sum_x/(columns*rows), sum_y/(columns*rows)])
    return centroids

class BbAnalyzer:
    """"""
    images: 'list'
    data: 'dict'
    def __init__(self, json_path, defect_dir):
        f = open(json_path)
        self.data = json.load(f)
        self.images = utils.get_file_paths(defect_dir)

    def createSdHistogramFast(self, bins: 'int'):
        columns = self.data["images"][0]["width"]
        rows = self.data["images"][0]["height"]
        data = self.data["annotations"]
        boxes = np.asarray(list((sub["bbox"] for sub in data)))
        centroids = list(map(lambda x: [x[0] + x[2] / 2, x[1] + x[3] / 2], boxes))
        x = [centroid[0] for centroid in centroids]
        y = [centroid[1] for centroid in centroids]
        plt.hist2d(x, y, range=[[0, columns], [0, rows]], bins=bins)
        plt.title("Spatial defect centroid distribution")
        plt.colorbar()

    def createSdHistogram(self, bins):
        columns = self.data["images"][0]["width"]
        rows = self.data["images"][0]["height"]
        data = self.data["annotations"]
        boxes, ids = np.asarray(list((defect["bbox"] for defect in data))), np.asarray(list((defect["image_id"] for defect in data))),
        centroids = computeCentroid(boxes, ids, self.images)
        x = [centroid[0] for centroid in centroids]
        y = [centroid[1] for centroid in centroids]
        plt.hist2d(x, y, range=[[0, columns], [0, rows]], bins=bins)
        plt.title("Spatial defect centroid distribution")
        plt.colorbar()

    def createDefSizeHistogram(self):
        data = self.data['annotations']
        areas = np.asarray(list((sub["area"] for sub in data)))
        sns.histplot(areas, log_scale=True)
        plt.title("Size defect distribution")
        plt.xlabel("Area [pixel^2]")

    def createNumOfDefPerImgHistogram(self):
        data = self.data['annotations']
        ids = np.asarray(list((sub["image_id"] for sub in data)))
        sns.histplot(ids, discrete=True)
        plt.title("Number of defects per image")
        plt.xlabel("Image id")

    def createDefSizePerImhHistogram(self):
        data = self.data['annotations']
        ids = list((sub["image_id"] for sub in data))
        areas = np.asarray(list((sub["area"] for sub in data)))
        hist_data = pd.DataFrame({"Image_id":ids, "Area": areas})
        sns.histplot(hist_data, x='Image_id', y='Area', log_scale=(False, True), discrete=True, cbar=True)
        plt.title("Defect area per image")
        plt.xlabel("Image id")

    def createNumOfDefPfrImgHistogram(self):
        data = self.data['annotations']
        #SizeIdPairs = {"image_id": sub['image_id'], "area" }
        #sns.histplot(ids, discrete=True)
        plt.title("Number of defects per image")
        plt.xlabel("Image id")




