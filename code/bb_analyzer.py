import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import utils

class BbAnalyzer:
    """"""
    images: 'list'
    data: 'dict'
    def __init__(self, json_path: 'str', defect_dir: 'str'):
        f = open(json_path)
        self.data = json.load(f)
        self.images = utils.get_file_paths(defect_dir)

    def createSdHistFast(self, bins: 'list'):
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

    def createSdHist(self, bins: 'list'):
        columns = self.data["images"][0]["width"]
        rows = self.data["images"][0]["height"]
        data = self.data["annotations"]
        centroids = utils.computeCentroid(data, self.images)
        x = [centroid[0] for centroid in centroids]
        y = [centroid[1] for centroid in centroids]
        plt.hist2d(x, y, range=[[0, columns], [0, rows]], bins=bins)
        plt.title("Spatial defect centroid distribution")
        plt.colorbar()

    def createDefAreaHist(self):
        data = self.data['annotations']
        areas = np.asarray(list((sub["area"] for sub in data)))
        sns.histplot(areas, log_scale=True)
        plt.title("Defect area distribution")
        plt.xlabel("Area")

    def createNumOfDefPerImgHist(self):
        data = self.data['annotations']
        ids = np.asarray(list((sub["image_id"] for sub in data)))
        sns.histplot(ids, discrete=True)
        plt.title("Number of defects per image")
        plt.xlabel("Image id")

    def createDefAreaPerImgHist(self):
        data = self.data['annotations']
        ids = list((sub["image_id"] for sub in data))
        areas = np.asarray(list((sub["area"] for sub in data)))
        hist_data = pd.DataFrame({"Image_id":ids, "Area": areas})
        sns.histplot(hist_data, x='Image_id', y='Area', log_scale=(False, True), discrete = (True, False), cbar=True, bins = (10, 10))
        plt.title("Defect area per image")
        plt.xlabel("Image id")

    def createNumOfDefPfrImgHist(self):
        data = self.data['annotations']
        #SizeIdPairs = {"image_id": sub['image_id'], "area" }
        #sns.histplot(ids, discrete=True)
        plt.title("Number of defects per image")
        plt.xlabel("Image id")

    def createDefPixelIntHist(self, bins):
        data = self.data["annotations"]
        defects = utils.getDefects(data, self.images)
        freqs = np.zeros([256])
        for defect in defects:
            freqs = utils.addFrequencies(freqs, defect.ravel())
        data = {'Frequency': freqs / np.sum(freqs), 'Intensity': np.arange(256)}
        sns.histplot(data, x='Intensity', weights='Frequency', bins=bins, discrete=True)
        plt.ylabel("Frequency(normalized)")
        plt.title("Overall defect pixel distribution")

    def createKDEDefPixelIntHist(self):
        data = self.data["annotations"]
        defects = utils.getDefects(data, self.images)
        freqs = np.zeros([256])
        for defect in defects:
            freqs = utils.addFrequencies(freqs, defect.ravel())
        data = {'Frequency': freqs / np.sum(freqs), 'Intensity': np.arange(256)}
        sns.kdeplot(data, x='Intensity', weights='Frequency', bw_adjust=0.1)
        plt.ylabel("Frequency(normalized)")
        plt.title("Overall defect pixel distribution density")

    def createDefPixNumHist(self):
        data = self.data["annotations"]
        defects = utils.getDefects(data, self.images)
        hist_data = [np.count_nonzero(defect) for defect in defects]
        sns.histplot(hist_data, log_scale=True)
        plt.ylabel("Count")
        plt.xlabel("Defect size in number of pixels")
        plt.title("Defect size distribution")

    def createDefPixSizeHist(self):
        data = self.data["annotations"]
        defects = utils.getDefects(data, self.images)
        pixels = []
        sizes = []
        for defect in defects:
            size = np.count_nonzero(defect)
            for row in defect:
                for pixel in row:
                    pixels.append(pixel)
                    sizes.append(size)
        #hist_data = pd.DataFrame(pixels_and_sizes, columns=['Pixel_value', 'Size'])
        #sns.histplot(hist_data, x='Pixel_value', y='Size', log_scale=(False, True), bins=(50, 50), cbar=True, fill=True)
        plt.hist2d(x=pixels, y=np.log10(sizes), bins = [40, 40])
        plt.colorbar()
        plt.ylabel("Size powers")

















