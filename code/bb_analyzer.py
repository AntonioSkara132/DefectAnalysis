import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import utils
from PIL import Image
from pycocotools import coco
import transformer
import os
from tqdm import tqdm
from matplotlib import cm
import cv2

class BbAnalyzer:
    """"""
    u_images: 'str'
    l_images: 'str'
    image_width: 'int'
    image_height: 'int'
    data: 'dict'
    defects: 'list'

    def __init__(self, images_path, masks_path, anns_path):
        f = open(anns_path)
        self.u_images = images_path
        self.l_images = masks_path
        self.data = json.load(f)
        data = self.data["annotations"]
        self.defects = utils.getDefects(self.l_images, self.u_images)
        self.image_height = 2056
        self.image_width = 2464

    def createCdHistFast(self, bins: 'list'):
        columns = self.data["images"][0]["width"]
        rows = self.data["images"][0]["height"]
        data = self.data["annotations"]
        boxes = np.asarray(list((sub["bbox"] for sub in data)))
        centroids = list(map(lambda x: [x[0] + x[2] / 2, x[1] + x[3] / 2], boxes))
        x = np.array([centroid[0] for centroid in centroids])
        y = np.array([centroid[1] for centroid in centroids])
        plt.hist2d(x, -y, range=[[0, columns], [-rows, 0]], bins=bins)
        plt.title("Spatial defect centroid distribution")
        plt.colorbar()

    def createCdHist(self, bins: 'list'):
        centroids = utils.computeCentroid(self.defects)
        x = np.array([centroid[1] for centroid in centroids])
        y = np.array([centroid[0] for centroid in centroids])
        plt.hist2d(x, -y, range=[[0, self.image_width], [-self.image_height, 0]], bins=bins)
        plt.title("Spatial defect centroid distribution")
        plt.colorbar()

    def createAreaHist(self):
        data = self.data['annotations']
        areas = np.asarray(list((sub["area"] for sub in data)))
        sns.histplot(areas, log_scale=True)
        plt.title("Defect area distribution")
        plt.xlabel("Area")

    def createDefNumImgHist(self):
        data = self.data['annotations']
        ids = np.asarray(list((sub["image_id"] for sub in data)))
        freqs = np.zeros([len(self.data["images"])])
        print(len(freqs))
        print(len(self.data["images"]))
        for id in ids: freqs[id] += 1
        freqs = sorted(freqs)
        values = np.arange(0, len(self.data["images"]))
        data = pd.DataFrame({"Frequencies": freqs, "Values": values})
        sns.histplot(data, x = "Values", weights="Frequencies", bins=len(self.data["images"]))
        plt.title("Number of defects per image(sorted)")

    def createAreaImgHist(self):
        data = self.data['annotations']
        ids = list((sub["image_id"] for sub in data))
        areas = np.asarray(list((sub["area"] for sub in data)))
        hist_data = pd.DataFrame({"Image_id":ids, "Area": areas})
        sns.histplot(hist_data, x='Image_id', y='Area', log_scale=(False, True), discrete = (True, False), cbar=True)
        plt.title("Defect area per image")
        plt.xlabel("Image id")

    def createNumOfDefPfrImgHist(self):
        data = self.data['annotations']
        #SizeIdPairs = {"image_id": sub['image_id'], "area" }
        #sns.histplot(ids, discrete=True)
        plt.title("Number of defects per image")
        plt.xlabel("Image id")

    def createPixelHist(self, bins=256):
        "creates defect pixel distribution"
        data = self.data["annotations"]
        defects = utils.getDefects(unlabeled_images=self.u_images, labeled_images=self.l_images)
        freqs = np.zeros([256])
        for defect in defects:
            pixels = [el[2] for el in defect]
            freqs = utils.addFrequencies(freqs, pixels)
        data = {'Frequency': freqs / np.sum(freqs), 'Intensity': np.arange(256)}
        print(data)
        sns.histplot(data, x='Intensity', weights='Frequency', bins=bins, discrete=True)
        plt.ylabel("Frequency(normalized)")
        plt.title("Overall defect pixel distribution")

    def createKdePixelHist(self):
        freqs = np.zeros([256])
        for defect in self.defects:
            pixels = [el[2] for el in defect]
            freqs = utils.addFrequencies(freqs, pixels)
        data = {'Frequency': freqs / np.sum(freqs), 'Intensity': np.arange(256)}
        sns.kdeplot(data, x='Intensity', weights='Frequency', bw_adjust=0.1)
        plt.ylabel("Frequency(normalized)")
        plt.title("Overall defect pixel distribution density")

    def createDefPixNumHist(self):
        hist_data = [len(defect) for defect in self.defects]
        sns.histplot(hist_data, log_scale=True)
        plt.ylabel("Count")
        plt.xlabel("Defect size in number of pixels")
        plt.title("Defect size distribution")

    def createPixIntSizeHist(self):
        pixels = []
        sizes = []
        for defect in self.defects:
            size = len(defect)
            for row in defect:
                for pixel in row:
                    pixels.append(pixel)
                    sizes.append(size)

        #hist_data = pd.DataFrame(pixels_and_sizes, columns=['Pixel_value', 'Size'])
        #sns.histplot(hist_data, x='Pixel_value', y='Size', log_scale=(False, True), bins=(50, 50), cbar=True, fill=True)

        plt.hist2d(x=pixels, y=np.log10(sizes), bins = [40, 40])
        plt.colorbar()
        plt.ylabel("Size powers")
        plt.title('Defect pixel distribution with the respect to defect size')

    def createPixAvgSizeHist(self):
        avgs = []
        sizes = []
        for defect in self.defects:
            size = len(defect)
            pixels = [el[2] for el in defect]
            sizes.append(size)
            avgs.append(np.sum(pixels)/size)
        # hist_data = pd.DataFrame(pixels_and_sizes, columns=['Pixel_value', 'Size'])
        # sns.histplot(hist_data, x='Pixel_value', y='Size', log_scale=(False, True), bins=(50, 50), cbar=True, fill=True)

        plt.hist2d(x=avgs, y=np.log10(sizes), bins=[40, 40])
        plt.colorbar()
        plt.ylabel("Size powers")
        plt.title('Defect pixel distribution with the respect to defect size')

    def showMeasuers(self):
        measures = ["OV. average", "OV. median", "OV. standard\ndeviation"]
        data = []
        for i in range(len(self.defects)):
            def_data = [pix[2] for pix in self.defects[i]]
            if i == 0: data = def_data
            else: data = np.concatenate((data, def_data))
        values = [np.average(data), np.median(data), np.std(data)]
        plt.barh(measures, values)
        xtick_positions = np.linspace(0, max(values), 20)
        xtick_labels = [str(int(pos)) for pos in xtick_positions]
        plt.xticks(xtick_positions, xtick_labels)
        plt.yticks(rotation=90)

    def createPixDistSizeHist(self):
        """plots pixel intensity distribution for pixels in specified intervals"""
        sizes = [len(defect) for defect in self.defects]
        max_size = max(sizes)
        min_size = min(sizes)
        stop = int(np.ceil(np.log10(max_size)))
        start = int(np.floor(np.log10(min_size)))
        size_labels = np.logspace(start=start, stop=stop, base = 10, num=(stop-start)*4+1)
        size_keys = np.digitize(sizes, size_labels)
        map = np.zeros([len(size_labels)-1, 256])
        "punjenje mape"
        for i in range(len(self.defects)):
            defect = [el[2] for el in self.defects[i]]
            map[size_keys[i]-1] = utils.addFrequencies(map[size_keys[i]-1], list(defect))
        "normiranje mape"
        for i in range(len(map)):
            if (max(map[i])/255) != 0:
                map[i] /= (max(map[i]))
        sns.heatmap(map)
        size_labels = [utils.format_float_in_AeB(size) for size in size_labels]
        plt.yticks(np.arange(len(size_labels)), labels=size_labels, rotation=0)
        plt.ylabel("Size")
        plt.xlabel("Intensity")
        plt.title("Pixel distribution per size(in interval)")

    def createAccDefPixDist(self):
        heatmap = np.zeros([self.image_height, self.image_width])
        imgs = utils.get_file_paths(self.l_images)
        ground_truths = [np.array(Image.open(img).convert('L')) for img in imgs]
        ann_maps = [utils.getBlobMap(ground_truth) for ground_truth in ground_truths]
        for ann_map in ann_maps:
            heatmap = np.add(heatmap, ann_map)
        plt.imshow(heatmap)
        plt.title("Accumulated anomaly spatial pixel distribution")
        plt.colorbar()

    def createWeightCdhist(self, bins):
        """plots pixel intensity distribution for pixels in specified intervals"""
        sizes = [len(defect) for defect in self.defects]
        max_size = max(sizes)
        min_size = min(sizes)

        stop = np.log10(max_size)+1
        start = np.log10(min_size)-1
        size_labels = np.logspace(start=start, stop=stop, base=10, num=11)
        size_keys = np.digitize(sizes, size_labels)
        weights = [size_key + 1 for size_key in size_keys]

        centroids = utils.computeCentroid(self.defects)
        y = [centroid[0] for centroid in centroids]
        x = [centroid[1] for centroid in centroids]

        plt.hist2d(x, -y, range=[[0, self.image_width], [-self.image_height], 0], weights=weights, bins=bins)
        plt.colorbar()
        plt.title("Weighted centroid distribution")

    def create3dDefectPixelWiseDistribution(self):
        l_files = utils.get_file_paths("/home/antonio123/workspace/Github_projects/DefectAnalysis/defects/*.png")
        u_files = utils.get_file_paths(self.u_images)
        l_files.sort()
        u_files.sort()
        data = coco.COCO("/home/antonio123/workspace/Github_projects/DefectAnalysis/code/_annotations.coco.json")
        pointcloud = []

        for count in tqdm(range(len(l_files))):
            l_file = l_files[count]
            u_file = u_files[count]
            image_id = utils.findImageId(data.dataset, os.path.basename(u_file))
            ann_id = data.getAnnIds(imgIds=image_id, catIds=2)
            inner_bbox = data.dataset["annotations"][ann_id[0]]["bbox"]
            ann_id = data.getAnnIds(imgIds=image_id, catIds=3)
            outer_bbox = data.dataset["annotations"][ann_id[0]]["bbox"]

            if count == 0: pointcloud = transformer.threeDPlot(l_file, inner_bbox=inner_bbox, outer_bbox=outer_bbox)
            else: pointcloud += transformer.threeDPlot(l_file, inner_bbox, outer_bbox)

        x = [int(point[0]) for point in pointcloud]
        y = [int(point[1]) for point in pointcloud]
        z = [int(point[2]) for point in pointcloud]
        #print(x)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Plot the point cloud data
        ax.scatter(x, y, z, c='r')
        # Set the axis labels
        utils.plotCylder(ax)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        # Show the plot
        plt.show()

    def create3dDefectPixelWiseDistribution2(self):
        l_files = utils.get_file_paths("/home/antonio123/workspace/Github_projects/DefectAnalysis/defects/*.png")
        u_files = utils.get_file_paths(self.u_images)
        l_files.sort()
        u_files.sort()
        data = coco.COCO("/home/antonio123/workspace/Github_projects/DefectAnalysis/code/_annotations.coco.json")
        heatmap = np.zeros([500, 1000])

        for count in tqdm(range(len(l_files))):
            l_file = l_files[count]
            u_file = u_files[count]
            image_id = utils.findImageId(data.dataset, os.path.basename(u_file))
            ann_id = data.getAnnIds(imgIds=image_id, catIds=2)
            inner_bbox = data.dataset["annotations"][ann_id[0]]["bbox"]
            ann_id = data.getAnnIds(imgIds=image_id, catIds=3)
            outer_bbox = data.dataset["annotations"][ann_id[0]]["bbox"]
            image = np.array(Image.open(l_file).convert('L'))
            u_image = np.array(Image.open(u_file).convert('L'))
            u_transformed = transformer.transformImage(u_image, inner_bbox=inner_bbox, outer_bbox=outer_bbox)
            transformed = transformer.transformImage(image, inner_bbox=inner_bbox, outer_bbox=outer_bbox)
            """fig = plt.figure()
            ax1 = fig.add_subplot(2, 2, 1)
            ax1.imshow(image)
            ax2 = fig.add_subplot(2, 2, 2)
            ax2.imshow(u_transformed)
            ax2 = fig.add_subplot(2, 2, 3)
            ax2.imshow(transformed)
            plt.show()"""
            heatmap = np.add(heatmap, transformed)

        heatmap /= float(np.max(heatmap))
        plt.figure()
        plt.imshow(heatmap)
        plt.colorbar()
        plt.show()
        # Load a 2D grayscale image
        image = heatmap
        # Define cylinder parameters
        radius = 5
        height = 10

        # Create a 3D cylinder grid
        theta = np.linspace(0, 2 * np.pi, image.shape[1])
        z = np.linspace(0, height*2, image.shape[0])
        Theta, Z = np.meshgrid(theta, z)

        # Convert the 2D image coordinates to polar coordinates
        r_image = np.linspace(0, 1, image.shape[1])
        theta_image = np.linspace(0, 2 * np.pi, image.shape[0])
        R_image, Theta_image = np.meshgrid(r_image, theta_image)

        # Convert polar coordinates to Cartesian coordinates
        X_image = R_image * np.cos(Theta_image)
        Y_image = R_image * np.sin(Theta_image)

        # Map the 2D image onto the 3D cylinder surface
        X_cylinder = radius * np.cos(Theta)
        Y_cylinder = radius * np.sin(Theta)
        Z_cylinder = Z

        # Create a 3D figure and axis
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the image on the cylinder surface with a colormap
        cmap = plt.get_cmap('viridis')  # Grayscale colormap
        ax.plot_surface(X_cylinder, Y_cylinder, Z_cylinder, facecolors=cmap(image), rstride=10, cstride=10)

        # Set equal aspect ratio for all axes
        ax.set_box_aspect([1, 1, 1])

        # Set labels for axes
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Show the plot
        plt.show()
        plt.savefig('3d_plot', dpi = 300)








