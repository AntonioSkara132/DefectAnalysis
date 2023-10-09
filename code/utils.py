import glob
from PIL import Image
import numpy as np
import xlsxwriter as xls
import cv2
import openpyxl
from tqdm import tqdm
import matplotlib.pyplot as plt

def get_file_paths(directory_path: 'str') -> list:
    """Return a list of paths matching a pathname pattern."""
    return glob.glob(directory_path)

def addFrequencies(freqs: 'np.ndarray', list: 'list'):
    """Counts index values"""
    for i in list:
        freqs[i] += 1
    return freqs

def computeCentroid(defects: 'list'):
    centroids = []
    for defect in defects:
        sum_x = 0
        sum_y = 0
        for el in defect:
            sum_y += el[0]
            sum_x += el[1]
        centroids.append([sum_y/len(defect), sum_x/(len(defect))])
    return centroids

def getDefects(labeled_images: 'str', unlabeled_images: 'str'):
    lab_files = get_file_paths(labeled_images)
    unlab_files = get_file_paths(unlabeled_images)
    lab_files.sort()
    unlab_files.sort()
    defects = []
    for i in tqdm(range(len(lab_files))):
        blobs = getBlobs(np.array(Image.open(lab_files[i]).convert('L')))
        img = np.array(Image.open(unlab_files[i]).convert('L'))
        for blob in blobs:
            defect = []
            for indice in blob:
                defect.append([indice[0], indice[1], img[indice[0]][indice[1]]])
            defects.append(defect)
    return defects

def getBlobMap(image: 'np.ndarray'):
    blob_intensity = 1
    thresholded = np.where(image == blob_intensity, 1, 0).astype(np.uint8)
    return thresholded

def getBlobs(image: 'np.ndarray'):
    # Threshold the image based on blob intensity (1)
    blob_intensity = 1
    thresholded = np.where(image == blob_intensity, 255, 0).astype(np.uint8)

    # Connected Component Analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresholded)

    # Extract blob indices
    blobs = []
    for label in range(1, num_labels):  # Skip label 0 (background)
        blob_indices = np.argwhere(labels == label)
        blobs.append(blob_indices)
    return blobs


def saveAsExcel(name: 'str', filenames: 'list'):
    workbook = xls.Workbook(name)
    worksheet = workbook.add_worksheet()
    for i in range(len(filenames)):
        worksheet.insert_image(chr(66 + (i % 3) * 10) + str(2 + int((i / 3)) * 25), filenames[i]) #B1, L1, V1, B26, L26...
    workbook.close()

def format_float_in_AeB(number):
    # Convert the number to the "AeB" format
    exponent = int(np.floor(np.log10(abs(number))))
    coefficient = number / (10 ** exponent)

    formatted_string = f"{coefficient:.2f}e{exponent}"
    return formatted_string

def savingInDatasetStatstistics(sheet_num: 'int', images: 'list'):

    wb = openpyxl.load_workbook("/home/antonio123/Documents/Praksa/dataset statistics.xlsx")
    ws = wb.worksheets[sheet_num]

    for i, image in enumerate(images, 0):
        img = openpyxl.drawing.image.Image(image)
        img.anchor = chr(66 + (i % 3) * 7) + str(2 + int((i / 3)) * 25) # B1, L1, V1, B26, L26...
        ws.add_image(img)

    wb.save("/home/antonio123/Documents/Praksa/dataset statistics.xlsx")

def applyFunction(space: 'np.ndarray', fun):
    for z in range(len(space)):
        for y in range(len(space[0])):
            for x in range(len(space[0][0])):
                if space[z][y][x]:
                    space[z][y][x] = fun(x, y, z)
    return space

def findImageId(coco_data: 'dict', desired_file_name):
    # Find the image_id with the specified file_name
    image_id = None
    for image in coco_data['images']:
        if image['file_name'] == desired_file_name:
            image_id = image['id']
            break
    return image_id

def plotCylder(ax):
    radius = 150
    height = 800
    resolution = 1000  # Number of points to create the cylinder's surface

    # Create points for the cylinder's surface
    theta = np.linspace(0, 2 * np.pi, resolution)
    z = np.linspace(0, height, resolution)
    theta, z = np.meshgrid(theta, z)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    # Plot the cylinder surface
    ax.plot_surface(x, y, z, color='lightblue',  # Set the color to light blue
    rstride=1,
    cstride=1,
    linewidth=0,
    alpha=0.7)

def showImage(image: np.ndarray):
    plt.figure()
    plt.imshow(image)
    plt.colorbar()
    plt.show()
    pass
