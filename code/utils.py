import glob
from PIL import Image
import numpy as np
import xlsxwriter as xls

def get_file_paths(directory_path: 'str') -> list:
    """Return a list of paths matching a pathname pattern."""
    return glob.glob(directory_path)

def addFrequencies(freqs: 'np.ndarray', list: 'list'):
    """Counts index values"""
    for i in list:
        freqs[i] += 1
    return freqs

def computeCentroid(data: 'dict', files: 'list'):
    """computes coordinates of centroids and returns them as list"""
    boxes, ids = (np.asarray(list((defect["bbox"] for defect in data))),
                  np.asarray(list((defect["image_id"] for defect in data))))
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
        size = 0
        for i in range(columns):
            for j in range(rows):
                if defect[j][i] != 0:
                    size+=1
                    sum_x += x0 + i
                    sum_y += y0 + j
        centroids.append([sum_x/size, sum_y/size])
    return centroids

def getDefects(data: 'dict', files: 'list'):
    boxes, ids = (np.asarray(list((defect["bbox"] for defect in data))),
                  np.asarray(list((defect["image_id"] for defect in data))))
    defects = []
    images = [np.array(Image.open(file).convert('L')) for file in files]
    for i in range(len(boxes)):
        box = boxes[i]
        image = images[ids[i]]
        x0 = int(box[0])
        y0 = int(box[1])
        columns = int(box[2])
        rows = int(box[3])
        defect = image[y0:(y0 + rows), x0:(x0 + columns)]
        defects.append(defect)
    return defects

def saveAsExcel(name: 'str', filenames: 'list'):
    workbook = xls.Workbook(name)
    worksheet = workbook.add_worksheet()
    for i in range(len(filenames)):
        worksheet.insert_image(chr(66 + (i % 3) * 10) + str(2 + int((i / 3)) * 25), filenames[i]) #B1, L1, V1, B26, L26...
    workbook.close()