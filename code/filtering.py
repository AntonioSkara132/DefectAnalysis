import numpy as np
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.mplot3d import Axes3D
import utils
import os

files = utils.get_file_paths("/home/antonio123/workspace/Github_projects/DefectAnalysis/data/*.png")
files.sort()

for i in range(len(files)):
    file = files[i]
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    #plt.figure()
    #plt.imshow(img)
    #plt.colorbar()
    #plt.show()
    image = np.where(img == 1, 255, 0)
    #plt.figure()
    #plt.imshow(image)
    #plt.show()
    file_name = os.path.basename(file)
    cv2.imwrite("/home/antonio123/workspace/Github_projects/DefectAnalysis/defects/" + file_name, image)

