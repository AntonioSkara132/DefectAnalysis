import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as mpimg
import utils

import numpy as np

files = utils.get_file_paths("/home/antonio123/workspace/Github_projects/DefectAnalysis/data/*.png")
files.sort()
for file in files:
    print(file)
    plt.figure()
    img = np.array(Image.open(file).convert('L'))
    plt.imshow(img)
    plt.colorbar()
    plt.show()