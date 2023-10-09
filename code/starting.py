import cv2
import matplotlib.pyplot as plt

import utils
import numpy as np

image = "/home/antonio123/workspace/Github_projects/DefectAnalysis/defects/patch_0.jpg"
mask = "/home/antonio123/workspace/Github_projects/DefectAnalysis/defects/patch_mask_0.png"

defects = utils.getDefects(unlabeled_images=image, labeled_images=mask)
centroids = utils.computeCentroid(defects)

image = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)

img = np.zeros(image.shape)
for centroid in centroids:
    print((centroid[0], centroid[1]))
    for i in range(40):
        for j in range(40):
            img[int(centroid[0]) + j - 20, int(centroid[1]) + i - 20] = 255

f, axarr = plt.subplots(3,1)
y = np.array([int(centroid[0]) for centroid in centroids])
x = np.array([int(centroid[1]) for centroid in centroids])
print(y)

axarr[0].imshow(img)
axarr[1].imshow(image)
axarr[2].hist2d(x, y, range=[[0, len(image[0])], [0, len(image)]], bins=[120, 96])
plt.show()

