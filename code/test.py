import json

import matplotlib.pyplot as plt
import numpy as np
import xlsxwriter

# Create a workbook and add a worksheet.
workbook = xlsxwriter.Workbook('images.xlsx')
worksheet = workbook.add_worksheet()

#worksheet.insert_image('B2', "/home/antonio123/workspace/Github_projects/DefectAnalysis/results/centroid_distribution_(25X20).png")

#workbook.close()
weights = list(np.arange(12))[::-1]

x = [1, 1, 2, 2, 1, 3, 4, 2, 1, 3, 4, 1]
y = [1, 5, 2, 2, 4, 2, 1, 2, 1, 1, 4, 2]
d = np.array([[1, 2], [2, 5]])
print(np.average(d))
