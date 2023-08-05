import json
import numpy as np

f = open("/home/antonio123/workspace/Github_projects/DefectAnalysis/data/_annotations.coco.json")
data = json.load(f)
data = data['annotations']
#areas = np.asarray(list((sub["area"] for sub in data)))
ids = np.asarray(list((sub["image_id"] for sub in data)))
print(ids)

a = [[1, 3], [2, 4]]
print(data)


