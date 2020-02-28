import csv
import os
from PIL import Image
import numpy as np

persons=[{'image_name':0,'MEL':0,'NV':0,'BCC':0,'AK':0,'BKL':0,'DF':0,'VASC':0,'SCC':0, 'unknown':1}]
csvfile=open('../../data/ISIC19_test_data.csv','a', newline='')
fields = list(persons[0].keys())
obj = csv.DictWriter(csvfile, fieldnames=fields)
obj.writeheader()
counter = 0
for filename in os.listdir('/home/haal01/Desktop/Projects/val2017'):

    if filename.endswith(".jpg"):
        # path = os.path.join('/home/haal01/Desktop/Projects/val2017', filename)
        #
        # img = Image.open(path, 'r')
        # img = np.asarray(img).copy()
        #
        # if len(img.shape) != 3:
        #     print(img.shape, path, counter)
        #
        #     os.remove(path)
        #
        # counter += 1
        if 'ISIC' in filename:
            obj.writerow(
                {'image_name': os.path.splitext(filename)[0], 'MEL': 1, 'NV': 0, 'BCC': 0, 'AK': 0, 'BKL': 0, 'DF': 0,
                 'VASC': 0, 'SCC': 0, 'unknown': 0})
        else:
            obj.writerow({'image_name': os.path.splitext(filename)[0],'MEL':0,'NV':0,'BCC':0,'AK':0,'BKL':0,'DF':0,'VASC':0,'SCC':0,'unknown':1})
        continue
    else:
        continue

csvfile.close()