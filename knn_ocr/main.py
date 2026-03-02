import cv2
import numpy as np
from pathlib import Path
from skimage.measure import regionprops, label
from skimage.io import imread
import os

def extractor(image):
    if image.ndim == 2:
        binary = image
    else:
        gray = np.mean(image, 2).astype("u1")
        binary = gray > 0
    lb = label(binary)
    props = regionprops(lb)[0]
    return np.array([*props.moments_hu, props.eccentricity])

def make_train(path):
    train, responses = [], []
    ncls = 0
    for cls in sorted(path.glob("*")):
        ncls += 1
        for p in cls.glob("*.png"):
            # print(cls.name, ncls)
            train.append(extractor(imread(p)))
            responses.append(ncls)
    train = np.array(train, dtype="f4").reshape(-1, 8)
    responses = np.array(responses, dtype="f4").reshape(-1, 1)
    return train, responses

def is_space_between(props, idx1, idx2):
    if idx2 >= len(props) or idx1 >= len(props):
        return 0
    bbox1 = props[idx1].bbox
    bbox2 = props[idx2].bbox
    dist = bbox2[1] - bbox1[3]
    avg_width = np.max([p.bbox[3] - p.bbox[1] for p in props])
    return dist >= avg_width * 5

def remove_s(string):
    if string[0] == 's':
        string = string[1:]
    return string

path = Path("./task")

letters = sorted(list(map(remove_s, os.listdir("./task/train"))))
print(letters)
images = [imread(f"./task/{i}.png") for i in range(7)]

for image in images:
    train, responses = make_train(path / "train")
    knn = cv2.ml.KNearest.create()
    knn.train(train, cv2.ml.ROW_SAMPLE, responses)

    gray = image.mean(2)
    binary = gray > 0
    lb = label(binary)
    props = regionprops(lb)
    sorted_props = sorted(props, 
                   key=lambda x: x.bbox[1])
    find = []
    for prop in sorted_props:
        find.append(extractor(prop.image))
    find = np.array(find, dtype="f4").reshape(-1, 8)

    ret, results, neighbors, dist = knn.findNearest(find, 5)
    res = []
    for i, char in enumerate(results.flatten()):
        res.append(letters[int(char)-1])
        if is_space_between(props, i, i+1):
            res.append(' ')
    print("".join(res))