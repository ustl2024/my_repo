import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

image_folder = r"D:\pythonProject\Se\nails_segmentation\images"
mask_folder = r"D:\pythonProject\Se\nails_segmentation\labels"
output_folder = r"D:\pythonProject\Se\nails_segmentation\output"

image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg')])
mask_files = sorted([f for f in os.listdir(mask_folder) if f.endswith('.jpg')])

ious = []

def intersection_over_union(mask_flat, binary_image_flat):

    intersection = np.sum(np.logical_and(mask_flat, binary_image_flat))
    union = np.sum(np.logical_or(mask_flat, binary_image_flat))
    return intersection / union

for image_file, mask_file in zip(image_files, mask_files):

    image = cv.imread(os.path.join(image_folder, image_file))
    mask = cv.imread(os.path.join(mask_folder, mask_file), cv.IMREAD_GRAYSCALE)

    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    _, binary_image = cv.threshold(gray_image, 127, 255, cv.THRESH_BINARY)

    masked_image = cv.bitwise_and(image, image, mask=mask)

    cv.imwrite(os.path.join(output_folder, f'masked_{image_file}'), masked_image)

    mask_flat = mask.flatten() > 0
    binary_image_flat = binary_image.flatten() > 0

    iou = intersection_over_union(mask_flat, binary_image_flat)
    ious.append({'image': image_file, 'iou': iou})

ious_df = pd.DataFrame(ious)
print(ious_df)

plt.figure(figsize=(12, 8))
plt.bar(ious_df['image'], ious_df['iou'], color='blue')
plt.xlabel('Image File')
plt.ylabel('IoU Score')
plt.title('IoU')
plt.xticks(rotation=90, ha='right')
plt.ylim(0, 1)
plt.tight_layout()
plt.show()