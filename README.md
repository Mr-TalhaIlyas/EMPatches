
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
 [![Generic badge](https://img.shields.io/badge/Version-0.1.1-<COLOR>.svg)](https://shields.io/) [![Downloads](https://pepy.tech/badge/model-profiler)](https://pepy.tech/project/empatches)

# Extract and Merge Image Patches (EMPatches)

Extract and Merge image patches for easy, fast and self-contained digital image processing and deep learning model training.

* **Extract** patches
* **Merge** the extracted patches to obtain the original image back.


## Dependencies

```
python >= 3.6
numpy 
math
```

## Usage

### Extracting Patches
```python
from empatches import EMPatches

# get image either RGB or Grayscale
img = cv2.imread('../penguin.jpg')
# load module
emp = EMPatches()
img_patches, indices = emp.extract_patches(img, patchsize=32, overlap=0.2)

# displaying an image patch
plt.figure()
plt.imshow(img_patches[0])
```
### Image Processing
Now we can perform our operation on each patch independently and after we are done we can merge them back together.

```python
'''
pseudo code
'''
# do some processing, just store the patches in the list in same order
img_patches_processed = some_processing_func(img_patches)
# or run your deep learning model on patches independently and then merge the predictions
img_patches_processed = model.predict(img_patches)
```

### Merging Patches
After processing the patches if you can merge all of them back in original form as follows,
```python
merged_img = emp.merge_patches(img_patches_processed, indices)
# display
plt.figure()
plt.imshow(merged_img)
```


