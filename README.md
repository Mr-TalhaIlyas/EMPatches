
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
 [![Generic badge](https://img.shields.io/badge/Version-0.1.1-<COLOR>.svg)](https://shields.io/) [![Downloads](https://pepy.tech/badge/empatches)](https://pepy.tech/project/empatches) [![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FMr-TalhaIlyas%2FEMPatches&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

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
## Installation
[Pypi](https://pypi.org/project/empatches/)
```
pip install empatches
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


