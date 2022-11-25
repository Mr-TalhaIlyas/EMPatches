
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
 [![Generic badge](https://img.shields.io/badge/Version-0.2.0-<COLOR>.svg)](https://shields.io/) [![Downloads](https://pepy.tech/badge/empatches)](https://pepy.tech/project/empatches)  [![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FMr-TalhaIlyas%2FEMPatches&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

# Extract and Merge Image Patches (EMPatches)

Extract and Merge Batches/Image patches (tf/torch), fast and self-contained digital image processing and deep learning model training.

* **Extract** patches
* **Merge** the extracted patches to obtain the original image back.

### *Update 0.2.1*
- From now you don't need to manage patch images, indices seperatly.
- Updated patches can be restored.
  - Manage patches easily by using `update()`, `reset()` method. (See `patches.ipynb`)

### *Update 0.2.0 (New Functionalities)*

* Handling of `tensorflow`/`pytorch` **Batched images** of shape `BxCxHxW` -> `pytorch` or `BxHxWxC` -> `tf`. C can be any number not limited to just RGB channels.
* **Modes** added for mergeing patches.
    1. `overwrite`: next patch will overwrite the overlapping area of the previous patch.
    2. `max` : maximum value of overlapping area at each pixel will be written.
    3. `min`: minimum value of overlapping area at each pixel will be written.
    4. `avg` : mean/average value of overlapping area at each pixel will be written.
* Patching via providing **Indices**.
* **Strided** patching thanks to [Andreasgejlm](https://github.com/Andreasgejlm)

## Dependencies

```
python >= 3.6
numpy 
math
```

# Usage

## Extracting Patches
```python
from empatches import EMPatches
import imgviz # just for plotting

# get image either RGB or Grayscale
img = cv2.imread('../digits.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```

![alt text](https://github.com/Mr-TalhaIlyas/EMPatches/raw/main/screens/digit.jpg)

```python
# load module
emp = EMPatches()
patches = emp.extract_patches(img, patchsize=512, overlap=0.2)

# displaying 1st 10 image patches
tiled= imgviz.tile(list(map(np.uint8, patches.imgs)),border=(255,0,0))
plt.figure()
plt.imshow(tiled)
```

![alt text](https://github.com/Mr-TalhaIlyas/EMPatches/raw/main/screens/patched.png)

## Image Processing
Now we can perform our operation on each patch independently and after we are done we can merge them back together.

```python
'''
pseudo code
'''
# do some processing, just store the patches in the list in same order
img_patches_processed = some_processing_func(patches.imgs)
# or run your deep learning model on patches independently and then merge the predictions
img_patches_processed = model.predict(patches.imgs)
'''For now lets just flip channels'''
changed_imgs = cv2.cvtColor(patches.imgs[1], cv2.COLOR_BGR2RGB)
patches.update(changed_imgs, [1])
```
![alt text](https://github.com/Mr-TalhaIlyas/EMPatches/raw/main/screens/patched_process.png)

## Merging Patches via various `modes`

After processing the patches if you can merge all of them back in original form as follows,

```python
merged_img = emp.merge_patches(patches, mode='max') # or
merged_img = emp.merge_patches(patches, mode='min') # or
merged_img = emp.merge_patches(patches, mode='overwrite') # or
merged_img = emp.merge_patches(patches, mode='avg') # or
# display
plt.figure()
plt.imshow(merged_img.astype(np.uint8))
plt.title(Your mode)
```
![alt text](https://github.com/Mr-TalhaIlyas/EMPatches/raw/main/screens/modesS.png)

## Strided Patching

```python
patches = emp.extract_patches(img, patchsize=512, overlap=0.2, stride=128)
tiled= imgviz.tile(list(map(np.uint8, patches.imgs)),border=(255,0,0))
plt.figure()
plt.imshow(tiled.astype(np.uint8))
plt.title('Strided patching')
```
![alt text](https://github.com/Mr-TalhaIlyas/EMPatches/raw/main/screens/stride.png)

## Batched Patching

### Things to know.

* batch : Batch of images of shape either BxCxHxW -> pytorch or BxHxWxC -> tf
                to extract patches from in list(list1, list2, ...),
                where, list1->([H W C], [H W C], ...) and so on.
* patchsize :  size of patch to extract from image only square patches can be
             extracted for now.
* overlap (Optional): overlap between patched in percentage a float between [0, 1].
* stride (Optional): Step size between patches
* type (Optional): Type of batched images tf or torch type

* batch_patches : a list containing lists of extracted patches of images.
* batch_indices : a list containing lists of indices of patches in order, whihc can be used 
            at later stage for 'merging_patches'.

* merged_batch : a np array of shape BxCxHxW -> pytorch or BxHxWxC -> tf.


### Extraction
```python
from empatches import BatchPatching

bp = BatchPatching(patchsize=512, overlap=0.2, stride=None, typ='torch')
# extracging
batch_patches, batch_indices = bp.patch_batch(batch) # batch of shape BxCxHxW, C can be any number 3 or greater

plt.imshow(batch_patches[1][2])
plt.title('3rd patch of 2nd image in batch')
```
![alt text](https://github.com/Mr-TalhaIlyas/EMPatches/raw/main/screens/bp.png)

### Merging
```python
# merging
# output will be of shpae depending on typ variable
# BxCxHxW -> torch or BxHxWxC -> tf
merged_batch = bp.merge_batch(batch_patches, batch_indices, mode='avg') 

# accessing the merged images
plt.imshow(merged_batch[1,...].astype(np.uint8))
plt.title('2nd merged image in batch')
```
![alt text](https://github.com/Mr-TalhaIlyas/EMPatches/raw/main/screens/bm.png)


## Patching via Providing Indices

**NOTE** in this case merging is not supported.

```python
from empatches import patch_via_indices

img = cv2.imread('./digit.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (1024, 512))

i = [(0, 512, 0, 256),  # 1st patch dims/indices
     (0, 256, 310, 922),# 2nd patch dims/indices
     (0, 512, 512, 768)]# 3rd patch dims/indices
img_patches = patch_via_indices(img, indices)

# plotting
tiled= imgviz.tile(list(map(np.uint8, img_patches)),border=(255,0,0))
plt.figure()
plt.imshow(tiled.astype(np.uint8))
plt.title('patching via providing indices')
```

![alt text](https://github.com/Mr-TalhaIlyas/EMPatches/raw/main/screens/p_via_indices.png)
