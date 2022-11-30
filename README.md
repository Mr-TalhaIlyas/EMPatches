
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
 [![Generic badge](https://img.shields.io/badge/Version-0.2.0-<COLOR>.svg)](https://shields.io/) [![Downloads](https://pepy.tech/badge/empatches)](https://pepy.tech/project/empatches)  [![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FMr-TalhaIlyas%2FEMPatches&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

# Extract and Merge Image Patches (EMPatches)

Extract and Merge Batches/Image patches (tf/torch), fast and self-contained digital image processing and deep learning model training.

* **Extract** patches
* **Merge** the extracted patches to obtain the original image back.
### *Upadate 0.2.2 (New Functionalities)*

* Handling 1D spectral and 3D volumetric data structures, thanks to [antonyvam](https://github.com/antonyvam).
* Batch processing support for 1D, 2D, 3D (image/pixel + voxel/volumetric) data added.
* Bug fixes for multi-dimensional image patch merging for `C > 3`.

### *Update 0.2.0*

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
* [Extracting Patches](#Extracting-Patches)
* [Merging Patches](#Merging-Patches)
* [Voxel/Volumetric Data patching](#Voxel-patching)
* [1D spectral Data patching](#1D-patching)
* [Strided Patching](#Strided-Patching)
* [Batched Patching](#Batched-Patching)
* [Patching via Providing Indices](#Patching-via-Providing-Indices)

## <a name="Extracting-Patches">Extracting Patches</a>
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
img_patches, indices = emp.extract_patches(img, patchsize=512, overlap=0.2)

# displaying 1st 10 image patches
tiled= imgviz.tile(list(map(np.uint8, img_patches)),border=(255,0,0))
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
img_patches_processed = some_processing_func(img_patches)
# or run your deep learning model on patches independently and then merge the predictions
img_patches_processed = model.predict(img_patches)
'''For now lets just flip channels'''
img_patches[1] = cv2.cvtColor(img_patches[1], cv2.COLOR_BGR2RGB)
```
![alt text](https://github.com/Mr-TalhaIlyas/EMPatches/raw/main/screens/patched_process.png)

## <a name="Merging-Patches">Merging-Patches</a>

After processing the patches if you can merge all of them back in original form as follows,

```python
merged_img = emp.merge_patches(img_patches, indices, mode='max') # or
merged_img = emp.merge_patches(img_patches, indices, mode='min') # or
merged_img = emp.merge_patches(img_patches, indices, mode='overwrite') # or
merged_img = emp.merge_patches(img_patches, indices, mode='avg') # or
# display
plt.figure()
plt.imshow(merged_img.astype(np.uint8))
plt.title(Your mode)
```
![alt text](https://github.com/Mr-TalhaIlyas/EMPatches/raw/main/screens/modesS.png)

## <a name="Strided-Patching">Strided Patching</a>

```python
img_patches, indices = emp.extract_patches(img, patchsize=512, overlap=0.2, stride=128)
tiled= imgviz.tile(list(map(np.uint8, img_patches)),border=(255,0,0))
plt.figure()
plt.imshow(tiled.astype(np.uint8))
plt.title('Strided patching')
```
![alt text](https://github.com/Mr-TalhaIlyas/EMPatches/raw/main/screens/stride.png)

## <a name="Voxel-patching">Volumetric/Voxel data patching</a>

```python
# first generate a sample data
def midpoints(x):
    sl = ()
    for i in range(x.ndim):
        x = (x[sl + np.index_exp[:-1]] + x[sl + np.index_exp[1:]]) / 2.0
        sl += np.index_exp[:]
    return x
r, g, b = np.indices((17, 17, 17)) / 16.0
rc = midpoints(r)
gc = midpoints(g)
bc = midpoints(b)
# define a sphere about [0.5, 0.5, 0.5]
sphere = ((rc - 0.5)**2 + (gc - 0.5)**2 + (bc - 0.5)**2 < 0.5**2).astype(int)

ax = plt.figure().add_subplot(projection='3d')
ax.voxels(sphere)
plt.title(f'Voxel 3D data: {sphere.shape} shape')
```

Extract patches from voxel 3D data.

```python
emp = EMPatches()
patches, indices  = emp.extract_patches(sphere, patchsize=8, overlap=0.0, stride=None, vox=True)

ax = plt.figure().add_subplot(projection='3d')
ax.voxels(patches[1])
plt.title(f'Patched Voxel 3D data: {patches[0].shape} shape')

for i in range(len(patches)):
    print(patches[i].shape)

mp = emp.merge_patches(patches, indices)

```
```
###############___VOXEL DATA___ setting vox to True ########################
##  shape     indices in xyz dimension
>> (8, 8, 8) (0, 8, 0, 8, 0, 8)
>> (8, 8, 8) (0, 8, 0, 8, 8, 16)
>> (8, 8, 8) (8, 16, 0, 8, 0, 8)
>> (8, 8, 8) (8, 16, 0, 8, 8, 16)
>> (8, 8, 8) (0, 8, 8, 16, 0, 8)
>> (8, 8, 8) (0, 8, 8, 16, 8, 16)
>> (8, 8, 8) (8, 16, 8, 16, 0, 8)
>> (8, 8, 8) (8, 16, 8, 16, 8, 16)
```
![alt text](https://github.com/Mr-TalhaIlyas/EMPatches/raw/main/screens/v4.png)

### *⚠️NOTE⚠️*
Here the output shape is 8x8x8 i.e. the croping is also done in D/C dimension unlike when we are doing image croping/patching in that case the output would have shape 8x8x3 (RGB) or 8x8 (grayscale), and incides would be like.

```
###############___PIXEL DATA___ -> setting vox to False ########################
##  shape     indices in xy dimension
>> (8, 8, 16) (0, 8, 0, 8)
>> (8, 8, 16) (8, 16, 0, 8)
>> (8, 8, 16) (0, 8, 8, 16)
>> (8, 8, 16) (8, 16, 8, 16)
```
![alt text](https://github.com/Mr-TalhaIlyas/EMPatches/raw/main/screens/v3.png)

## <a name="1D-patching">1D spectral Data patching</a>


```python
x1 = np.linspace(0.0, 5.0)
y1 = np.cos(5 * np.pi * x1) * np.exp(-x1)
plt.plot(y1)
plt.title('1D spectra')

emp = EMPatches()
patches, indices  = emp.extract_patches(y1, patchsize=8, overlap=0.0, stride=None)
```
![alt text](https://github.com/Mr-TalhaIlyas/EMPatches/raw/main/screens/1D.png)
```python
ax1 = plt.subplot(1)
plt.plot(patches[0]) # 0th patch
ax2 = plt.subplot(2, sharex=ax1, sharey=ax1)
plt.plot(patches[2]) # 2nd pathc
plt.suptitle('patched 1D spectra')
# merge again
mp = emp.merge_patches(patches, indices)
```
![alt text](https://github.com/Mr-TalhaIlyas/EMPatches/raw/main/screens/1dp.png)

## <a name="Batched-Patching">Batched Patching</a>

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

## <a name="Patching-via-Providing-Indices">Patching via Providing Indices</a>

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

For more infomration visit Homepage.
