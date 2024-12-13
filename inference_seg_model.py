import numpy as np
import math
import tempfile
import shutil
import os
import cv2
class EMPatches_Effi_Seg_Inference(object):
    
    def __init__(self):
        self.temp_dir = None
        self.temp_dir_path = None

    def cleanup(self):
        if self.temp_dir:
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None

    def extract_patches(self, data, patchsize, overlap=None, stride=None, vox=False ,base_temp_dir=None):
        
        if base_temp_dir is not None:
            self.temp_dir = tempfile.mkdtemp(dir=base_temp_dir)
        else:
            self.temp_dir = tempfile.mkdtemp()

        '''
        Parameters
        ----------
        data : array to extract patches from; it can be 1D, 2D or 3D [W, H, D]. H: Height, W: Width, D: Depth,
               3D data includes images (RGB, RGBA, etc) or Voxel data.
        patchsize :  size of patch to extract from image only square patches can be
                    extracted for now.
        overlap (Optional): overlap between patched in percentage a float between [0, 1].
        stride (Optional): Step size between patches
        vox (Optional): Whether data is volumetric or not if set to true array will be cropped in last dimension too.
        
        base_temp_dir (Optional) : temporary storage to save the patches and can be deleted later using  _ = emp.cleanup()

        Returns
        -------
        temp_dir : Paths where the patches are saved into temporary memory.
        indices : a list containing indices of patches in order, whihc can be used 
                at later stage for reconstruction.
                
        Orignal Dimenssions:  (height,width,depth) of orignal shape of data

        '''

        height = data.shape[0]
        width = data.shape[1]
        depth = data.shape[2]

        maxWindowSize = patchsize
        windowSizeX = maxWindowSize
        windowSizeY = maxWindowSize
        windowSizeZ = maxWindowSize

        windowSizeX = min(windowSizeX, width)
        windowSizeY = min(windowSizeY, height)
        windowSizeZ = min(windowSizeZ, depth)
            
        if stride is not None:
                stepSizeX = stride
                stepSizeY = stride
                stepSizeZ = stride
                        
        elif overlap is not None:
            overlapPercent = overlap

            windowSizeX = maxWindowSize
            windowSizeY = maxWindowSize
            windowSizeZ = maxWindowSize
            
            # If the input data is smaller than the specified window size,
            # clip the window size to the input size on both dimensions
            windowSizeX = min(windowSizeX, width)
            windowSizeY = min(windowSizeY, height)
            windowSizeZ = min(windowSizeZ, depth)

            # Compute the window overlap and step size
            windowOverlapX = int(math.floor(windowSizeX * overlapPercent))
            windowOverlapY = int(math.floor(windowSizeY * overlapPercent))
            windowOverlapZ = int(math.floor(windowSizeZ * overlapPercent))

            stepSizeX = windowSizeX - windowOverlapX
            stepSizeY = windowSizeY - windowOverlapY                
            stepSizeZ = windowSizeZ - windowOverlapZ                

        else:
            stepSizeX = 1
            stepSizeY = 1
            stepSizeZ = 1
         
        # Determine how many windows we will need in order to cover the input data
        lastX = width - windowSizeX
        lastY = height - windowSizeY
        lastZ = depth - windowSizeZ
        
        xOffsets = list(range(0, lastX+1, stepSizeX))
        yOffsets = list(range(0, lastY+1, stepSizeY))
        zOffsets = list(range(0, lastZ+1, stepSizeZ))
        
        # Unless the input data dimensions are exact multiples of the step size,
        # we will need one additional row and column of windows to get 100% coverage
        if len(xOffsets) == 0 or xOffsets[-1] != lastX:
            xOffsets.append(lastX)
        if len(yOffsets) == 0 or yOffsets[-1] != lastY:
            yOffsets.append(lastY)
        if len(zOffsets) == 0 or zOffsets[-1] != lastZ:
            zOffsets.append(lastZ)
        indices = []
        
        patch_index = 0

        for xOffset in xOffsets:
            for yOffset in yOffsets:
                #if len(data.shape) >= 3:
                patch_path = os.path.join(self.temp_dir, f"patch_{patch_index}.png")
                cv2.imwrite(patch_path, data[(slice(yOffset, yOffset+windowSizeY, None),
                                        slice(xOffset, xOffset+windowSizeX, None))])
                patch_index += 1    
                indices.append((yOffset, yOffset+windowSizeY, xOffset, xOffset+windowSizeX))

        return self.temp_dir, indices , (height,width,depth)

emp = EMPatches_Effi_Seg_Inference()# Directories and input image

output_path = "path to save segmentaiton mask" + "reconstructed_seg_mask.png"
image_path = "path to the input RGB image"
temp_dir = "Optional to save temporray patches"


image = cv2.imread(image_path)
patches_path, indices , org_shape = emp.extract_patches(image,patchsize=224, overlap=0.0 , base_temp_dir=temp_dir)


from torch import nn
class Seg_Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
model = Seg_Model()  

reconstructed_seg_mask= np.zeros((org_shape), dtype=np.uint8)
# Iterate through patch files
for i, patch_file in enumerate(sorted(os.listdir(patches_path))):
    patch_path = os.path.join(patches_path, patch_file)
    patch = cv2.imread(patch_path)
    y = model(patch)
    y = np.stack((y,)*3, axis=-1)
    y_start, y_end, x_start, x_end = indices[i]
    reconstructed_seg_mask[y_start:y_end, x_start:x_end] = y
    del patch,y

cv2.imwrite(output_path, reconstructed_seg_mask)
print(f"Reconstructed segmentaiton  saved to: {output_path}")
_ = emp.cleanup()   ## to delete the patches from memory 
