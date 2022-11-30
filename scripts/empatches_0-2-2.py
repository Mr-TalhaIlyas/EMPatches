# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 15:32:58 2022

@author: talha
"""

import numpy as np
import math

class EMPatches(object):
    def __init__(self):
        pass

    def extract_patches(self, data, patchsize, overlap=None, stride=None, vox=False):
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

        Returns
        -------
        data_patches : a list containing extracted patches of images.
        indices : a list containing indices of patches in order, whihc can be used 
                at later stage for 'merging_patches'.

        '''

        dims = data.shape

        if len(dims)==1:        

            width = data.shape[0]
            maxWindowSize = patchsize
            windowSizeX = maxWindowSize
            windowSizeX = min(windowSizeX, width)
            
        elif len(dims)==2: 

            height = data.shape[0]
            width = data.shape[1]
            maxWindowSize = patchsize
            windowSizeX = maxWindowSize
            windowSizeY = maxWindowSize
            windowSizeX = min(windowSizeX, width)
            windowSizeY = min(windowSizeY, height)

        elif len(dims)==3:
            
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
            if len(dims)==1:
                stepSizeX = stride
            elif len(dims)==2:
                stepSizeX = stride
                stepSizeY = stride
            elif len(dims)==3:
                stepSizeX = stride
                stepSizeY = stride
                stepSizeZ = stride
                        
        elif overlap is not None:
            overlapPercent = overlap

            if len(dims)==1:
                windowSizeX = maxWindowSize     

                # If the input data is smaller than the specified window size,
                # clip the window size to the input size on both dimensions
                windowSizeX = min(windowSizeX, width)

                # Compute the window overlap and step size
                windowOverlapX = int(math.floor(windowSizeX * overlapPercent))

                stepSizeX = windowSizeX - windowOverlapX
                
            elif len(dims)==2:
                windowSizeX = maxWindowSize
                windowSizeY = maxWindowSize

                # If the input data is smaller than the specified window size,
                # clip the window size to the input size on both dimensions
                windowSizeX = min(windowSizeX, width)
                windowSizeY = min(windowSizeY, height)

                # Compute the window overlap and step size
                windowOverlapX = int(math.floor(windowSizeX * overlapPercent))
                windowOverlapY = int(math.floor(windowSizeY * overlapPercent))

                stepSizeX = windowSizeX - windowOverlapX
                stepSizeY = windowSizeY - windowOverlapY
                
            elif len(dims)==3:
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
            if len(dims)==1:
                stepSizeX = 1
            elif len(dims)==2:
                stepSizeX = 1
                stepSizeY = 1
            elif len(dims)==3:
                stepSizeX = 1
                stepSizeY = 1
                stepSizeZ = 1
        
        
        if len(dims)==1:

            # Determine how many windows we will need in order to cover the input data
            lastX = width - windowSizeX
            xOffsets = list(range(0, lastX+1, stepSizeX))
            
            # Unless the input data dimensions are exact multiples of the step size,
            # we will need one additional row and column of windows to get 100% coverage
            if len(xOffsets) == 0 or xOffsets[-1] != lastX:
                xOffsets.append(lastX)
            
            data_patches = []
            indices = []
            
            for xOffset in xOffsets:
                if len(data.shape) >= 3:
                    data_patches.append(data[(slice(xOffset, xOffset+windowSizeX, None))])
                else:
                    data_patches.append(data[(slice(xOffset, xOffset+windowSizeX))])
                    
                indices.append((xOffset, xOffset+windowSizeX))
                
        elif len(dims)==2:

            # Determine how many windows we will need in order to cover the input data
            lastX = width - windowSizeX
            lastY = height - windowSizeY
            xOffsets = list(range(0, lastX+1, stepSizeX))
            yOffsets = list(range(0, lastY+1, stepSizeY))
            
            # Unless the input data dimensions are exact multiples of the step size,
            # we will need one additional row and column of windows to get 100% coverage
            if len(xOffsets) == 0 or xOffsets[-1] != lastX:
                xOffsets.append(lastX)
            if len(yOffsets) == 0 or yOffsets[-1] != lastY:
                yOffsets.append(lastY)
            
            data_patches = []
            indices = []
            
            for xOffset in xOffsets:
                for yOffset in yOffsets:
                    if len(data.shape) >= 3:
                        data_patches.append(data[(slice(yOffset, yOffset+windowSizeY, None),
                                                slice(xOffset, xOffset+windowSizeX, None))])
                    else:
                        data_patches.append(data[(slice(yOffset, yOffset+windowSizeY),
                                                slice(xOffset, xOffset+windowSizeX))])
                        
                    indices.append((yOffset, yOffset+windowSizeY, xOffset, xOffset+windowSizeX))            
        
        elif len(dims)==3:
            
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
            
            data_patches = []
            indices = []

            if not vox: # for images 
                for xOffset in xOffsets:
                    for yOffset in yOffsets:
                        if len(data.shape) >= 3:
                            data_patches.append(data[(slice(yOffset, yOffset+windowSizeY, None),
                                                    slice(xOffset, xOffset+windowSizeX, None))])
                        else:
                            data_patches.append(data[(slice(yOffset, yOffset+windowSizeY),
                                                    slice(xOffset, xOffset+windowSizeX))])
                        
                        indices.append((yOffset, yOffset+windowSizeY, xOffset, xOffset+windowSizeX))
            if vox: # for volumetric data
                for xOffset in xOffsets:
                    for yOffset in yOffsets:
                        for zOffset in zOffsets:
                            if len(data.shape) >= 4:
                                data_patches.append(data[(slice(yOffset, yOffset+windowSizeY, None),
                                                        slice(xOffset, xOffset+windowSizeX, None),
                                                        slice(zOffset, zOffset+windowSizeZ, None))])
                            else:
                                data_patches.append(data[(slice(yOffset, yOffset+windowSizeY),
                                                        slice(xOffset, xOffset+windowSizeX),
                                                        slice(zOffset, zOffset+windowSizeZ))])
                                
                            indices.append((yOffset, yOffset+windowSizeY, xOffset, xOffset+windowSizeX, zOffset, zOffset+windowSizeZ))   
        
        return data_patches, indices


    def merge_patches(self, data_patches, indices, mode='overwrite'):
        '''
        Parameters
        ----------
        data_patches : list containing image patches that needs to be joined, dtype=uint8
        indices : a list of indices generated by 'extract_patches' function of the format;
                    (yOffset, yOffset+windowSizeY, xOffset, xOffset+windowSizeX)
        mode : how to deal with overlapping patches;
                overwrite -> next patch will overwrite the overlapping area of the previous patch.
                max -> maximum value of overlapping area at each pixel will be written.
                min -> minimum value of overlapping area at each pixel will be written.
                avg -> mean/average value of overlapping area at each pixel will be written.
        Returns
        -------
        Stitched image.
        '''
        modes = ["overwrite", "max", "min", "avg"]
        if mode not in modes:
            raise ValueError(f"mode has to be either one of {modes}, but got {mode}")

        dims = len(indices[-1])
        
        if dims==2:
            orig_h = indices[-1][1]
        elif dims==4:
            orig_h = indices[-1][1]
            orig_w = indices[-1][3]
        elif dims==6:
            orig_h = indices[-1][1]
            orig_w = indices[-1][3]
            orig_d = indices[-1][5]
        
        ### There is scope here for rgb/hyperspectral volume (i.e. 4D -> 3 spatial and 1 spectral dimensions, simplest case is only 3 channles for the spectral dimension)
        rgb = True
        if len(data_patches[0].shape) == 2:
            rgb = False
        
        if mode == 'min':
            if dims == 2:
                empty_data = np.zeros((orig_h)).astype(np.float32) + np.inf # using float here is better
                
            elif dims==4:
                if rgb:
                    empty_data = np.zeros((orig_h, orig_w, data_patches[0].shape[-1])).astype(np.float32) + np.inf # using float here is better
                else:
                    empty_data = np.zeros((orig_h, orig_w)).astype(np.float32) + np.inf # using float here is better

            elif dims==6:
                empty_data = np.zeros((orig_h, orig_w, orig_d)).astype(np.float32) + np.inf # using float here is better
                
        else:
            if dims == 2:
                empty_data = np.zeros((orig_h)).astype(np.float32) # using float here is better
                
            elif dims==4:
                if rgb:
                    empty_data = np.zeros((orig_h, orig_w, data_patches[0].shape[-1])).astype(np.float32) # using float here is better
                else:
                    empty_data = np.zeros((orig_h, orig_w)).astype(np.float32) # using float here is better

            elif dims==6:
                empty_data = np.zeros((orig_h, orig_w, orig_d)).astype(np.float32) # using float here is better

        for i, indice in enumerate(indices):

            if mode == 'overwrite':

                if dims == 2:
                    empty_data[indice[0]:indice[1]] = data_patches[i]

                elif dims == 4:
                    if rgb:
                        empty_data[indice[0]:indice[1], indice[2]:indice[3], :] = data_patches[i]
                    else:
                        empty_data[indice[0]:indice[1], indice[2]:indice[3]] = data_patches[i]
                        
                elif dims==6:
                        empty_data[indice[0]:indice[1], indice[2]:indice[3], indice[4]:indice[5]] = data_patches[i]
                        
                        
            elif mode == 'max':

                if dims == 2:
                    empty_data[indice[0]:indice[1]] = np.maximum(data_patches[i], empty_data[indice[0]:indice[1]])
                elif dims == 4:
                    if rgb:
                        empty_data[indice[0]:indice[1], indice[2]:indice[3], :] = np.maximum(data_patches[i], empty_data[indice[0]:indice[1], indice[2]:indice[3], :])
                    else:
                        empty_data[indice[0]:indice[1], indice[2]:indice[3]] = np.maximum(data_patches[i], empty_data[indice[0]:indice[1], indice[2]:indice[3]])
                elif dims==6:
                        empty_data[indice[0]:indice[1], indice[2]:indice[3], indice[4]:indice[5]] = np.maximum(data_patches[i], empty_data[indice[0]:indice[1], indice[2]:indice[3], indice[4]:indice[5]])


            elif mode == 'min':
                if dims == 2:
                    empty_data[indice[0]:indice[1]] = np.minimum(data_patches[i], empty_data[indice[0]:indice[1]])
                elif dims == 4:
                    if rgb:
                        empty_data[indice[0]:indice[1], indice[2]:indice[3], :] = np.minimum(data_patches[i], empty_data[indice[0]:indice[1], indice[2]:indice[3], :])
                    else:
                        empty_data[indice[0]:indice[1], indice[2]:indice[3]] = np.minimum(data_patches[i], empty_data[indice[0]:indice[1], indice[2]:indice[3]])
                elif dims==6:
                        empty_data[indice[0]:indice[1], indice[2]:indice[3], indice[4]:indice[5]] = np.minimum(data_patches[i], empty_data[indice[0]:indice[1], indice[2]:indice[3], indice[4]:indice[5]])
                    
            elif mode == 'avg':

                if dims == 2:
                    empty_data[indice[0]:indice[1]] = np.where(empty_data[indice[0]:indice[1]] == 0,
                                                                                    data_patches[i], 
                                                                                    np.add(data_patches[i],empty_data[indice[0]:indice[1]])/2)
                elif dims == 4:
                    if rgb:
                        empty_data[indice[0]:indice[1], indice[2]:indice[3], :] = np.where(empty_data[indice[0]:indice[1], indice[2]:indice[3], :] == 0,
                                                                                            data_patches[i], 
                                                                                            np.add(data_patches[i],empty_data[indice[0]:indice[1], indice[2]:indice[3], :])/2)
                        # Below line should work with np.ones mask but giving Weights sum to zero error and is approx. 10 times slower then np.where
                        # empty_data[indice[0]:indice[1], indice[2]:indice[3], :] = np.average(([empty_data[indice[0]:indice[1], indice[2]:indice[3], :],
                        #                                                                         data_patches[i]]), axis=0,
                        #                                                                         weights=(np.asarray([empty_data[indice[0]:indice[1], indice[2]:indice[3], :],
                        #                                                                                               data_patches[i]])>0))
                    else:
                        empty_data[indice[0]:indice[1], indice[2]:indice[3]] = np.where(empty_data[indice[0]:indice[1], indice[2]:indice[3]] == 0,
                                                                                        data_patches[i], 
                                                                                        np.add(data_patches[i],empty_data[indice[0]:indice[1], indice[2]:indice[3]])/2)
                elif dims==6:
                        empty_data[indice[0]:indice[1], indice[2]:indice[3], indice[4]:indice[5]] = np.where(empty_data[indice[0]:indice[1], indice[2]:indice[3], indice[4]:indice[5]] == 0,
                                                                                        data_patches[i], 
                                                                                        np.add(data_patches[i],empty_data[indice[0]:indice[1], indice[2]:indice[3], indice[4]:indice[5]])/2)

        return empty_data

class BatchPatching(EMPatches):
    def __init__(self, patchsize, overlap=None, stride=None, typ='tf', vox=False):
        '''
        Parameters
        ----------
        patchsize :  size of patch to extract from image only square patches can be
                     extracted for now.
        overlap (Optional): overlap between patched in percentage a float between [0, 1].
        stride (Optional): Step size between patches
        type: Type of batched images tf or torch type
        '''
        super().__init__()
        self.patchsize = patchsize
        self.overlap = overlap
        self.stride = stride
        self.typ = typ
        self.vox = vox

    def patch_batch(self, batch):
        '''
        Parameters
        ----------
        batch : Batch of images of shape either BxCxHxW -> pytorch or BxHxWxC -> tf
                to extract patches. For 1D spectra or batchs shape should be BxD.
        Returns
        -------
        batch_patches : a list containing lists of extracted patches of images.
        batch_indices : a list containing lists of indices of patches in order, whihc can be used 
                  at later stage for 'merging_patches'.
    
        '''
        if len(batch.shape) != 2:
            typs = ["tf", "torch"]
            if self.typ not in typs:
                raise ValueError(f"mode has to be either one of {typs}, but got {self.typ}. For numpy also use 'tf' type.")
            if len(batch.shape) != 4:
                raise ValueError(f'Input batch should be of shape BxDxHxW or BxHxWxD i.e. 4D for image or volumetric data or BxD i.e. 2D for 1D spectral data, but got {len(batch.shape)} dims')
            
            if self.typ == 'torch':
                batch = batch.permute(0,2,3,1)
        else:
            pass
        
        img_list = list(batch)

        b_patches, b_indices = [], []
        for i in range(len(img_list)):
            patches, indices = super().extract_patches(img_list[i], self.patchsize, self.overlap, self.stride, self.vox)
            b_patches.append(patches)
            b_indices.append(indices)
        
        return b_patches, b_indices

    def merge_batch(self, b_patches, b_indices, mode='overwrite'):
        '''
        Parameters
        ----------
        b_patches : list containing lists of patches of images to be merged together
                    e.g. list(list1, list2, ...), where, list1->([H W C], [H W C], ...) and so on.
        b_indices : list containing lists of indices of images to be merged in format as return by
                    patch_batch method.
        Returns
        -------
        merged_batch : a np array of shape BxCxHxW -> pytorch or BxHxWxC -> tf.
        
        '''
        m_patches = []
        for p, i in zip(b_patches, b_indices):
            m = super().merge_patches(p, i, mode)
            m_patches.append(m)

        m_patches = np.asarray(m_patches)
        
        if self.typ == 'torch':
            m_patches = m_patches.permute(0,3,2,1)

        return m_patches
        

def patch_via_indices(data, indices):
    '''
        Parameters
        ----------
        img : array to extract patches from; it can be 1D, 2D or 3D [W, H, D]. H: Height, W: Width, D: Depth.
              3D data includes images (RGB, RGBA, etc) or Voxel data.
        indices :   list of indices/tuple of 4 e.g;
                    [(ystart, yend, xstart, xend, zstart, zend), -> indices of 1st patch 
                     (ystart, yend, xstart, xend, zstart, zend), -> indices of 2nd patch
                     ...] -> for 3D data
        Returns
        -------
        img_patches : a list containing extracted patches of image.
        '''
    dims = len(indices[-1])

    data_patches=[]

    if dims==2:
    
        for indice in indices:
            data_patches.append(data[(slice(indice[0], indice[1]))])
        
    elif dims==4:
        
        for indice in indices:
            if len(data.shape) >= 3:
                data_patches.append(data[(slice(indice[0], indice[1], None),
                                        slice(indice[2], indice[3], None))])
            else:
                data_patches.append(data[(slice(indice[0], indice[1]),
                                        slice(indice[2], indice[3]))])
            
    elif dims==6:            
            
        for indice in indices:
            
            data_patches.append(data[(slice(indice[0], indice[1]),
                                    slice(indice[2], indice[3]),
                                    slice(indice[3], indice[4]))])
    
    return data_patches
