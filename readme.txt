 written by Mattias Heinrich.
 Copyright (c) 2016. All rights reserved.
 See the LICENSE.txt file in the root folder
 
 contact: heinrich (at) imi (dot) uni-luebeck (dot) de
          www.mpheinrich.de
 
 If you use this implementation or parts of it please cite:
 
 "Multi-Organ Segmentation using Vantage Point Forests and Binary Context Features"
 by Mattias P. Heinrich and Max Blendowski.
 in Medical Image Computing and Computer Assisted Intervention (MICCAI) 2016. 
 LNCS Springer (2016)
 
The main script vantagePointForest_Segmentation.m requires the following input (all scans should have similar voxel-spacings):
- scansTrain    (cell array of 3D volumes of grayvalue training scans)
- segmentsTrain (cell array of 3D volumes of same size with GT segmentations of scans)
- masksTrain    (cell array of 3D volumes of masks - region of interest)

- scanTest      (a 3D volume of test scan)
- maskTest      (region of interest for the above)

- output is a 3D volume segmentTest - the segmentation of the given scan

 This is intended for research purposes only and should not be used in clinical applications (no warranty), see license.txt!

If you find any bugs or have questions please don’t hesitate to contact me by email.
