# Optical Stereo 3D

<p style="text-align:justify;">
Optical Stereo 3D from SAR (Synthetic Aperture Radar) involves combining SAR data with optical stereo imaging techniques to achieve high-resolution three-dimensional (3D) models of the Earth's surface or structures. It is a hybrid approach that leverages the unique advantages of both SAR and optical imagery.
</p>

## Overview of Optical Stereo 3D and SAR
+ `Optical Stereo Imaging`: This is a photogrammetric technique that uses overlapping optical images (stereo pairs) taken from slightly different angles to calculate depth and generate 3D models. It relies on visible or infrared light and requires good lighting conditions.

+ `SAR Imaging`: SAR generates 2D images by actively transmitting microwave signals and recording their echoes. It operates independently of lighting conditions and can penetrate clouds, vegetation, and some structures, making it useful for all-weather imaging.

By fusing SAR with optical stereo techniques, researchers can overcome the limitations of each modality. For instance, SAR data enhances the 3D depth precision, while optical imagery provides higher spatial resolution and detail.

## Key Components and Workflow
+ Data Collection:

    + `Optical Data`: High-resolution satellite or airborne images from systems like WorldView, Pleiades, or SPOT are commonly used.
    + `SAR Data`: Data from SAR systems such as Sentinel-1, TerraSAR-X, or RADARSAT is collected. This could be single-pass or multi-pass SAR data for interferometric analysis.

+ Preprocessing:

    + `Optical Data`:
        + Georeferencing: Aligning optical images with geographic coordinates.
        + Radiometric Correction: Adjusting for variations in lighting or sensor sensitivity.
        + Stereo Matching: Identifying corresponding points in overlapping stereo images.
    + `SAR Data`:
        + Coregistration: Aligning SAR images for coherent analysis.
        + Speckle Filtering: Reducing noise inherent in SAR data.

+ 3D Model Generation:

    + `Depth Estimation`:
        + For optical images, depth is derived using the parallax effect from stereo pairs.
        + SAR data contributes elevation or displacement data via interferometry or tomographic SAR techniques.
    + `Data Fusion`:
        + Algorithms like weighted averaging, Kalman filters, or deep learning models combine optical and SAR-derived depth maps into a unified 3D representation.
    + `Post-Processing`:

        + Filtering and smoothing techniques enhance the final 3D model.
        + Integrating additional data (e.g., LiDAR or GNSS measurements) improves accuracy.

Research :
+ [A framework for SAR-optical stereogrammetry over urban areas](https://www.sciencedirect.com/science/article/pii/S0924271618302739)
+ [Towards the Reconstruction of Prismatic Building Models by SAR-Optical Stereogrammetry](https://ieeexplore.ieee.org/document/8808985)
+ [Merging of SAR and optical features for 3D reconstruction in a radargrammetric framework](https://perso.telecom-paristech.fr/tupin/PUB/igarss04.pdf)
+ [Fusing Spaceborne SAR Interferometry and Street View Images for 4D Urban Modeling](https://ieeexplore.ieee.org/document/8455498)
+ @github/[3d-stereo-vision](https://github.com/agavazov/3d-stereo-vision), /[Stereo 3D Reconstruction](https://github.com/hzxie/Stereo-3D-Reconstruction)
+ [3D-Reconstruction-with-Deep-Learning-Methods](https://github.com/natowi/3D-Reconstruction-with-Deep-Learning-Methods)
+ [3D from Stereo - 5 Minutes with Cyrill](https://youtu.be/8lcwwfFcp-I?si=9jVCiLAIVDaDfgiF) | [3D Reconstruction from Stereo](https://www.youtube.com/playlist?list=PLiOyhn3fx76z-zUZAjGhmonMNrDC4oF15)
+ [Computer Vision - Lecture 4.1 (Stereo Reconstruction: Preliminaries)](https://youtu.be/6hr6xpOU-uw?si=oeZHtd22wUKAPaxM)
+ [3D Computer Vision: From Stereo Vision to 3D Reconstruction](https://www.thinkautonomous.ai/blog/3d-computer-vision/)
+ [3D Reconstruction with Stereo Images -Part 1: Camera Calibration](https://medium.com/@dc.aihub/3d-reconstruction-with-stereo-images-part-1-camera-calibration-d86f750a1ade)
+ @github/[Stereo-images-to-3D-reconstruction](https://github.com/AttilioLughetta/Stereo-images-to-3D-reconstruction), /[opencv](https://github.com/saurabhkemekar/Stereo-3D-Reconstruction), /[2dtoStereo](https://github.com/youngseok-seo/stereo-vision), /[StereoVision3DReconstruction
](https://github.com/jimazeyu/StereoVision3DReconstruction), /[stereo_dense_reconstruction](https://github.com/ut-amrl/stereo_dense_reconstruction).