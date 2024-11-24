# Differential Tomography SAR (D-TomoSAR)

<p style="text-align:justify;">
Differential Tomographic Synthetic Aperture Radar (D-TomoSAR) is a powerful technique in remote sensing that combines principles of Differential Interferometry SAR (DInSAR) and SAR Tomography (TomoSAR) to achieve detailed three-dimensional (3D) and temporal deformation analysis of a scene. It is particularly useful for monitoring complex urban environments, infrastructure stability, and natural phenomena like landslides or subsidence.
</p>

# Key Principles and Components of D-TomoSAR :

+ `Synthetic Aperture Radar (SAR)`:

    + SAR is a radar imaging technique that captures high-resolution 2D images of the Earth's surface by synthesizing a large aperture from successive radar signals.
    + It provides amplitude and phase information of backscattered signals, which are critical for interferometric and tomographic processing.

+ `SAR Interferometry (InSAR)`:

    + InSAR uses the phase difference between two SAR images acquired from slightly different positions to extract elevation or surface deformation.
    + Differential InSAR (DInSAR) extends this to detect millimeter-level deformations over time, such as subsidence or uplift.

+ `SAR Tomography (TomoSAR)`:

    + TomoSAR expands SAR into the vertical dimension (height) by using multiple acquisitions to reconstruct a 3D representation of the scene.
    + It achieves this by forming a synthetic aperture in the elevation direction through multiple spatial baselines.

+ `Differential Tomography SAR (D-TomoSAR)`:

    + D-TomoSAR combines DInSAR and TomoSAR techniques, allowing for the simultaneous extraction of 3D spatial structures and temporal deformation.
    + It separates scatterers within a single resolution cell, resolves their heights, and tracks their deformation over time.


# Working Mechanism of D-TomoSAR

+ `Data Acquisition`:

    + Multiple SAR images are acquired over the same region at different times and slightly different viewing angles.
    + The acquisition geometry (spatial baselines) and temporal information (time intervals) form the foundation of D-TomoSAR.

+ `Pre-Processing`:

    + Coregistration: Aligning SAR images to ensure pixel-to-pixel correspondence.
    + Phase Calibration: Correcting for phase inconsistencies caused by system errors or atmospheric disturbances.

+ `Interferogram Formation`:

    + Pairwise SAR images are processed to form interferograms, which encode phase differences.
    + These interferograms contain information about both height and deformation.

+ `Spectral Analysis in the Elevation Dimension`:

    + The synthetic aperture in the elevation dimension is formed using the spatial baselines of the SAR acquisitions.
    + By applying spectral estimation techniques like Singular Value Decomposition (SVD) or adaptive beamforming, the vertical distribution of scatterers is reconstructed.

+ `Differential Analysis`:

    + Temporal phase changes caused by deformation are separated from the elevation phase using the temporal information in the dataset.
    + Advanced algorithms, such as Persistent Scatterer Interferometry (PSI), are often integrated to track stable reflectors over time.

+ `Output`:

    + 3D Point Clouds: A dense 3D reconstruction of the scene.
    + Deformation Maps: Temporal deformation profiles of scatterers.
    + Multi-Layer Reconstruction: Resolving multiple scatterers within a single resolution cell.

# Advantages of D-TomoSAR
+ `High Precision`:

    + Millimeter-level accuracy in deformation monitoring.
    + Detailed 3D reconstruction of complex environments.

+ `Separation of Scatterers`:

    + Resolves multiple scatterers within a single SAR resolution cell, essential in urban areas with dense infrastructure.

+ `Temporal Analysis`:

    + Tracks changes over time, making it invaluable for long-term monitoring applications.

+ `Wide Area Coverage`:

    + Can monitor large regions compared to ground-based techniques.

# Applications of D-TomoSAR
+ `Urban Infrastructure Monitoring`:

    + Assessing the stability of buildings, bridges, and other structures in cities.
    + Identifying areas of subsidence or structural deformation.

+ `Landslide and Subsidence Detection`:

    + Monitoring gradual ground movements in landslide-prone areas.
    + Detecting subsidence due to natural or anthropogenic activities (e.g., mining, groundwater extraction).

+ `Environmental Monitoring`:

    + Tracking changes in vegetation, glacier movement, or snow accumulation.

+ `Disaster Management`:

    + Providing pre- and post-event analysis for earthquakes, floods, or volcanic activities.

Resources:

+ [Tomographic SAR](./EN-SET-191-08.pdf) | @github/[TomoSAR](https://github.com/DinhHoTongMinh/TomoSAR)
+ [Seasonal Deformation and Accelerated Motion of Infrastructure Monitoring Using a Generalized Differential SAR Tomography](https://ieeexplore.ieee.org/document/9014555)
+ [High-Resolution 3-D and 4-D SAR Imaging-The Case Study of Shenzhen](https://ieeexplore.ieee.org/document/10028456)
+ [TomoSAR 3D Reconstruction for Buildings Using Very Few Tracks of Observation: A Conditional Generative Adversarial Network Approach](https://www.mdpi.com/2072-4292/13/24/5055)
+ [Three-Dimensional Deformation Monitoring of Urban Infrastructure by Tomographic SAR Using Multitrack TerraSAR-X Data Stacks](https://ieeexplore.ieee.org/document/7548332)
+ [Extended D-TomoSAR Displacement Monitoring for Nanjing (China) City Built Structure Using High-Resolution TerraSAR/TanDEM-X and Cosmo SkyMed SAR Data](https://www.mdpi.com/2072-4292/11/22/2623)
+ [Elevation Extraction from Spaceborne SAR Tomography Using Multi-Baseline COSMO-SkyMed SAR Data](https://www.mdpi.com/2072-4292/14/16/4093)
+ [DEFORMATION MONITORING OF URBAN INFRASTRUCTURE BY TOMOGRAPHIC SAR USING MULTI-VIEW TERRASAR-X DATA STACKS](https://proceedings.esa.int/files/314.pdf)
+ [Very High Resolution Tomographic SAR Inversion for Urban Infrastructure Monitoring — A Sparse and Nonlinear Tour](https://mediatum.ub.tum.de/doc/1071713/1071713.pdf)
+ [A TomoSAR regularization-based method for height change detection in urban areas](https://www.sciencedirect.com/science/article/pii/S1569843224002061)
+ [TomoSense: A unique 3D dataset over temperate forest combining multi-frequency mono- and bi-static tomographic SAR with terrestrial, UAV and airborne lidar, and in-situ forest census](https://www.researchgate.net/publication/369977717_TomoSense_A_unique_3D_dataset_over_temperate_forest_combining_multi-frequency_mono-_and_bi-static_tomographic_SAR_with_terrestrial_UAV_and_airborne_lidar_and_in-situ_forest_census)
+ [Tomo-GENESIS: DLR's tomographic SAR processing system](https://ieeexplore.ieee.org/document/6550690)
+ [High-Resolution and Wide-Swath 3D Imaging for Urban Areas Based on Distributed Spaceborne SAR](https://www.mdpi.com/2072-4292/15/16/3938)
+ [Research on 4-D Imaging of Holographic SAR Differential Tomography](https://www.mdpi.com/2072-4292/15/13/3421)