# Pixel Offset Tracking (POT) :

Pixel Offset Tracking (POT), also known as sub-pixel correlation or intensity tracking, is a remote sensing technique used to measure large-scale displacements of the Earth's surface. Unlike Interferometric Synthetic Aperture Radar (InSAR), which relies on phase differences, POT leverages the intensity and coherence of radar images to track displacements by detecting shifts in pixel positions.

<img src="./img/pot.webp" width=100%>

## Key Features of Pixel Offset Tracking :
+ `Capability for Large Displacements`: POT is particularly useful in areas where displacement exceeds the coherence limit of InSAR (e.g., during earthquakes, glacier movement, or landslides).
+ `Intensity-Based Approach`: Instead of phase, POT uses image intensity and patterns to measure offsets, making it robust in areas of low coherence.
+ `Wide Applications`: Useful for monitoring ice sheets, landslides, fault zones, and volcanic eruptions.
+ `Less Sensitive to Temporal Decorrelation`: POT is less affected by changes in surface characteristics over time compared to InSAR.

## How Pixel Offset Tracking Works
+ SAR Image Pair Selection:

    + Two radar images (master and slave) of the same region are acquired at different times.
    + The images must have overlapping regions and sufficient resolution.

+ Image Preprocessing:

    + The images are co-registered to align them spatially.
    + Sub-pixel accuracy is critical to ensure reliable offset measurements.

+ Cross-Correlation Analysis:

    + The master and slave images are divided into small patches (e.g., 64x64 or 128x128 pixels).
    + A 2D cross-correlation function is calculated for each patch to find the offset between corresponding features in the two images.

+ Offset Calculation:

    + The position of the peak in the cross-correlation matrix indicates the offset (shift in pixels).
    + Both range (across-track) and azimuth (along-track) offsets are calculated.

+ Geocoding:

    + The pixel offsets are converted to real-world displacements in meters using sensor parameters, acquisition geometry, and orbital data.

+ Visualization and Interpretation:

    + Displacement maps are generated to visualize movement patterns.
    + These maps are used to analyze phenomena like fault slip, glacier dynamics, or lava flow.

## Advantages of Pixel Offset Tracking
+ Handles Large Displacements:

    + Unlike InSAR, which is limited by phase unwrapping errors for displacements greater than a few centimeters, POT can measure displacements in meters.

+ Works in Low-Coherence Areas:

    + Useful in environments with rapid changes, such as glaciers or deserts, where InSAR coherence is poor.

+ Broad Applicability:

    + Effective for monitoring natural hazards (e.g., earthquakes, landslides) and long-term processes (e.g., glacier flow, sediment transport).

## Applications of Pixel Offset Tracking

+ Earthquake Monitoring:

    + Measures fault slip and surface displacement caused by seismic events.
    + Provides insights into earthquake mechanics and fault behavior.

+ Glacier and Ice Sheet Movement:

    + Tracks the flow of glaciers and ice sheets over time.
    + Helps in understanding climate change impacts on polar regions.

+ Landslide Dynamics:
 
    + Identifies the extent and velocity of landslides, aiding in hazard assessment.

+ Volcanic Activity:

    + Monitors ground deformation caused by magma intrusion or lava flow.
    + Useful in predicting volcanic eruptions.

+ Infrastructure Monitoring:

    + Tracks displacement of bridges, dams, or buildings due to natural or man-made activities.


| **Step**               | **Description**                                                                                                  | **Challenges**                                                                                     |
|-------------------------|------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| **1. Image Pair Selection** | Select two radar images (master and slave) with adequate temporal and spatial baselines. These images must cover the same geographic area and should be acquired under similar conditions to minimize errors. | Ensuring sufficient overlap between images and choosing an optimal baseline to avoid decorrelation or geometric distortions. |
| **2. Co-Registration**  | Align the master and slave images with sub-pixel accuracy to ensure the features in both images overlap precisely. This step involves geometric corrections and resampling. | Accurate alignment is computationally intensive, especially for high-resolution SAR data. Errors here propagate through the entire process. |
| **3. Patch Division**   | Divide the co-registered images into small overlapping patches (e.g., 64x64 or 128x128 pixels) for localized analysis. This step facilitates detecting local offsets within the region of interest. | Selecting an optimal patch size is critical; too small may increase noise, while too large may smooth out fine displacements. |
| **4. Cross-Correlation**| Compute the cross-correlation function for each patch to determine the relative shift (offset) between corresponding features in the two images. A 2D correlation matrix is used to identify peak displacement values. | Noise, temporal decorrelation, and surface changes can reduce correlation strength, leading to ambiguous or inaccurate results. |
| **5. Offset Extraction**| Extract displacement values by identifying the location of the correlation peak in the matrix. Sub-pixel accuracy is achieved using interpolation techniques like parabolic fitting. | Weak correlation peaks can lead to erroneous offset estimation, especially in low-coherence or featureless regions. |
| **6. Geocoding**        | Convert the pixel offsets into real-world coordinates and physical distances (meters). This requires precise sensor parameters, orbit data, and a Digital Elevation Model (DEM) to correct for topographic effects. | Geocoding accuracy depends on high-quality DEMs and accurate SAR metadata. Errors in these inputs lead to systematic displacement errors. |
| **7. Noise Filtering**  | Apply filtering techniques (e.g., median filtering, low-pass filtering) to reduce noise and improve the quality of the displacement map. | Over-filtering may remove genuine displacement signals, while under-filtering may leave noise that affects interpretation. |
| **8. Visualization**    | Generate displacement maps (e.g., range offset, azimuth offset) to visualize movement patterns. Use color-coding or vector fields for better interpretation. | Effective visualization depends on choosing the right scale and format, which can vary based on the application and displacement magnitude. |
| **9. Error Assessment** | Validate the results by comparing the offset measurements with independent data (e.g., GPS measurements, ground truth data). Assess error margins and uncertainties. | Lack of ground truth data can make validation challenging, especially in remote or inaccessible areas. |
| **10. Displacement Conversion** | Analyze the offset data to compute total displacement vectors by combining range and azimuth offsets. This step provides a 2D or 3D representation of movement. | Requires careful handling of coordinate transformations and sensor geometry to avoid introducing projection errors. |
| **11. Interpretation**  | Interpret the displacement maps to infer the physical processes causing the observed changes (e.g., fault slip, glacier flow, landslide dynamics). Generate reports or integrate results into geospatial models for further analysis. | Interpretation requires domain expertise, as misinterpreting displacement patterns can lead to incorrect conclusions about the underlying processes. |


Resources : [POT-mintpy](https://github.com/forrestfwilliams/pixeloffset_mintpy), [Evaluation of the Use of Sub-Pixel Offset Tracking Techniques to Monitor Landslides in Densely Vegetated Steeply Sloped Areas](https://www.mdpi.com/2072-4292/8/8/659), [Time-Series SBAS Pixel Offset Tracking Method for Monitoring Three-Dimensional Deformation in a Mining Area](https://ieeexplore.ieee.org/document/9123340), [Beyond the phase: High-resolution Pixel Offset Tracking to complement InSAR](https://satelliteblog.cgg.com/beyond-the-phase/), [ID533 Landslide Characterization Using Hybrid Spaceborne and Airborne InSAR and Pixel Offset Trackin](https://youtu.be/cGElHzEajiY?si=oDVs7HEAgvHeZGGW), [Monitoring Fast-Moving Landslide in the Three Gorges Area By Offset Tracking Method](https://youtu.be/KVQFbiyoIn8?si=yGxPF2LHeppSEosS), [ID 394 Different Scales of Surface Deformation Identified by Offset Tracking and InSAR at Bezymianny](https://youtu.be/IHTDVyUlDi8?si=j2wa09CtIV2v-iv4), [2024 InSAR Processing and Analysis (ISCE+)](https://www.youtube.com/playlist?list=PLGQwSTwiUcKxnQYpD4Y2ItViU-BBlS5Ti).