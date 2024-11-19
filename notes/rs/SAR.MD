# Synthetic Aperture Radar (SAR)

<p style="text-align:justify;">
Synthetic Aperture Radar (SAR) is an advanced remote sensing technology that uses radar systems to create high-resolution images of the Earth's surface. Unlike optical sensors, SAR operates by transmitting microwave signals and measuring the reflected signals from the Earth's surface. This technology is widely used for Earth observation, military surveillance, environmental monitoring, and more due to its unique capabilities.
</p>

<img src="./img/sat.jpg" width=100%>

Synthetic Aperture Radar (SAR)
Synthetic Aperture Radar (SAR) is an advanced remote sensing technology that uses radar systems to create high-resolution images of the Earth's surface. Unlike optical sensors, SAR operates by transmitting microwave signals and measuring the reflected signals from the Earth's surface. This technology is widely used for Earth observation, military surveillance, environmental monitoring, and more due to its unique capabilities.

## Principle of SAR

SAR works on the principle of active remote sensing:

+ Active Illumination:

    + The radar system transmits microwave pulses towards the ground. These pulses interact with the surface and are scattered back to the sensor.

+ Echo Detection:

    + The radar detects the reflected signals (echoes), which vary based on the surface's geometry, texture, and dielectric properties.

+ Synthetic Aperture:

    + SAR uses the motion of the radar platform (e.g., satellite, aircraft) to simulate a very large antenna or aperture. By combining the signals received at different positions along the flight path, SAR achieves high spatial resolution that would otherwise require an impractically large physical antenna.

<img src="./img/sar.png" width=100%>

## SAR Data Characteristics
+ `Polarization`: SAR systems can transmit and receive in different polarizations (horizontal or vertical).

+ Common modes include:
    - Single polarization (HH, VV)
    - Dual polarization (HH+VV, HV+VH)
    - Full polarization (HH, HV, VH, VV)

+ `Frequency Bands`: SAR operates in specific microwave frequency bands, each with distinct advantages:

    | **Band**   | **Frequency Range**   | **Applications**                        |
    |------------|-----------------------|-----------------------------------------|
    | X-band     | 8-12 GHz              | High-resolution mapping, urban studies |
    | C-band     | 4-8 GHz               | Vegetation monitoring, agriculture      |
    | L-band     | 1-2 GHz               | Forest biomass, subsurface penetration |
    | P-band     | 0.3-1 GHz             | Deep subsurface studies                |


+ `Spatial Resolution`: Depends on system design and operating frequency. SAR can achieve sub-meter resolution for high-precision applications.

+ `Interferometry (InSAR)`: By comparing phase differences between multiple SAR images, InSAR can measure surface deformations, subsidence, and terrain elevation.

## SAR Processing Techniques :
1. `Image Formation` : Raw SAR data must be processed to form interpretable images using algorithms like:
    + Range-Doppler algorithm
    + Chirp scaling algorithm
    + Back-projection method
2. `Speckle Reduction` : SAR images often contain speckle noise due to coherent imaging. Filtering techniques like Lee, Frost, or Gamma MAP filters are used.
3. `Polarimetric Analysis` : Explores the polarization characteristics of SAR data to distinguish surface features.
4. `SAR Interferometry (InSAR)` : Uses phase differences between two SAR images to measure surface deformation or topography.
5. `Differential SAR Interferometry (DInSAR)` : Enhances InSAR by isolating deformation signals for applications like subsidence monitoring.
6. `Persistent Scatterer Interferometry (PSI)` : Analyzes long-term deformation trends using stable reflectors (persistent scatterers).
7. `Time-Series Analysis` : Monitors temporal changes in SAR data to study dynamic processes like vegetation growth or ice flow.

### SAR Satellites and Missions
1. Active SAR Missions
+ Sentinel-1 (ESA):
C-band SAR for environmental monitoring and disaster response.
+ RADARSAT-2 (Canada):
Multi-mode imaging for agriculture, forestry, and defense.
+ TerraSAR-X (Germany):
High-resolution X-band SAR for urban and infrastructure studies.

2. Upcoming and Planned Missions
+ NISAR (NASA-ISRO SAR Mission):
Dual-band (L and S-band) SAR for global Earth observation.
+ BIOMASS (ESA):
P-band SAR for forest biomass and carbon stock estimation.

3. Historic Missions
+ ERS-1/2 (ESA):
Pioneered SAR applications in Earth observation.
+ ALOS PALSAR (Japan):
L-band SAR for forestry and disaster monitoring.


## Tools for SAR Data Processing
+ Open-Source Tools:

    + SNAP (Sentinel Application Platform):
    Free software by ESA for Sentinel-1 SAR data processing.
    + ISCE (InSAR Scientific Computing Environment):
    For SAR and InSAR workflows.

+ Commercial Software:

    + SARscape (ENVI):
    Advanced SAR processing for interferometry  and polarimetry.
    + ERDAS IMAGINE:
    SAR data analysis and interpretation.
+ Python Libraries:

    + PySAR:
    For time-series InSAR analysis.
    + GDAL and Rasterio:
    For SAR data manipulation.

## SAR Data Processing Workflow :

```
Raw Data Acquisition → Radiometric Calibration → Geometric Correction → SAR Image Formation → Speckle Reduction → (Optional: Polarimetric/InSAR/DInSAR Processing) 
→ Image Interpretation → Data Integration and Visualization
```
