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

<table border="1" style="border-collapse: collapse; width: 100%;">
    <thead>
        <tr>
            <th style="text-align: left; padding: 8px; background-color: #000;">Step</th>
            <th style="text-align: left; padding: 8px; background-color: #000;">Description</th>
            <th style="text-align: left; padding: 8px; background-color: #000;">Challenges</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td style="padding: 8px;">1. Raw Data Acquisition</td>
            <td style="padding: 8px;">SAR systems collect raw electromagnetic signals in slant range geometry. These signals contain phase and amplitude information.</td>
            <td style="padding: 8px;">High data volumes; noisy raw data requiring extensive processing.</td>
        </tr>
        <tr>
            <td style="padding: 8px;">2. Radiometric Calibration</td>
            <td style="padding: 8px;">Corrects system distortions and converts signal intensity into radar backscatter coefficients (σ⁰).</td>
            <td style="padding: 8px;">Addressing antenna pattern and gain variations accurately.</td>
        </tr>
        <tr>
            <td style="padding: 8px;">3. Geometric Correction</td>
            <td style="padding: 8px;">Aligns SAR images to geographic coordinates via geocoding, orthorectification, and slant-to-ground range conversion.</td>
            <td style="padding: 8px;">Terrain-induced distortions and precise DEM integration.</td>
        </tr>
        <tr>
            <td style="padding: 8px;">4. SAR Image Formation</td>
            <td style="padding: 8px;">Converts raw SAR signals into interpretable 2D images using range and azimuth compression techniques.</td>
            <td style="padding: 8px;">Complexity of focusing algorithms like Range-Doppler or Chirp Scaling.</td>
        </tr>
        <tr>
            <td style="padding: 8px;">5. Speckle Noise Reduction</td>
            <td style="padding: 8px;">Reduces speckle noise using filters (e.g., Lee, Frost) or multi-look processing for smoother images.</td>
            <td style="padding: 8px;">Balancing noise reduction and edge preservation.</td>
        </tr>
        <tr>
            <td style="padding: 8px;">6. Polarimetric Processing (Optional)</td>
            <td style="padding: 8px;">Analyzes polarized backscatter data for surface classification and scattering mechanism analysis.</td>
            <td style="padding: 8px;">Handling high data dimensionality and decomposition complexities.</td>
        </tr>
        <tr>
            <td style="padding: 8px;">7. Interferometric SAR (InSAR) Processing (Optional)</td>
            <td style="padding: 8px;">Generates interferograms to measure elevation or displacement by computing phase differences between SAR images.</td>
            <td style="padding: 8px;">Coregistration accuracy and phase unwrapping challenges.</td>
        </tr>
        <tr>
            <td style="padding: 8px;">8. Differential InSAR (DInSAR) Processing (Optional)</td>
            <td style="padding: 8px;">Detects temporal surface changes by subtracting static topography from interferograms.</td>
            <td style="padding: 8px;">Minimizing noise due to atmospheric effects or decorrelation.</td>
        </tr>
        <tr>
            <td style="padding: 8px;">9. Time-Series Analysis (Optional)</td>
            <td style="padding: 8px;">Analyzes SAR data over time for dynamic changes, using techniques like PSI and SBAS.</td>
            <td style="padding: 8px;">Consistency in identifying stable reflectors or minimizing baseline errors.</td>
        </tr>
        <tr>
            <td style="padding: 8px;">10. Image Interpretation and Classification</td>
            <td style="padding: 8px;">Extracts meaningful information using thresholding, machine learning, or texture analysis.</td>
            <td style="padding: 8px;">Accurate feature extraction amidst noise and speckle artifacts.</td>
        </tr>
        <tr>
            <td style="padding: 8px;">11. Data Integration and Visualization</td>
            <td style="padding: 8px;">Combines SAR data with other datasets for comprehensive analysis, using GIS tools.</td>
            <td style="padding: 8px;">Harmonizing diverse datasets and ensuring spatial consistency.</td>
        </tr>
    </tbody>
</table>
