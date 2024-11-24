# Polarization Processing and other functions

<p style="text-align:justify;">
Polarization processing in Interferometric Synthetic Aperture Radar (InSAR) is a sophisticated method to enhance data analysis by utilizing the polarization properties of radar signals. By analyzing the polarization, scientists can extract valuable information about the Earth's surface, such as vegetation, soil properties, water bodies, and man-made structures.
</p>

<img src="./img/polsar.png" width=100%>

Here’s a detailed breakdown of polarization processing in InSAR:

1. Basics of Polarization in SAR
Polarization refers to the orientation of the electromagnetic wave's electric field. In SAR, the radar signal can be transmitted and received in different polarizations:

+ `Horizontal (H)`: Electric field oscillates horizontally.
+ `Vertical (V)`: Electric field oscillates vertically.

    SAR systems can operate in one or more of the following modes:

    + `Single-polarization` (e.g., HH or VV): Same polarization for transmission and reception.
    + `Dual-polarization` (e.g., HH+HV or VV+VH): Two combinations of polarizations.
    + `Quad-polarization` (e.g., HH, HV, VH, VV): All four combinations of transmit and receive polarizations.


2. Role of Polarization in InSAR
    + InSAR uses the phase difference between two SAR images acquired at different times or locations to measure surface deformation, topography, or structure. Adding polarization information enables:

    + Enhanced Target Characterization: Different materials (vegetation, soil, water) interact with radar waves differently based on polarization.
    + Discrimination of Scattering Mechanisms: Polarization helps differentiate between surface, double-bounce, and volume scattering.
    + Improved Coherence: Selecting optimal polarizations can improve the coherence of the interferometric signal, enhancing deformation or topographic measurements.

3. Polarimetric InSAR (PolInSAR)
+ PolInSAR combines polarimetry and InSAR to exploit the full polarization information in interferometric processing. It uses polarimetric data from multiple polarizations to improve the understanding of scattering mechanisms and to enhance the accuracy of measurements. The key steps in PolInSAR are:

    + a. Acquisition of Polarimetric Data
        + Acquired using a quad-pol radar system.
        + Captures four polarization combinations (HH, HV, VH, VV).
    + b. Interferometric Processing
        + Generate interferograms for each polarization channel.
        + Compute phase differences to measure surface height or displacement.
    + c. Polarimetric Decomposition
        + Decompose the polarimetric data to understand the scattering properties of the target:
            + Surface scattering: From smooth surfaces like water.
            + Volume scattering: From vegetation or forest canopies.
            + Double-bounce scattering: From structures like buildings.
    + d. Coherence Optimization
        + Compute coherence for each polarization channel and select the channel or combination that maximizes coherence.

5. Tools and Techniques in Polarization Processing
    + a. Cloude-Pottier Decomposition
        + Separates polarimetric data into three components: surface, volume, and double-bounce scattering.
    + b. Coherent Target Decomposition
        + Analyzes polarimetric interferometric coherence to assess surface and volume interactions.
    + c. Polarization Coherence Optimization
        + Finds the optimal linear polarization combination to maximize interferometric coherence.
    + d. Machine Learning
        + Recent advancements include the use of machine learning techniques to classify scattering mechanisms and improve interpretation of polarimetric data.

Performing Polarimetric Interferometric Synthetic Aperture Radar (PolInSAR) involves a complex set of processes combining polarimetric and interferometric SAR data to retrieve detailed information about the Earth's surface, including vegetation structure, topography, and deformation. 

1. PolInSAR Workflow Overview : 

The PolInSAR workflow involves the following steps:

Data Acquisition → 
Data Preprocessing → 
Polarimetric Analysis → 
Interferometric Processing → 
Coherence Optimization → 
Scattering Model Decomposition → 
Height or Biophysical Parameter Retrieval

## Data Acquisition
`PolInSAR requires`:

+ Polarimetric SAR (PolSAR) data: Typically, quad-polarization data (HH, HV, VH, VV).
+ Interferometric SAR data: Acquired as a pair of SAR images (e.g., repeat-pass or bistatic).

`Key Considerations`:
+ Radar Wavelength: Longer wavelengths (e.g., L-band, P-band) penetrate deeper into vegetation and are ideal for forest studies. Shorter wavelengths (e.g., C-band, X-band) are better for surface characterization.
+ Baseline: The distance between the radar antennas; it must be optimized for sensitivity to height variations without excessive decorrelation.
+ Temporal Baseline: Minimize time between acquisitions to reduce decorrelation.

## Data Preprocessing
Preprocessing prepares raw SAR data for PolInSAR analysis.

+ a. `Radiometric Calibration` : 
Correct for system gains and distortions to ensure consistency across polarimetric channels.
Output: Calibrated complex SAR images for each polarization (HH, HV, VH, VV).

+ b. `Co-Registration` : 
Precisely align SAR images from different passes or antennas to the same spatial grid.
Accuracy is critical (sub-pixel level) to preserve interferometric phase information.

+ c. `Filtering` : 
Apply multi-looking or adaptive filters (e.g., Lee or Frost filters) to reduce speckle noise while preserving polarimetric and interferometric information.

## Polarimetric Analysis
Polarimetric analysis helps understand scattering mechanisms in the observed area.

+ a. `Polarimetric Covariance Matrix` : 
Construct the covariance matrix for each pixel.

+ b. `Decomposition Techniques` : 
Cloude-Pottier Decomposition: Separates scattering into surface, volume, and double-bounce components.
Freeman-Durden Decomposition: Focuses on three scattering types (surface, volume, double-bounce) using physical models.

+ c. `Polarimetric Parameters` : 
Calculate parameters such as entropy (H), anisotropy (A), and alpha angle (α) for understanding scattering mechanisms.


## Interferometric Processing
Interferometric processing involves generating and analyzing phase differences.

+ a. `Interferogram Generation` : Combine two co-registered SAR images to compute the interferometric phase.

+ b. `Coherence Estimation` : Compute coherence (𝛾) to assess the quality of the interferometric signal.

+ c. `Coherence Matrix` : Generate a polarimetric coherence matrix for each pixel, combining information from all polarization channels.

## Coherence Optimization
Maximize coherence by finding the optimal polarization combination for height or deformation retrieval.

+ a. Linear Polarization Optimization.
+ b. Eigenvalue Decomposition

Use eigenvalue decomposition of 𝑇 to determine the dominant scattering mechanisms.


## Scattering Model Decomposition
Decompose the observed scattering into physical mechanisms to derive biophysical parameters.

+ a. Volume Scattering Models
Use models (e.g., Random Volume over Ground, RVoG) to separate ground and vegetation contributions.

+ b. Canopy Height Estimation
Estimate vegetation height (ℎ) using interferometric phase and volume decorrelation models:

$$ 
ϕ_{volume}=k_z⋅h⋅sin(θ)
$$

where 
𝑘_𝑧 is the vertical wavenumber, and 𝜃 is the incidence angle.

## Height or Biophysical Parameter Retrieval
Retrieve biophysical parameters (e.g., forest height, biomass) or topographic height using:

+ Phase Unwrapping: Recover the absolute interferometric phase.
+ DEM Generation: Use interferometric phase to generate a Digital Elevation Model (DEM).
+ Vegetation Indices: Derive indices like Leaf Area Index (LAI) or biomass using volume scattering.

## Implementation Tools
Use specialized SAR processing software or libraries:

+ [PolSARpro](https://earth.esa.int/eogateway/tools/polsarpro): Open-source software for polarimetric SAR and PolInSAR processing.
+ GAMMA Software: For advanced interferometric and polarimetric analysis.
+ Python Libraries: Libraries like GDAL, PySAR, and ISCE for custom implementations.

Resources :

+ [Using Dual-Polarization Interferograms to Correct Atmospheric Effects for InSAR Topographic Mapping](https://www.mdpi.com/2072-4292/10/8/1310)
+ [NASA - Polarimetry](https://nisar.jpl.nasa.gov/mission/get-to-know-sar/polarimetry/) | [ESA - PolInSAR and Biomass 23'](https://www.youtube.com/watch?v=5EzDx28ChGM&list=PLvT7fd9OiI9U5WOO_0U3cGTHQ-MVnx9oF) | [Pol-SAR & Pol-InSAR](https://eo.ifu.ethz.ch/forschung/Techniques/Polarimetry.html)
+ [Comparison of Polarization Models in SAR technology for Water Body Extraction](https://kth.diva-portal.org/smash/get/diva2:1901548/FULLTEXT01.pdf)
+ [POLinSAR: Advances in radar remote sensing](https://www.esa.int/Applications/Observing_the_Earth/POLinSAR_Advances_in_radar_remote_sensing)
+ [Polinsar Ground and Volume Separation for Polarimetric Change Analysis in Agricultural Monitoring](https://ieeexplore.ieee.org/document/10641271)
+ [ESA Polarimetry Training 2023 #4 Polarimetric SAR Interferometry with Prof. Matteo Pardini](https://youtu.be/sudxqE1Xof8?si=uXugTAcEevqI0fsY)
+ [ESA's 7th Advanced Training Course on Radar Polarimetry](https://www.youtube.com/playlist?list=PLvT7fd9OiI9U2P1sZTfz2Hhc_H6qIVrXm)
+ [Part 4/4: SAR Polarimetry: Basics and Advanced Concepts - Prof. Eric Pottier (theory)](https://youtu.be/QabUcscQoFY?si=9QRAzFyccCdNVM3q)
+ [Synthetic Aperture Radar: Applications (InSAR, PolSAR, PolInSAR, Multi-temporal, multi-frequency)](https://youtu.be/c67Wt8H5m2Y?si=Qh11LbKKf5nlqEKg)
+ ["Synthetic Aperture Radar Polarimetry Polarimetric Interferometry Remote Sensing" by Dr. Parul Patel](https://youtu.be/iBTipKYJzz8?si=vnXYfEeZDzD4iaSE)
+ [Part 3/4: Spectral analysis techniques for Pol-TomoSAR focusing - Prof. Laurent Ferro-Famil (theory)](https://youtu.be/SCohSRoxrPY?si=yHYh5PVc7sXLQyrr)
+ [NASA ARSET: Introduction to Polarimetric SAR Session 3/4](https://youtu.be/kxmL3JA2eHY?si=uwVAh2IaO-FIIWtd)
+ [Applied Mathematics for SAR and PolSAR Information Extraction (Avik Bhattacharya)](https://youtu.be/ONyNzpRCKNY?si=zLQCEZ__nPUTvOAf)
+ [Part 3/3: Principles and basics of InSAR and Pol-InSAR - Prof. Irena Hajnsek (theory)](https://youtu.be/p1F9qmsLJYg?si=y6tbF8l4wGKdIOP_)
+ [SAR Polarimetry: Polarimetric Model-based Decomposition Theory & POLSAR applications](https://youtu.be/TkqIw0CBq5c?si=ui8seQ49a78SygRG)
+ [Satellite Image Processing with PolSAR Pro](https://www.youtube.com/playlist?list=PL5a4cu_cmV4s0TzPJNQhlfLuwAnn_Oe-w)
+ [IEEE-GRSS INTERNATIONAL SCHOOL on SAR POLARIMETRY](https://sites.google.com/view/polsar-grss-stirling/about)