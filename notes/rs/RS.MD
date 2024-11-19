# Remote Sensing 🛰️

<p style="text-align:justify;">
Remote sensing refers to the process of acquiring information about an object or phenomenon without making physical contact with it. This is typically achieved by detecting and analyzing reflected or emitted electromagnetic radiation from the Earth's surface and atmosphere. It plays a crucial role in various fields, including environmental monitoring, urban planning, agriculture, forestry, disaster management, and defense.
</p>

## How Remote Sensing Works

+ `Energy Source or Illumination` : A source (usually the Sun or an artificial emitter) provides electromagnetic radiation. Passive sensors rely on natural energy, while active sensors generate their own signal (e.g., radar).

+ `Propagation of Energy` : The energy travels through the atmosphere, interacts with the Earth's surface, and is reflected or emitted back.

+ `Interaction with Objects` : Different surfaces (water, vegetation, soil) interact differently with radiation, depending on their physical and chemical properties.

+ `Sensor Reception` : Sensors aboard satellites, aircraft, or drones capture the reflected/emitted radiation and convert it into data.

+ `Data Processing` : The raw data undergo preprocessing (removing noise, correcting distortions) and analysis to extract meaningful information.

<p style="text-align:justify;">
Remote sensing employs a variety of sensors to capture data from the Earth's surface and atmosphere. These sensors differ based on the energy source, platform, wavelength sensitivity, and application. Below is a detailed explanation of the different types of sensors used in remote sensing.
</p>

### Classification of Sensors :

A. Based on Energy Source
+ `Passive Sensors`:
These sensors rely on natural energy sources, primarily sunlight. They measure the energy reflected or emitted by objects on the Earth's surface.
    + `Examples`: Optical cameras, multispectral scanners, thermal sensors.

+ `Active Sensors`:
These sensors emit their own energy (typically in the form of electromagnetic waves) and measure the reflected signal from the target.
    + `Examples`: Radar (Radio Detection and Ranging), LiDAR (Light Detection and Ranging).

B. Based on Platform
+ `Spaceborne Sensors`: Mounted on satellites, they provide global and large-area coverage.
    + `Examples`: Landsat, Sentinel, MODIS.

+ `Airborne Sensors`: Mounted on aircraft or drones for high-resolution, localized imaging.
    + `Examples`: Hyperspectral imagers on drones, LiDAR on airplanes.

+ `Ground-Based Sensors`: Deployed on the Earth's surface for localized and high-precision data collection.
    + `Examples`: Spectroradiometers, soil moisture probes.

C. Based on Wavelength Sensitivity
+ `Optical Sensors`: Operate in the visible, near-infrared (NIR), and short-wave infrared (SWIR) regions.
+ `Thermal Infrared Sensors`: Measure emitted thermal radiation, providing data on temperature.
+ `Microwave Sensors`: Operate in the microwave region and can penetrate clouds, vegetation, and even soil.

### Types of Sensors :


<table style="width:100%" >

<tr>
<th>Sensors</th>
<th>Description <br /></th>
<th>Reference <br /></th>
</tr>

<tr>
<th>Optical Sensors</th>
<td>

These sensors use visible and near-visible wavelengths to capture images of the Earth's surface.

+ Multispectral Sensors:
Capture data across a few spectral bands, typically 3 to 10.
    + Examples:

        - Landsat TM (Thematic Mapper) and OLI (Operational Land Imager)
        - Sentinel-2 MSI (Multispectral Instrument)
+ Hyperspectral Sensors:
Capture data across hundreds of continuous spectral bands, enabling detailed material analysis.
    + Examples:

        - Hyperion (onboard NASA's EO-1 satellite)
        - AVIRIS (Airborne Visible/Infrared Imaging Spectrometer)
    + Applications:

        - Vegetation health monitoring (e.g., NDVI).
        - Water quality analysis.
        - Mineral exploration.
<br /></td>
<td>- <br /></td>
</tr>

<tr>
<th>Thermal Infrared Sensors</th>
<td>

These sensors detect the Earth's emitted thermal radiation, which is related to surface temperature.

+ Examples:

    + MODIS (Moderate Resolution Imaging Spectroradiometer)
    + ASTER (Advanced Spaceborne Thermal Emission and Reflection Radiometer)

+ Applications:

    + Urban heat island studies.
    + Volcano and forest fire monitoring.
    + Soil moisture estimation.
 <br /></td>
<td>$ bc <br /></td>
</tr>

<tr>
<th>Microwave Sensors</th>
<td>

Operate in the microwave region (1mm to 1m wavelength) and are less affected by weather conditions like clouds and rain.

+ Active Microwave Sensors (Radar):
Emit microwave signals and measure the return after interacting with the surface.
    + Examples:

        + Synthetic Aperture Radar (SAR) on Sentinel-1
        + RADARSAT

+ Passive Microwave Sensors:
Measure natural microwave emissions from the Earth's surface.
    + Examples:

        + SMOS (Soil Moisture and Ocean Salinity)
        + AMSR-E (Advanced Microwave Scanning Radiometer)
    + Applications:

    + Soil moisture and snowpack measurement.
    + Monitoring deforestation.
    + Ocean wind speed and wave height estimation.
</td>
<td>Logd</td>

</tr>

<tr>
<th>LiDAR (Light Detection and Ranging)</th>
<td>LiDAR sensors emit laser pulses and measure the time taken for the reflected signals to return. This provides high-resolution 3D data.

    + Examples:

        + ICESat (Ice, Cloud, and Land Elevation Satellite)
        + Airborne LiDAR systems.
    + Applications:

        + Terrain mapping and digital elevation models (DEMs).
        + Forest canopy height measurement.
        + Urban infrastructure mapping.</td>
<td>Logd</td>

</tr>

</table>

<table style="width:100%" >

<tr>
<th>Radiometers </th>
<th>Spectroradiometers <br /></th>
<th>Scatterometers <br /></th>
<th>Imaging Spectrometers <br /></th>
<th>Gravimeters <br /></th>
</tr>

<tr>
<td>

Measure the intensity of electromagnetic radiation across various wavelengths.

+ Examples:

    + AVHRR (Advanced Very High-Resolution Radiometer)
    + MODIS
+ Applications:

    + Monitoring sea surface temperatures.
    + Analyzing atmospheric and surface radiation.
</td>

<td>

Measure reflected or emitted energy across specific spectral ranges with high accuracy.

+ Examples:

    + MERIS (Medium Resolution Imaging Spectrometer)
    + SeaWiFS (Sea-viewing Wide Field-of-view Sensor)
+ Applications:

    + Monitoring chlorophyll concentration in oceans.
    + Atmospheric studies (aerosols, gases).
</td>

<td>
Active microwave sensors that measure the scattering of microwave energy. They are particularly useful for ocean and wind studies.

+ Examples:

    + QuikSCAT (Quick Scatterometer)
    + ASCAT (Advanced Scatterometer)

+ Applications:

    + Measuring ocean surface wind speeds.
    + Sea ice monitoring.

</td>


<td>
These combine the features of cameras and spectrometers, capturing both spatial and spectral information simultaneously.

+ Examples:

    + HyTES (Hyperspectral Thermal 
    + Emission Spectrometer)

+ Applications:

    + Mineral mapping.
    + Identifying crop stress.

</td>


<td>
Measure variations in the Earth's gravitational field to understand subsurface features.

+ Examples:

    + GRACE (Gravity Recovery and Climate Experiment)

+ Applications:

    + Groundwater studies.
    + Ice mass loss in polar regions.

</td>

</tr>

</table>