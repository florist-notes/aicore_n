# GNSS Point-to-Surface Data Fusion

Global Navigation Satellite System (GNSS) point-to-surface data fusion refers to the integration of GNSS point data with surface models or measurements to enhance the accuracy, reliability, and usability of positioning and mapping information. This fusion technique is widely used in various applications, including surveying, geodesy, autonomous navigation, and environmental monitoring.

<img src="./img/ins.png" width=100%>

## Key Concepts
+ `GNSS Point Data`:

    + GNSS provides precise geospatial positions in the form of point coordinates (latitude, longitude, and elevation).
    + Accuracy depends on satellite geometry, signal quality, and environmental factors such as multipath effects or obstructions.
    + GNSS points represent discrete locations without direct reference to terrain or surface features.

+ `Surface Models`:

    + Surface models represent the physical surface of the Earth or other environments in 2D or 3D forms.
    + Types include:
        + Digital Elevation Models (DEM): Elevation data at regular intervals.
        + Digital Surface Models (DSM): Include natural and man-made features.
        + Digital Terrain Models (DTM): Focus on bare Earth topography.
    + These models are derived from sources like LiDAR, satellite imagery, or photogrammetry.

+ `Data Fusion`:

    + Combines GNSS point data with surface information to achieve a spatially coherent representation.
    + Requires interpolation, transformation, and adjustment techniques to align disparate datasets.


## Steps in GNSS Point-to-Surface Data Fusion
+ `Data Acquisition`:

    + GNSS Data: Collect GNSS point measurements using receivers.
    + Surface Data: Obtain DEM, DSM, or DTM from remote sensing or surveying.

+ `Preprocessing`:

    + Coordinate transformation to ensure both datasets are in the same reference system (e.g., WGS84, UTM).
    + Error correction for GNSS data (e.g., using RTK or PPP techniques) and surface data (e.g., filtering LiDAR noise).

+ `Alignment`:

    + Ensure the GNSS points and surface model are geographically and geometrically aligned.

+ `Techniques include`:
    + Transformation to match projection systems.
    + Geoid modeling to convert GNSS ellipsoidal heights to orthometric heights (referenced to mean sea level).

+ `Data Interpolation`:

    + Interpolate the surface model to estimate surface parameters (elevation, slope) at the GNSS point locations.
    + Common interpolation methods: 
        + Nearest neighbor
        + Bilinear
        + Bicubic
        + Kriging
+ `Fusion Algorithms`:

    + Weighted Averaging: Combine GNSS elevation with surface model elevation using weights based on accuracy.
    + Kalman Filtering: Dynamically integrate GNSS data with surface model updates in real-time applications.
    + Machine Learning Models: Use neural networks or regression techniques to predict surface characteristics from GNSS points.

+ `Error Modeling and Correction`:

    + Analyze residuals (differences between GNSS points and interpolated surface values).
    + Apply corrections based on identified biases or systematic errors.

+ `Integration and Analysis`:

    + Combine point and surface data into a unified model.
    + Analyze for applications such as: 
        + Terrain profiling
        + Surface deformation monitoring
        + Path planning for autonomous systems

<p style="text-align:justify;">
GNSS point-to-surface data fusion is a sophisticated technique used to enhance spatial data analysis and interpretation by integrating GNSS signals with surface data like satellite imagery or ground-based measurements. This fusion often involves methodologies such as Kalman filtering, statistical interpolation, or machine learning to improve accuracy, especially in applications like environmental monitoring, navigation, and resource management.
</p>




Resources :

+ [A New Data Fusion Method for GNSS/INS Integration Based on Weighted Multiple Criteria](https://www.mdpi.com/2072-4292/16/17/3275)
+ [Inversion of Regional Groundwater Storage Changes Based on the Fusion of GNSS and GRACE Data: A Case Study of Shaanxi–Gansu–Ningxia](https://www.mdpi.com/2072-4292/15/2/520)
+ [InSAR and GNSS data fusion for improved urban heat island estimation using local climate zone classification](https://www.sciencedirect.com/science/article/pii/S1569843224002607)
+ [GNSS-IR Soil Moisture Retrieval Using Multi-Satellite Data Fusion Based on Random Forest](https://www.mdpi.com/2072-4292/16/18/3428)
+ [Inversion of surface vegetation water content based on GNSS-IR and MODIS data fusion](https://satellite-navigation.springeropen.com/articles/10.1186/s43020-020-00021-z)
+ [GNSS-INSAR Fusion Method for High Precision Monitoring of Surface Deformation](https://www.geores.com.cn/ygjsyyy/EN/10.11873/j.issn.1004-0323.2023.5.1028)
+ [Improving soil moisture retrieval from GNSS-interferometric reflectometry: parameters optimization and data fusion via neural network](https://www.tandfonline.com/doi/full/10.1080/01431161.2021.1988186)
+ [Combination on Gnss Precise Point Positioning and Insar Data Fusion to Monitor the Land Subsidence in Taiwan](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4875530)
+ [GNSS-Imaging Data Fusion for Integrity Enhancement in Autonomous Vehicles](https://ieeexplore.ieee.org/document/9754275?denied=)
+ [Water vapor mapping by fusing InSAR and GNSS remote sensing data and atmospheric simulations](https://hess.copernicus.org/articles/19/4747/2015/hess-19-4747-2015.pdf)