# Binary Snow Stability Classification with Terrain Metrics

## Research Objective
The goal of this project is **to develop a new binary snow stability classification scheme that incorporates terrain-based metrics alongside traditional stability tests**.
The new scheme is evaluated for its effectiveness in reducing false-stable assessments compared to previous stability-test-only schemes.

## Overview
This repository contains code and data processing workflows for:
- Integrating field snow pit data with the gridded snow depth dataset.
- Converting Swiss geographic coordinates from LV03 to LV95 and interpolating snow depth values at pit locations.
- Calculating local snow depth variance (and CV) using a 3×3 grid window around each pit.
- Merging meteorological data (daily mean wind speed) with pit observations.
- Linking stability classifications from the original schemes and avalanche activity records for analysis.
- Preparing feature sets for testing the proposed binary stability classification scheme.

## Data Processing Workflow
1. Load and Normalize Field Data
   Extract observation dates, convert coordinates from LV03 to LV95, and prepare a working DataFrame.
   
2. Snow Depth Extraction
   Retrieve daily snow depth values from SPASS at pit coordinates, using bilinear interpolation to account for coarse grid spacing.

3. Variance Calculation
   Compute the variance of snow depth (and CV) in a 3×3 grid window around each pit, also via linear interpolation, to measure local snowpack variability.

4. Wind Data Integration
   Match each pit observation date to its corresponding daily mean wind speed from WFJ station data.

5. Stability and Avalanche Labels
   Attach traditional stability classifications and avalanche activity data to each pit observation.

## Why Linear Interpolation?
SPASS data is relatively coarsely gridded, so direct nearest-neighbor sampling can miss sub-grid variability. Linear interpolation provides a smoother and more accurate estimate by blending values from surrounding cells in proportion to distance. This is applied both to depth and to variance calculations.

## Datasets:
- Stability test dataset: 589 snowpit samples from the eastern Swiss Alps, Davos region, from winters of 2002-2019.

   https://opendata.swiss/en/dataset/field-observations-of-snow-instabilities
- Snow depth dataset (SPASS): long-term daily 1 km gridded data of snow depth for Switzerland spanning 1962-2023.

   https://envidat.ch/#/metadata/spass---new-gridded-snow-datasets-for-switzerland
- Wind dataset: daily average wind speeds taken from MeteoSwiss (Federal Office for Meteorology and Climatology), specifically the WFJ station.

   https://www.meteoswiss.admin.ch/services-and-publications/service/open-data.html

## Key Libraries Used
- numpy: Efficient numeric operations.
- pandas: Data cleaning, time series handling, and merging multiple datasets.
- xarray: Accessing and interpolating snow depth data from SPASS.
- matplotlib: Creating visualizations of snow depth, as well as bar charts for final results.
