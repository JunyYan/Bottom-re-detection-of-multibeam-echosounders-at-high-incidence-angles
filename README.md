# Bottom-re-detection-of-multibeam-echosounders-at-high-incidence-angles

![Result](Result.png)
![Result2](Result2.png)

## What is this repository for?

Data and codes for the manuscript "Bottom re-detection of multibeam echosounders at high incidence angles based on water column data and self-attention network"

## Main Codes and Files

MBESSample60_.zip: samples for model training and validation.
- source: The extracted echo strength sequece with varouise length
- label: The manual bottom detection postion in the corresponding source sequece

EMAllParser.py：codes for decoding the raw multibeam binary file.

WCP.py: codes for generating along-track multibeam images and one beam echo sequece.

models.py: codes for the propsed model.

tools.py: codes for model comparison.

overall_flow.ipynb: jupyter notebook which shows the overall flow of the proposed method.

## Multibeam files 

Due to the limitation of upload file size, only 2 small raw multibeam files were uploaded with the codes. More multibeam files can be accessed at the [NOAA website](https://www.ncei.noaa.gov/maps/water-column-sonar/). 

## Usage


## Who do I talk to?
Jun Yan, Anhui University 
jun.yan@ahu.edu.cn

## License
Copyright (C) 2026 Jun Yan. This program is free software.
