# Bottom-re-detection-of-multibeam-echosounders-at-high-incidence-angles

Data and codes for the manuscript "Bottom re-detection of multibeam echosounders at high incidence angles based on water column data and self-attention network"

![Result](Result.png)
![Result2](Result2.png)
 

## Main Codes and Files

MBESSample60_.zip: samples for model training and validation.
- source: The extracted echo strength sequece with varouise length
- label: The manual bottom detection postion in the corresponding source sequece

EMAllParser.py：codes for decoding the raw multibeam binary file.

WCP.py: codes for generating along-track multibeam images and one beam echo sequece.

models.py: codes for the model.

tools.py: codes for model comparison.

# Multibeam files 

Due to the limitation of upload file size, only 2 small raw multibeam files were uploaded with the code.  More multibeam files can be accessed at the [NOAA website](https://www.ncei.noaa.gov/maps/water-column-sonar/). 
