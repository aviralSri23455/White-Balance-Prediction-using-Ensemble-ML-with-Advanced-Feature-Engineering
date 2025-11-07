# White Balance Prediction using Ensemble ML with Advanced Feature Engineering

ML solution for White Balance Prediction Challenge. Predicts Temperature & Tint adjustments using ensemble of gradient boosting models (LightGBM, XGBoost, CatBoost) + neural networks with 100+ engineered features from image analysis and EXIF metadata.

## Problem Overview

White Balance (Temperature and Tint) sliders are used by a lot of people to adjust the colours and lighting of their images as per the situation accordingly. Thus, it becomes extremely important to accurately and consistently edit the user images, as per their style reflected in their training data.

* Every image has an "As Shot" White Balance, which differs between different camera brands. This As Shot WB helps the model to get a reference point for the image's current WB while inference, to determine what direction it needs to move and by what magnitude.

* The total range of Temperature is 2000 to 50000K and for Tint, it is -150 to +150. Now, the sensitivity of the Temperature slider decreases non-linearly as it increases. i.e., a delta of 500 from 2K to 2.5K will be much more visible in the image as compared to a delta of 500 from 5K to 5.5K. This is something that the model needs to understand as well, so that it produces a smaller error delta on the leftmost Temperature ranges. Tint, however, scales linearly.

* The inclusion of an extra light source in the image can cause the As Shot WB of the image to be different than the one which doesn't have that light source and thus, the two images would need to have different WB edits applied on them to look the same at the end.

Now, one of the major problems at hand is with **Consistency**. For e.g., The inclusion of an extra light source, some different colored subject, zoom-in/out can cause high level changes in the image's pixel representation and cause the model to give inconsistent results in terms of the final output, even though the images look similar and should receive same/similar edits applied to them, in terms of WB.

##  Description

Develop a machine learning model that, given an image and its corresponding input features, accurately predicts Temperature and Tint values.

The model can use image-only features, metadata-only features, or a combination of both.

Any regression, classification, or sequential approaches can be used based on the data along with the loss function and metric of choice.

The quality of the solution is based on **accuracy, consistency, speed, and size** (in that order).



### Contents:

**Train/**
- `images/` — Contains 2,539 TIFF images, each resized to 256×256 pixels. These images serve as the primary visual input.
- `sliders.csv` — Contains both input features and output labels for each image.
  - `id_global` — Unique identifier for each image (matches the image filename).
  - Input features — Various numerical and categorical attributes used as model inputs.
  - Output labels:
    * Temperature
    * Tint

**Validation/**
- `images/` — Contains 493 TIFF images (256×256) for which predictions are generated.
- `sliders_inputs.csv` — Contains all the same input columns as training/sliders.csv, except the output labels.


## Variable Description

The columns provided in the dataset are as follows:

| Column Name        | Description                                                    |
|--------------------|----------------------------------------------------------------|
| id_global          | Represents a unique image id (maps to tiff file name)          |
| As Shot WB         | "currTemp", "currTint"                                         |
| EXIF               | "aperture", "flashFired", "focalLength", "isoSpeedRating", "shutterSpeed" |
| Camera Info        | "camera_model", "camera_group"                                 |
| Extra Properties   | "intensity", "ev"                                              |
| Target Variables   | "Temperature", "Tint"                                          |

## Evaluation Metric

The metric used is **Mean Absolute Error (MAE)**

### Formula:

```
mae_temperature = 1 / (1 + metrics.mean_absolute_error(actual['Temperature'], predicted['Temperature']))
mae_tint = 1 / (1 + metrics.mean_absolute_error(actual['Tint'], predicted['Tint']))
```

