## Pneumonia Detection Using Chest X-Ray Imaging
This project develops a predictive machine learning model to detect pneumonia by analyzing chest X-ray images. Pretrained deep learning models are leveraged to extract high-level visual features such as lung opacities, consolidation patterns, and abnormal textures that are indicative of pneumonia, allowing efficient learning even with limited medical imaging data. By fine-tuning these pretrained networks on labeled X-ray images, the model aims to accurately distinguish between normal and pneumonia-affected lungs.

## Problem Statement
Pneumonia is a common and potentially life-threatening respiratory infection that significantly impacts global morbidity and mortality, particularly among children, older adults, and immunocompromised individuals. Early and accurate diagnosis is critical for effective treatment and improved patient outcomes. However, conventional diagnostic methods relying on manual interpretation of chest X-ray images can be time-consuming, subjective, and dependent on the availability of experienced radiologists, especially in resource-limited settings. There is a need for an automated, reliable, and data-driven approach that can assist in the detection of pneumonia from chest X-ray images. The objective of this project is to develop a machine learning model using transfer learning to predict pneumonia from X-ray images, supporting early diagnosis, reducing diagnostic burden, and enhancing clinical decision-making in healthcare systems.

## Overview

This project focuses on building a pneumonia classification model using chest X-ray images. **Exploratory data analysis (EDA)** was first conducted to verify the total number of images, analyze the distribution across the two classes **(Pneumonia and Normal)**, and identify **class imbalance** within the dataset.

An initial prediction was performed using a **pre-trained CNN** to assess baseline performance, which demonstrated that the model did not directly generalize well to the target use case. **Transfer learning** was then applied to adapt the pre-trained network to the dataset, and model performance was evaluated using both **accuracy and ROC-AUC**, with ROC-AUC included to account for class imbalance.

**Hyperparameter tuning** was carried out for the **learning rate**, where multiple values were tested. Based on training and validation performance trends, a learning rate of 0.001 was selected as the optimal value.

The final trained model was saved as a .pth file and used for inference. Additionally, the model was exported to **ONNX** format. The ONNX model was packaged with an inference script into a **Docker container** and deployed locally as a **web API** using an AWS Lambda container image.

## Dataset
The data used for training the model is from [kaggle](https://www.kaggle.com/models/huzaifa10/pneumonia-prediction-model-using-vgg16?select=pneumonia_prediction.h5)

With kagglehub installed in your python environment, you can download the dataset by running the code below
```
import kagglehub

# Download latest version
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
print(f"Dataset downloaded to: {path}")
```


