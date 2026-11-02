## Dataset Overview

This project uses the **Xeek Well 15/9-15** well log dataset, which contains geological and petrophysical measurements collected from a single oil and gas well. Each record represents measurements taken at a specific depth, including borehole characteristics, resistivity, density, gamma ray, neutron porosity, photoelectric factor, and sonic travel time, along with geological group and formation information.

### Features

- **WELL** – Well identifier
- **DEPTH_MD** – Measured depth
- **GROUP** – Geological group
- **FORMATION** – Geological formation
- **CALI** – Borehole diameter (Caliper)
- **RDEP** – Deep resistivity
- **RHOB** – Bulk density
- **GR** – Gamma ray
- **NPHI** – Neutron porosity
- **PEF** – Photoelectric factor
- **DTC** – Sonic travel time
- **LITH** – Lithology class (target/label)

## Project Goal

The objective of this project is to develop an **unsupervised anomaly detection model** that identifies unusual or abnormal well log measurements. Detecting anomalies can help identify sensor errors, logging issues, unexpected geological formations, and other irregular subsurface conditions.

The project uses **Isolation Forest** as the primary anomaly detection algorithm. Since the objective is unsupervised anomaly detection, the model learns patterns and distributions from well log measurements without requiring labeled anomaly examples.

## Training and Deployment Strategy

A key aspect of this project is that the available dataset contains measurements from only one well. Therefore, the entire dataset from the **Xeek Well 15/9-15** well is used to train the anomaly detection model rather than splitting it into training and testing subsets.

The purpose of this approach is to allow the model to learn the statistical characteristics and patterns of well log measurements from the available reference well.

Once trained, the model is intended to be deployed on data collected from **new, previously unseen wells** to identify measurements that deviate from the patterns learned during training.

### Workflow

1. **Reference Well Data Collection:** Use the complete Xeek Well 15/9-15 dataset as the reference training dataset.
2. **Data Preprocessing:** Handle missing values, select relevant features, and apply the necessary transformations.
3. **Exploratory Data Analysis:** Investigate feature distributions, correlations, and relationships between well log measurements.
4. **Model Training:** Train an Isolation Forest model using the complete reference well dataset.
5. **Model Persistence:** Save the trained model and preprocessing components for future inference.
6. **New Well Data Inference:** Load the saved model and preprocessing components when new well log data becomes available.
7. **Anomaly Detection:** Apply the trained model to the new well measurements to identify observations that significantly deviate from the learned patterns.

### Model Deployment Concept

The intended deployment workflow is illustrated below:

```text
        Reference Well
       (Xeek Well 15/9-15)
                |
                v
       Data Preprocessing
                |
                v
        Isolation Forest
           Training
                |
                v
       Trained Model
        + Imputer
        + Scaler
                |
                v
       Model Persistence
                |
                |
       ---------------------
                |
                v
        New Well Data
                |
                v
       Same Preprocessing
                |
                v
       Trained Isolation
            Forest
                |
                v
       Anomaly Predictions
                |
          -------------
          |           |
          v           v
        Normal     Anomalous
      Measurements Measurements
```

**Important:** The model is trained once using the reference well and subsequently used to evaluate incoming measurements from new wells. The model is not retrained on each new well during inference.

## Evaluation Strategy

Since the available reference dataset does not contain ground-truth anomaly labels, conventional supervised evaluation metrics such as accuracy, precision, recall, and F1-score cannot be directly calculated.

Instead, the model is evaluated using unsupervised and domain-oriented approaches, including:

- **Anomaly Score Analysis:** Examine the distribution of anomaly scores to understand how the model distinguishes unusual measurements.
- **Anomaly Detection Rate:** Monitor the proportion of observations classified as anomalous.
- **Feature Distribution Analysis:** Investigate the characteristics of detected anomalies and compare them with normal measurements.
- **Depth-Based Visualization:** Visualize detected anomalies against measured depth to identify unusual measurements and anomalous intervals.
- **Domain-Based Validation:** Investigate whether detected anomalies correspond to potential logging problems, measurement errors, or unusual geological conditions.

Because the model will be applied to different wells, its ability to generalize across wells is an important consideration. Geological characteristics, logging tools, and measurement distributions can vary between wells. Therefore, predictions on new wells should be interpreted carefully and validated against domain knowledge whenever possible.
