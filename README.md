# Dataset Overview

This project uses the **Xeek Well 15/9-15** well log dataset, which contains geological and petrophysical measurements collected from a single oil and gas well. Each record represents measurements taken at a specific depth, including borehole characteristics, resistivity, density, gamma ray, neutron porosity, photoelectric factor, and sonic travel time, along with geological group and formation information.

## Features

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

The objective of this project is to develop an **anomaly detection model** that identifies unusual or abnormal well log measurements. Detecting anomalies can help identify sensor errors, logging issues, unexpected geological formations, or other irregular subsurface conditions. The workflow includes data preprocessing, exploratory data analysis (EDA), feature engineering, model training, and evaluation using unsupervised anomaly detection techniques.
