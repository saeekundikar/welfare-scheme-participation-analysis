# Welfare Scheme Participation Analysis: PM-KISAN & Ayushman Bharat

## Overview

This project presents a comprehensive data-driven analysis of welfare scheme participation across Indian districts, focusing on PM-KISAN and Ayushman Bharat. The objective is to identify coverage gaps, understand socioeconomic drivers of enrollment, detect vulnerable regions, and recommend targeted policy interventions.

The analysis combines data cleaning, exploratory analysis, statistical modelling, machine learning segmentation, and geographic intervention mapping to support evidence-based decision making.

---

## Problem Statement

Despite large-scale welfare initiatives, participation levels vary significantly across regions due to differences in literacy, digital accessibility, poverty, rural concentration, and infrastructure availability.

This project aims to answer:

* Which districts have the largest welfare coverage gaps?
* Which socioeconomic factors influence scheme participation?
* Which regions require immediate policy intervention?
* How can districts be segmented for targeted action?

---

## Dataset Information

* Total districts analyzed: **750**
* Number of features: **40+ socioeconomic and welfare indicators**

Key variables include:

* Literacy rate
* Female literacy rate
* Rural population percentage
* BPL concentration
* SC/ST population concentration
* Internet penetration
* Healthcare facility availability
* PM-KISAN eligibility and enrollment
* Ayushman Bharat eligibility and enrollment

---

## Project Workflow

```
Raw Data
   ↓
Data Cleaning & Validation
   ↓
Feature Engineering
   ↓
Exploratory Data Analysis (EDA)
   ↓
Statistical Analysis & Regression
   ↓
District Segmentation using K-Means
   ↓
Intervention Priority Mapping
   ↓
Final Analytical Report
```

---

## Project Structure

```
welfare_scheme_analysis/
│
├── 01_data_download.py          # Dataset creation/loading
├── 02_data_cleaning.py          # Data validation and feature engineering
├── 03_eda.py                    # Exploratory data analysis and visualizations
├── 04_statistical_analysis.py   # Regression, hypothesis testing, anomaly detection
├── 05_segmentation.py           # K-Means district clustering
├── 06_intervention_map.py       # Intervention maps and priority analysis
├── 07_report_generator.py       # Automated report generation
│
├── data/
│   ├── raw/
│   └── processed/
│
├── outputs/
│   ├── figures/
│   ├── maps/
│   └── report/
│
├── requirements.txt
└── README.md
```

---

## Key Findings

### National Welfare Coverage

| Metric                                  | Result        |
| --------------------------------------- | ------------- |
| PM-KISAN average uptake                 | 71.4%         |
| Ayushman Bharat average uptake          | 71.7%         |
| PM-KISAN uncovered beneficiaries        | 70.09 million |
| Ayushman Bharat uncovered beneficiaries | 36.88 million |
| Critical Priority districts             | 134 districts |

---

### Regression Model Performance

#### PM-KISAN Model

| Metric      | Value  |
| ----------- | ------ |
| R² Score    | 0.6165 |
| Adjusted R² | 0.6124 |
| RMSE        | 0.0567 |
| MAE         | 0.0459 |

#### Ayushman Bharat Model

| Metric      | Value  |
| ----------- | ------ |
| Metric      | Value  |
| R² Score    | 0.6320 |
| Adjusted R² | 0.6280 |
| RMSE        | 0.0604 |
| MAE         | 0.0482 |

---

## Statistical Insights

* Higher literacy and internet penetration were strongly associated with increased welfare participation.
* Higher poverty and rural concentration were associated with lower scheme coverage.
* Districts with high SC/ST concentration showed statistically significant lower participation levels.
* Literacy showed a strong relationship with welfare gap categories through ANOVA testing.
* Residual-based analysis identified 17 PM-KISAN and 24 Ayushman Bharat underperforming districts.

---

## Machine Learning Segmentation

K-Means clustering classified districts into four intervention categories:

| Category     | Districts | Percentage |
| ------------ | --------- | ---------- |
| Low Coverage | 269       | 35.9%      |
