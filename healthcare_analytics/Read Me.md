# Healthcare Analytics Module

## Overview

The Healthcare Analytics module provides specialized tools and methods for analyzing healthcare data, focusing on patient segmentation, resource optimization, and clinical decision support. This module leverages advanced analytics techniques to extract actionable insights from healthcare data, allowing for more personalized patient care and efficient resource allocation.

## Main Features

### RFM Analysis (`rfm_analysis.py`)

The RFM (Recency, Frequency, Monetary) analysis tool implements a method for patient segmentation based on healthcare utilization patterns. This approach categorizes patients based on:

- **Recency**: How recently a patient has visited a healthcare facility
- **Frequency**: How often a patient uses healthcare services
- **Monetary**: The financial value associated with patient visits

This analysis helps healthcare providers:
- Identify high-risk patients who require proactive intervention
- Recognize loyal patients who might benefit from specialized programs
- Target specific patient groups for appropriate outreach
- Optimize resource allocation based on utilization patterns

### Healthcare Segmentation (`healthcare_segmentation.ipynb`)

This Jupyter notebook demonstrates advanced patient clustering and segmentation using various machine learning techniques, including:

- K-means clustering for general segmentation
- DBSCAN for density-based clustering, identifying unusual patient groups
- Hierarchical clustering for nested segmentation
- Feature importance analysis to understand key segmentation drivers
- Visualization of patient segments for clinical interpretation

The notebook explores multiple approaches to patient segmentation, enabling the identification of meaningful subgroups with similar healthcare needs, behaviors, or risks.

## Usage Examples

### RFM Analysis

```python
from healthcare_analytics.rfm_analysis import HealthcareRFM

# Create an RFM analyzer
rfm_analyzer = HealthcareRFM()

# Load patient data
patient_data = pd.read_csv('patient_visits.csv')

# Perform RFM analysis
rfm_result = rfm_analyzer.analyze(
    data=patient_data,
    patient_id_col='patient_id',
    date_col='visit_date',
    monetary_col='visit_cost'
)

# Get patient segments
segments = rfm_analyzer.segment_patients(rfm_result)

# View segment distribution
segment_distribution = rfm_analyzer.get_segment_distribution(segments)
segment_distribution.plot(kind='bar')
```

### Healthcare Segmentation

Please refer to `healthcare_segmentation.ipynb` for detailed examples and instructions on patient clustering and segmentation.

## Dependencies

- Python 3.8+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

## References

1. Wei, L., et al. (2022). "Application of RFM model in healthcare: Identifying high-value patients for personalized care." *Journal of Healthcare Analytics*, 4(2), 87-103.
2. Johnson, A., et al. (2023). "Patient segmentation using unsupervised learning: Implications for resource allocation and personalized care." *BMC Medical Informatics and Decision Making*, 23, 102.
