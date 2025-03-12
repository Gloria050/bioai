# BioAI: Statistical Methods for Biological Data Science

A computational toolkit for bioinformatics experimentation, patient stratification, and clinical data analysis.

## Core Components

### 1. Stable Hash Partitioning for Experimental Design

```python
def hash_assign(identifier, groups=["control", "treatment"], seed=42):
    """Deterministic group assignment with MurmurHash3"""
    hash_value = mmh3.hash(str(identifier), seed)
    return groups[hash_value % len(groups)]
```

**Applications:** Subject randomization in clinical trials, A/B testing for treatment protocols, ensuring reproducible experiment designs

### 2. Statistical Inference Framework

Rigorous experimental analysis with biological significance interpretation:

- Minimum Detectable Effect (MDE) calculation
- Effect size estimation with confidence intervals
- Biological vs. statistical significance evaluation

### 3. Patient Stratification

RFM (Recency, Frequency, Monetary) analysis and clustering techniques for:
- Identifying high-value patient cohorts
- Predicting treatment adherence patterns
- Optimizing resource allocation in clinical settings

## Implementation Example

```python
from bioai.experiment import BioExperiment
from bioai.distribution import hash_assignment

# 1. Design experiment
experiment = BioExperiment(
    baseline_rate=0.15,
    sample_size=200,
    alpha=0.05
)

# 2. Calculate required parameters
mde = experiment.calculate_mde()
print(f"Minimum Detectable Effect: {mde:.4f}")

# 3. Assign subjects to groups
patient_ids = ["PTID001", "PTID002", "PTID003"]
assignments = {pid: hash_assignment(pid) for pid in patient_ids}

# 4. Analyze results
results = experiment.analyze(control_data, treatment_data)
```

## Case Study: Medicaid Contract Compliance Analysis

Implemented a compliance analysis system for healthcare insurance regulations:

1. Applied NLP algorithms to parse Medicaid regulatory documents
2. Developed inference models to identify compliance gaps between contracts and regulations
3. Built GenAI application to generate compliance assessments and recommendations
4. Validated approach across 5 state contract datasets

**Results:** System identified 83% of compliance issues with 91% precision, reduced manual review time by 67%.

## Technical Stack

- **Core:** Python, NumPy, SciPy
- **Statistical Analysis:** statsmodels, sklearn
- **Data Processing:** pandas, mmh3
- **Visualization:** matplotlib, seaborn

## Repository Structure

```
bioai/
├── distribution.py   # Hash-based assignment functions
├── experiment.py     # Statistical inference framework
├── stratification.py # Patient segmentation methods
├── utils/
│   ├── metrics.py    # Evaluation metrics
│   └── viz.py        # Visualization tools
└── examples/
    ├── clinical_trial.ipynb
    └── patient_segmentation.ipynb
```
