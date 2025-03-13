#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Healthcare RFM Analysis Tool

This module implements RFM (Recency, Frequency, Monetary) analysis for healthcare settings, 
specifically designed for patient segmentation based on healthcare utilization patterns.

Author: Gloria
Date: 2025-03-11
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple, Union, Optional


class HealthcareRFM:
    """
    A class for performing RFM (Recency, Frequency, Monetary) analysis in healthcare settings.
    
    RFM analysis segments patients based on three key metrics:
    - Recency: How recently a patient visited a healthcare facility
    - Frequency: How often a patient uses healthcare services
    - Monetary: The financial value associated with patient visits
    
    This class provides methods to calculate RFM scores, segment patients, 
    and visualize the results.
    """
    
    def __init__(self, 
                 recency_scoring: Dict[Tuple[int, int], int] = None,
                 frequency_scoring: Dict[Tuple[int, int], int] = None,
                 monetary_scoring: Dict[Tuple[float, float], int] = None,
                 segment_definitions: Dict[str, Dict[str, int]] = None):
        """
        Initialize the HealthcareRFM analyzer with optional custom scoring parameters.
        
        Args:
            recency_scoring: Dictionary mapping (min_days, max_days) to scores
            frequency_scoring: Dictionary mapping (min_visits, max_visits) to scores
            monetary_scoring: Dictionary mapping (min_value, max_value) to scores
            segment_definitions: Dictionary defining patient segments based on RFM scores
        """
        # Default recency scoring (days since last visit)
        self.recency_scoring = recency_scoring or {
            (0, 30): 5,     # Very recent visit (0-30 days)
            (31, 90): 4,    # Recent visit (31-90 days)
            (91, 180): 3,   # Moderate recency (91-180 days)
            (181, 365): 2,  # Not recent (181-365 days)
            (366, float('inf')): 1  # Long time no visit (366+ days)
        }
        
        # Default frequency scoring (number of visits)
        self.frequency_scoring = frequency_scoring or {
            (10, float('inf')): 5,  # Very frequent (10+ visits)
            (7, 9): 4,              # Frequent (7-9 visits)
            (4, 6): 3,              # Moderate frequency (4-6 visits)
            (2, 3): 2,              # Infrequent (2-3 visits)
            (0, 1): 1               # Rare (0-1 visits)
        }
        
        # Default monetary scoring (healthcare costs)
        # Ranges depend on specific healthcare context
        self.monetary_scoring = monetary_scoring or {
            (5000, float('inf')): 5,    # Very high value
            (2500, 4999): 4,            # High value
            (1000, 2499): 3,            # Medium value
            (500, 999): 2,              # Low value
            (0, 499): 1                 # Very low value
        }
        
        # Default segment definitions based on RFM scores
        self.segment_definitions = segment_definitions or {
            'High-Risk Patient': {'min_r': 1, 'max_r': 2, 'min_f': 4, 'max_f': 5, 'min_m': 4, 'max_m': 5},
            'Chronic Care Patient': {'min_r': 4, 'max_r': 5, 'min_f': 4, 'max_f': 5, 'min_m': 3, 'max_m': 5},
            'Loyal Patient': {'min_r': 3, 'max_r': 5, 'min_f': 3, 'max_f': 5, 'min_m': 3, 'max_m': 5},
            'Potential Loyal': {'min_r': 3, 'max_r': 5, 'min_f': 1, 'max_f': 2, 'min_m': 3, 'max_m': 5},
            'Need Attention': {'min_r': 1, 'max_r': 2, 'min_f': 1, 'max_f': 3, 'min_m': 1, 'max_m': 3},
            'At Risk': {'min_r': 1, 'max_r': 2, 'min_f': 4, 'max_f': 5, 'min_m': 1, 'max_m': 3},
            'One-Time Patient': {'min_r': 1, 'max_r': 5, 'min_f': 1, 'max_f': 1, 'min_m': 1, 'max_m': 5}
        }
    
    def _calculate_recency(self, data: pd.DataFrame, 
                          date_col: str, 
                          patient_id_col: str,
                          reference_date: Optional[Union[str, datetime]] = None) -> pd.DataFrame:
        """
        Calculate recency score for each patient.
        
        Args:
            data: DataFrame containing patient visit data
            date_col: Name of the column containing visit dates
            patient_id_col: Name of the column containing patient IDs
            reference_date: Reference date for recency calculation, default is current date
            
        Returns:
            pd.DataFrame: DataFrame with patient IDs and recency scores
        """
        # Convert date column to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
            data[date_col] = pd.to_datetime(data[date_col])
        
        # Set reference date if not provided
        if reference_date is None:
            reference_date = datetime.now()
        elif isinstance(reference_date, str):
            reference_date = pd.to_datetime(reference_date)
        
        # Get the most recent visit date for each patient
        last_visit = data.groupby(patient_id_col)[date_col].max().reset_index()
        last_visit.columns = [patient_id_col, 'last_visit_date']
        
        # Calculate days since last visit
        last_visit['days_since_last_visit'] = (reference_date - last_visit['last_visit_date']).dt.days
        
        # Assign recency score based on days since last visit
        last_visit['recency_score'] = 0
        for (min_days, max_days), score in self.recency_scoring.items():
            mask = (last_visit['days_since_last_visit'] >= min_days) & (last_visit['days_since_last_visit'] <= max_days)
            last_visit.loc[mask, 'recency_score'] = score
        
        return last_visit[[patient_id_col, 'days_since_last_visit', 'recency_score']]
    
    def _calculate_frequency(self, data: pd.DataFrame, 
                            patient_id_col: str, 
                            date_col: Optional[str] = None) -> pd.DataFrame:
        """
        Calculate frequency score for each patient.
        
        Args:
            data: DataFrame containing patient visit data
            patient_id_col: Name of the column containing patient IDs
            date_col: Name of the column containing visit dates (optional, for unique visit counting)
            
        Returns:
            pd.DataFrame: DataFrame with patient IDs and frequency scores
        """
        # Count visits for each patient
        if date_col is not None:
            # Count unique dates for each patient
            frequency = data.groupby(patient_id_col)[date_col].nunique().reset_index()
            frequency.columns = [patient_id_col, 'visit_count']
        else:
            # Simply count rows for each patient
            frequency = data.groupby(patient_id_col).size().reset_index()
            frequency.columns = [patient_id_col, 'visit_count']
        
        # Assign frequency score based on visit count
        frequency['frequency_score'] = 0
        for (min_visits, max_visits), score in self.frequency_scoring.items():
            mask = (frequency['visit_count'] >= min_visits) & (frequency['visit_count'] <= max_visits)
            frequency.loc[mask, 'frequency_score'] = score
        
        return frequency[[patient_id_col, 'visit_count', 'frequency_score']]
    
    def _calculate_monetary(self, data: pd.DataFrame, 
                           patient_id_col: str, 
                           monetary_col: str) -> pd.DataFrame:
        """
        Calculate monetary score for each patient.
        
        Args:
            data: DataFrame containing patient visit data
            patient_id_col: Name of the column containing patient IDs
            monetary_col: Name of the column containing monetary values
            
        Returns:
            pd.DataFrame: DataFrame with patient IDs and monetary scores
        """
        # Calculate total monetary value for each patient
        monetary = data.groupby(patient_id_col)[monetary_col].sum().reset_index()
        monetary.columns = [patient_id_col, 'total_value']
        
        # Assign monetary score based on total value
        monetary['monetary_score'] = 0
        for (min_value, max_value), score in self.monetary_scoring.items():
            mask = (monetary['total_value'] >= min_value) & (monetary['total_value'] <= max_value)
            monetary.loc[mask, 'monetary_score'] = score
        
        return monetary[[patient_id_col, 'total_value', 'monetary_score']]
    
    def analyze(self, data: pd.DataFrame, 
               patient_id_col: str,
               date_col: str, 
               monetary_col: str,
               reference_date: Optional[Union[str, datetime]] = None) -> pd.DataFrame:
        """
        Perform RFM analysis on patient data.
        
        Args:
            data: DataFrame containing patient visit data
            patient_id_col: Name of the column containing patient IDs
            date_col: Name of the column containing visit dates
            monetary_col: Name of the column containing monetary values
            reference_date: Reference date for recency calculation, default is current date
            
        Returns:
            pd.DataFrame: DataFrame with patient IDs and RFM scores
        """
        # Calculate recency, frequency, and monetary scores
        recency_df = self._calculate_recency(data, date_col, patient_id_col, reference_date)
        frequency_df = self._calculate_frequency(data, patient_id_col, date_col)
        monetary_df = self._calculate_monetary(data, patient_id_col, monetary_col)
        
        # Merge results
        rfm_result = recency_df.merge(frequency_df, on=patient_id_col)
        rfm_result = rfm_result.merge(monetary_df, on=patient_id_col)
        
        # Calculate combined RFM score
        rfm_result['rfm_score'] = (
            rfm_result['recency_score'] * 100 + 
            rfm_result['frequency_score'] * 10 + 
            rfm_result['monetary_score']
        )
        
        return rfm_result
    
    def segment_patients(self, rfm_result: pd.DataFrame) -> pd.DataFrame:
        """
        Assign patients to segments based on their RFM scores.
        
        Args:
            rfm_result: DataFrame containing RFM analysis results
            
        Returns:
            pd.DataFrame: DataFrame with patient IDs and segment assignments
        """
        # Create a copy to avoid modifying the original DataFrame
        result = rfm_result.copy()
        
        # Initialize segment column
        result['segment'] = 'Other'
        
        # Assign segments based on definitions
        for segment, criteria in self.segment_definitions.items():
            mask = (
                (result['recency_score'] >= criteria['min_r']) & 
                (result['recency_score'] <= criteria['max_r']) &
                (result['frequency_score'] >= criteria['min_f']) & 
                (result['frequency_score'] <= criteria['max_f']) &
                (result['monetary_score'] >= criteria['min_m']) & 
                (result['monetary_score'] <= criteria['max_m'])
            )
            result.loc[mask, 'segment'] = segment
        
        return result
    
    def get_segment_distribution(self, segmented_result: pd.DataFrame) -> pd.Series:
        """
        Get the distribution of patients across segments.
        
        Args:
            segmented_result: DataFrame containing segmented RFM results
            
        Returns:
            pd.Series: Series containing segment counts
        """
        return segmented_result['segment'].value_counts()
    
    def plot_segment_distribution(self, segmented_result: pd.DataFrame, 
                                 figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Plot the distribution of patients across segments.
        
        Args:
            segmented_result: DataFrame containing segmented RFM results
            figsize: Figure size as a tuple (width, height)
        """
        plt.figure(figsize=figsize)
        
        # Get segment distribution
        segment_counts = self.get_segment_distribution(segmented_result)
        
        # Create bar plot
        ax = segment_counts.plot(kind='bar', color='skyblue')
        plt.title('Patient Segment Distribution', fontsize=14)
        plt.xlabel('Segment', fontsize=12)
        plt.ylabel('Number of Patients', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # Add count labels on top of bars
        for i, count in enumerate(segment_counts):
            ax.text(i, count + 0.5, str(count), ha='center')
        
        plt.tight_layout()
        plt.show()
    
    def plot_rfm_heatmap(self, segmented_result: pd.DataFrame, 
                        figsize: Tuple[int, int] = (15, 6)) -> None:
        """
        Create a heatmap visualization of RFM scores by segment.
        
        Args:
            segmented_result: DataFrame containing segmented RFM results
            figsize: Figure size as a tuple (width, height)
        """
        plt.figure(figsize=figsize)
        
        # Calculate average RFM scores by segment
        segment_avg = segmented_result.groupby('segment')[
            ['recency_score', 'frequency_score', 'monetary_score']
        ].mean().reset_index()
        
        # Reshape data for heatmap
        heatmap_data = segment_avg.set_index('segment')
        
        # Create heatmap
        sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', linewidths=0.5, fmt='.2f')
        plt.title('Average RFM Scores by Patient Segment', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    def get_patients_in_segment(self, segmented_result: pd.DataFrame, 
                               segment: str, 
                               patient_id_col: str) -> List:
        """
        Get list of patients in a specific segment.
        
        Args:
            segmented_result: DataFrame containing segmented RFM results
            segment: Name of the segment to filter by
            patient_id_col: Name of the column containing patient IDs
            
        Returns:
            List: List of patient IDs in the specified segment
        """
        return segmented_result[segmented_result['segment'] == segment][patient_id_col].tolist()


if __name__ == "__main__":
    # Example usage
    
    # Generate synthetic patient data
    np.random.seed(42)
    n_patients = 1000
    n_visits = 5000
    
    # Patient IDs
    patient_ids = [f"P{i:04d}" for i in range(1, n_patients + 1)]
    
    # Generate random visits
    visit_data = {
        'patient_id': np.random.choice(patient_ids, size=n_visits),
        'visit_date': pd.date_range(start='2023-01-01', end='2025-03-01', periods=n_visits),
        'visit_cost': np.random.gamma(shape=5, scale=200, size=n_visits).round(2)
    }
    
    # Create DataFrame
    df = pd.DataFrame(visit_data)
    
    # Add some randomness to dates
    df['visit_date'] = df['visit_date'] + pd.to_timedelta(np.random.randint(0, 60, size=n_visits), unit='d')
    
    # Sort by patient and date
    df = df.sort_values(['patient_id', 'visit_date'])
    
    # Create and configure RFM analyzer
    rfm_analyzer = HealthcareRFM()
    
    # Perform RFM analysis
    rfm_result = rfm_analyzer.analyze(
        data=df,
        patient_id_col='patient_id',
        date_col='visit_date',
        monetary_col='visit_cost',
        reference_date='2025-03-10'
    )
    
    # Segment patients
    segmented_result = rfm_analyzer.segment_patients(rfm_result)
    
    # Print segment distribution
    print("Patient Segment Distribution:")
    print(rfm_analyzer.get_segment_distribution(segmented_result))
    
    # Plot segment distribution
    rfm_analyzer.plot_segment_distribution(segmented_result)
    
    # Plot RFM heatmap
    rfm_analyzer.plot_rfm_heatmap(segmented_result)
    
    # Get patients in high-risk segment
    high_risk_patients = rfm_analyzer.get_patients_in_segment(
        segmented_result,
        segment='High-Risk Patient',
        patient_id_col='patient_id'
    )
    
    print(f"\nNumber of high-risk patients: {len(high_risk_patients)}")
    print(f"Sample of high-risk patients: {high_risk_patients[:5]}")


# In[ ]:




