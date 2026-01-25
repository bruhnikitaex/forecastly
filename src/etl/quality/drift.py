"""
Data drift detection for monitoring data quality over time.

Detects changes in data distribution that may affect model performance.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
from datetime import datetime

from src.utils.logger import logger


class DataDriftDetector:
    """
    Detect data drift between reference and current datasets.

    Uses statistical tests to identify significant changes in data distribution.
    """

    def __init__(self, significance_level: float = 0.05):
        """
        Initialize drift detector.

        Args:
            significance_level: P-value threshold for statistical tests
        """
        self.significance_level = significance_level
        self.reference_stats: Optional[Dict] = None

    def fit(self, reference_df: pd.DataFrame):
        """
        Fit detector on reference dataset.

        Args:
            reference_df: Reference DataFrame
        """
        self.reference_stats = self._calculate_statistics(reference_df)
        logger.info("Data drift detector fitted on reference data")

    def detect(
        self,
        current_df: pd.DataFrame,
        columns: Optional[List[str]] = None
    ) -> Dict[str, Dict]:
        """
        Detect drift between reference and current data.

        Args:
            current_df: Current DataFrame to check
            columns: Columns to check (None = all numeric columns)

        Returns:
            Dictionary with drift results per column
        """
        if self.reference_stats is None:
            raise ValueError("Detector not fitted. Call fit() first.")

        if columns is None:
            columns = current_df.select_dtypes(include=[np.number]).columns.tolist()

        current_stats = self._calculate_statistics(current_df)
        drift_results = {}

        for col in columns:
            if col not in self.reference_stats or col not in current_stats:
                continue

            drift_results[col] = self._detect_column_drift(
                col,
                self.reference_stats[col],
                current_stats[col]
            )

        return drift_results

    def _calculate_statistics(self, df: pd.DataFrame) -> Dict:
        """Calculate statistics for each numeric column."""
        stats_dict = {}

        for col in df.select_dtypes(include=[np.number]).columns:
            values = df[col].dropna()

            if len(values) == 0:
                continue

            stats_dict[col] = {
                'mean': float(values.mean()),
                'std': float(values.std()),
                'min': float(values.min()),
                'max': float(values.max()),
                'median': float(values.median()),
                'q25': float(values.quantile(0.25)),
                'q75': float(values.quantile(0.75)),
                'count': len(values),
                'values': values.values  # Store for KS test
            }

        return stats_dict

    def _detect_column_drift(
        self,
        column: str,
        reference_stats: Dict,
        current_stats: Dict
    ) -> Dict:
        """Detect drift for a single column."""
        # Kolmogorov-Smirnov test
        ks_statistic, ks_pvalue = stats.ks_2samp(
            reference_stats['values'],
            current_stats['values']
        )

        # Calculate percentage changes
        mean_change = (
            (current_stats['mean'] - reference_stats['mean']) /
            reference_stats['mean'] * 100
            if reference_stats['mean'] != 0 else 0
        )

        std_change = (
            (current_stats['std'] - reference_stats['std']) /
            reference_stats['std'] * 100
            if reference_stats['std'] != 0 else 0
        )

        # Determine if drift is significant
        has_drift = ks_pvalue < self.significance_level

        return {
            'has_drift': bool(has_drift),
            'ks_statistic': float(ks_statistic),
            'ks_pvalue': float(ks_pvalue),
            'mean_change_pct': float(mean_change),
            'std_change_pct': float(std_change),
            'reference': {
                'mean': reference_stats['mean'],
                'std': reference_stats['std'],
                'count': reference_stats['count']
            },
            'current': {
                'mean': current_stats['mean'],
                'std': current_stats['std'],
                'count': current_stats['count']
            }
        }


def detect_concept_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    target_column: str = 'units',
    feature_columns: Optional[List[str]] = None
) -> Dict:
    """
    Detect concept drift (changes in target distribution).

    Args:
        reference_df: Reference DataFrame
        current_df: Current DataFrame
        target_column: Target column name
        feature_columns: Feature column names

    Returns:
        Dictionary with drift information
    """
    if target_column not in reference_df.columns or target_column not in current_df.columns:
        return {'error': f'Column {target_column} not found'}

    ref_values = reference_df[target_column].dropna()
    cur_values = current_df[target_column].dropna()

    # KS test
    ks_stat, ks_pval = stats.ks_2samp(ref_values, cur_values)

    # Distribution comparison
    ref_mean = ref_values.mean()
    cur_mean = cur_values.mean()
    mean_change = ((cur_mean - ref_mean) / ref_mean * 100) if ref_mean != 0 else 0

    return {
        'target_column': target_column,
        'has_concept_drift': ks_pval < 0.05,
        'ks_statistic': float(ks_stat),
        'ks_pvalue': float(ks_pval),
        'reference_mean': float(ref_mean),
        'current_mean': float(cur_mean),
        'mean_change_pct': float(mean_change),
        'reference_count': len(ref_values),
        'current_count': len(cur_values)
    }


def calculate_psi(
    reference: pd.Series,
    current: pd.Series,
    bins: int = 10
) -> float:
    """
    Calculate Population Stability Index (PSI).

    PSI is a measure of how much a population has shifted over time.
    - PSI < 0.1: No significant change
    - 0.1 <= PSI < 0.2: Moderate change
    - PSI >= 0.2: Significant change

    Args:
        reference: Reference data
        current: Current data
        bins: Number of bins for discretization

    Returns:
        PSI value
    """
    # Create bins based on reference data
    breakpoints = np.percentile(reference, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)  # Remove duplicates

    # Calculate distributions
    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    cur_counts, _ = np.histogram(current, bins=breakpoints)

    # Normalize to get percentages
    ref_percents = ref_counts / len(reference)
    cur_percents = cur_counts / len(current)

    # Avoid division by zero
    ref_percents = np.where(ref_percents == 0, 0.0001, ref_percents)
    cur_percents = np.where(cur_percents == 0, 0.0001, cur_percents)

    # Calculate PSI
    psi = np.sum((cur_percents - ref_percents) * np.log(cur_percents / ref_percents))

    return float(psi)
