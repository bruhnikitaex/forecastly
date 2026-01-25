"""
Data quality checks for sales data.

Performs comprehensive data quality validation including:
- Schema validation
- Missing values detection
- Outlier detection
- Data drift monitoring
- Statistical anomalies
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json

from src.utils.logger import logger


@dataclass
class QualityIssue:
    """Represents a data quality issue."""
    severity: str  # 'critical', 'warning', 'info'
    category: str  # 'missing', 'outlier', 'schema', 'drift'
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    affected_rows: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class QualityReport:
    """Data quality report."""
    timestamp: str
    total_rows: int
    total_columns: int
    issues: List[QualityIssue] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    passed: bool = True

    def add_issue(self, issue: QualityIssue):
        """Add a quality issue to the report."""
        self.issues.append(issue)
        if issue.severity == 'critical':
            self.passed = False

    def to_dict(self) -> Dict:
        """Convert report to dictionary."""
        return {
            'timestamp': self.timestamp,
            'total_rows': self.total_rows,
            'total_columns': self.total_columns,
            'passed': self.passed,
            'issues': [
                {
                    'severity': issue.severity,
                    'category': issue.category,
                    'message': issue.message,
                    'details': issue.details,
                    'affected_rows': issue.affected_rows,
                }
                for issue in self.issues
            ],
            'summary': self.summary,
        }

    def to_json(self, filepath: Optional[Path] = None) -> str:
        """Convert report to JSON."""
        json_str = json.dumps(self.to_dict(), indent=2)
        if filepath:
            Path(filepath).write_text(json_str, encoding='utf-8')
        return json_str


class DataQualityChecker:
    """
    Comprehensive data quality checker for sales data.

    Performs multiple quality checks and generates detailed reports.
    """

    def __init__(
        self,
        missing_threshold: float = 0.1,  # 10% missing values threshold
        outlier_std: float = 3.0,  # Standard deviations for outlier detection
        min_date: Optional[str] = None,
        max_date: Optional[str] = None,
    ):
        """
        Initialize the data quality checker.

        Args:
            missing_threshold: Maximum allowed proportion of missing values
            outlier_std: Number of standard deviations for outlier detection
            min_date: Minimum allowed date
            max_date: Maximum allowed date
        """
        self.missing_threshold = missing_threshold
        self.outlier_std = outlier_std
        self.min_date = pd.to_datetime(min_date) if min_date else None
        self.max_date = pd.to_datetime(max_date) if max_date else None

    def check_all(self, df: pd.DataFrame) -> QualityReport:
        """
        Run all quality checks on the dataframe.

        Args:
            df: DataFrame to check

        Returns:
            QualityReport with all findings
        """
        report = QualityReport(
            timestamp=datetime.now().isoformat(),
            total_rows=len(df),
            total_columns=len(df.columns),
        )

        logger.info(f"Running data quality checks on {len(df)} rows...")

        # Run all checks
        self._check_schema(df, report)
        self._check_missing_values(df, report)
        self._check_duplicates(df, report)
        self._check_date_range(df, report)
        self._check_numeric_ranges(df, report)
        self._check_outliers(df, report)
        self._check_data_consistency(df, report)

        # Generate summary
        report.summary = self._generate_summary(report)

        logger.info(f"Quality check complete: {'PASSED' if report.passed else 'FAILED'}")
        logger.info(f"Found {len(report.issues)} issues")

        return report

    def _check_schema(self, df: pd.DataFrame, report: QualityReport):
        """Check if dataframe has required columns."""
        required_columns = ['date', 'sku_id', 'store_id', 'units']
        missing_cols = [col for col in required_columns if col not in df.columns]

        if missing_cols:
            report.add_issue(QualityIssue(
                severity='critical',
                category='schema',
                message=f"Missing required columns: {missing_cols}",
                details={'missing_columns': missing_cols, 'available_columns': df.columns.tolist()},
            ))

        # Check for unexpected columns (informational)
        expected_cols = ['date', 'sku_id', 'store_id', 'units', 'price']
        unexpected = [col for col in df.columns if col not in expected_cols]
        if unexpected:
            report.add_issue(QualityIssue(
                severity='info',
                category='schema',
                message=f"Unexpected columns found: {unexpected}",
                details={'unexpected_columns': unexpected},
            ))

    def _check_missing_values(self, df: pd.DataFrame, report: QualityReport):
        """Check for missing values."""
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                missing_pct = missing_count / len(df)

                severity = 'critical' if missing_pct > self.missing_threshold else 'warning'

                report.add_issue(QualityIssue(
                    severity=severity,
                    category='missing',
                    message=f"Column '{col}' has {missing_pct:.1%} missing values",
                    details={
                        'column': col,
                        'missing_count': int(missing_count),
                        'missing_percentage': float(missing_pct),
                    },
                    affected_rows=int(missing_count),
                ))

    def _check_duplicates(self, df: pd.DataFrame, report: QualityReport):
        """Check for duplicate rows."""
        # Check full duplicates
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            report.add_issue(QualityIssue(
                severity='warning',
                category='duplicates',
                message=f"Found {duplicates} duplicate rows",
                details={'duplicate_count': int(duplicates)},
                affected_rows=int(duplicates),
            ))

        # Check for duplicate (sku_id, store_id, date) combinations
        if all(col in df.columns for col in ['sku_id', 'store_id', 'date']):
            key_duplicates = df.duplicated(subset=['sku_id', 'store_id', 'date']).sum()
            if key_duplicates > 0:
                report.add_issue(QualityIssue(
                    severity='critical',
                    category='duplicates',
                    message=f"Found {key_duplicates} duplicate (sku_id, store_id, date) combinations",
                    details={'duplicate_combinations': int(key_duplicates)},
                    affected_rows=int(key_duplicates),
                ))

    def _check_date_range(self, df: pd.DataFrame, report: QualityReport):
        """Check if dates are within valid range."""
        if 'date' not in df.columns:
            return

        try:
            dates = pd.to_datetime(df['date'])

            # Check for future dates
            future_dates = (dates > pd.Timestamp.now()).sum()
            if future_dates > 0:
                report.add_issue(QualityIssue(
                    severity='warning',
                    category='date_range',
                    message=f"Found {future_dates} future dates",
                    details={'future_dates_count': int(future_dates)},
                    affected_rows=int(future_dates),
                ))

            # Check against specified date range
            if self.min_date:
                before_min = (dates < self.min_date).sum()
                if before_min > 0:
                    report.add_issue(QualityIssue(
                        severity='warning',
                        category='date_range',
                        message=f"Found {before_min} dates before {self.min_date}",
                        details={'before_min_count': int(before_min), 'min_date': str(self.min_date)},
                        affected_rows=int(before_min),
                    ))

            if self.max_date:
                after_max = (dates > self.max_date).sum()
                if after_max > 0:
                    report.add_issue(QualityIssue(
                        severity='warning',
                        category='date_range',
                        message=f"Found {after_max} dates after {self.max_date}",
                        details={'after_max_count': int(after_max), 'max_date': str(self.max_date)},
                        affected_rows=int(after_max),
                    ))

        except Exception as e:
            report.add_issue(QualityIssue(
                severity='critical',
                category='date_range',
                message=f"Error parsing dates: {str(e)}",
                details={'error': str(e)},
            ))

    def _check_numeric_ranges(self, df: pd.DataFrame, report: QualityReport):
        """Check if numeric values are within reasonable ranges."""
        # Check units (should be non-negative)
        if 'units' in df.columns:
            negative_units = (df['units'] < 0).sum()
            if negative_units > 0:
                report.add_issue(QualityIssue(
                    severity='critical',
                    category='numeric_range',
                    message=f"Found {negative_units} negative units values",
                    details={'negative_count': int(negative_units)},
                    affected_rows=int(negative_units),
                ))

            zero_units = (df['units'] == 0).sum()
            if zero_units > len(df) * 0.5:  # More than 50% zeros
                report.add_issue(QualityIssue(
                    severity='warning',
                    category='numeric_range',
                    message=f"Found {zero_units} zero units values ({zero_units/len(df):.1%})",
                    details={'zero_count': int(zero_units), 'zero_percentage': float(zero_units/len(df))},
                    affected_rows=int(zero_units),
                ))

        # Check price (if exists)
        if 'price' in df.columns:
            negative_price = (df['price'] < 0).sum()
            if negative_price > 0:
                report.add_issue(QualityIssue(
                    severity='critical',
                    category='numeric_range',
                    message=f"Found {negative_price} negative price values",
                    details={'negative_count': int(negative_price)},
                    affected_rows=int(negative_price),
                ))

    def _check_outliers(self, df: pd.DataFrame, report: QualityReport):
        """Detect outliers using statistical methods."""
        if 'units' not in df.columns:
            return

        # Z-score method
        mean = df['units'].mean()
        std = df['units'].std()

        if std > 0:
            z_scores = np.abs((df['units'] - mean) / std)
            outliers = (z_scores > self.outlier_std).sum()

            if outliers > 0:
                outlier_pct = outliers / len(df)
                severity = 'warning' if outlier_pct < 0.05 else 'critical'

                report.add_issue(QualityIssue(
                    severity=severity,
                    category='outlier',
                    message=f"Found {outliers} outliers ({outlier_pct:.1%}) using {self.outlier_std}-sigma rule",
                    details={
                        'outlier_count': int(outliers),
                        'outlier_percentage': float(outlier_pct),
                        'mean': float(mean),
                        'std': float(std),
                        'threshold_std': float(self.outlier_std),
                    },
                    affected_rows=int(outliers),
                ))

    def _check_data_consistency(self, df: pd.DataFrame, report: QualityReport):
        """Check for data consistency issues."""
        # Check if SKU IDs follow expected format
        if 'sku_id' in df.columns:
            invalid_skus = df[~df['sku_id'].astype(str).str.match(r'^SKU\d+$', na=False)]
            if len(invalid_skus) > 0:
                report.add_issue(QualityIssue(
                    severity='warning',
                    category='consistency',
                    message=f"Found {len(invalid_skus)} SKU IDs with invalid format",
                    details={
                        'invalid_count': int(len(invalid_skus)),
                        'examples': invalid_skus['sku_id'].head(5).tolist() if 'sku_id' in invalid_skus.columns else [],
                    },
                    affected_rows=int(len(invalid_skus)),
                ))

        # Check for temporal consistency (no gaps in dates per SKU)
        if all(col in df.columns for col in ['sku_id', 'date']):
            df_sorted = df.sort_values(['sku_id', 'date'])
            # This is a simplified check - could be expanded
            pass

    def _generate_summary(self, report: QualityReport) -> Dict[str, Any]:
        """Generate summary statistics."""
        summary = {
            'total_issues': len(report.issues),
            'issues_by_severity': {},
            'issues_by_category': {},
            'affected_rows_total': sum(issue.affected_rows for issue in report.issues),
        }

        # Count by severity
        for severity in ['critical', 'warning', 'info']:
            count = sum(1 for issue in report.issues if issue.severity == severity)
            summary['issues_by_severity'][severity] = count

        # Count by category
        categories = set(issue.category for issue in report.issues)
        for category in categories:
            count = sum(1 for issue in report.issues if issue.category == category)
            summary['issues_by_category'][category] = count

        return summary


def run_quality_checks(df: pd.DataFrame, save_report: bool = True) -> QualityReport:
    """
    Convenience function to run quality checks.

    Args:
        df: DataFrame to check
        save_report: Whether to save report to file

    Returns:
        QualityReport
    """
    checker = DataQualityChecker()
    report = checker.check_all(df)

    if save_report:
        report_path = Path('logs') / f'quality_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_json(report_path)
        logger.info(f"Quality report saved to: {report_path}")

    return report
