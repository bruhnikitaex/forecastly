"""
Tests for data quality checks.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.etl.quality import (
    DataQualityChecker,
    QualityReport,
    QualityIssue,
    validate_schema,
    validate_date_range,
    validate_numeric_range,
    check_duplicates,
    check_missing_values,
    detect_outliers_zscore,
    detect_outliers_iqr,
)


class TestDataQualityChecker:
    """Tests for DataQualityChecker class."""

    @pytest.fixture
    def valid_data(self):
        """Create valid test data."""
        return pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=100, freq='D'),
            'sku_id': ['SKU001'] * 100,
            'store_id': ['S01'] * 100,
            'units': np.random.randint(10, 50, 100),
            'price': np.random.uniform(100, 200, 100),
        })

    @pytest.fixture
    def data_with_issues(self):
        """Create data with quality issues."""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=100, freq='D'),
            'sku_id': ['SKU001'] * 100,
            'store_id': ['S01'] * 100,
            'units': np.random.randint(10, 50, 100),
            'price': np.random.uniform(100, 200, 100),
        })

        # Add missing values
        df.loc[0:10, 'units'] = np.nan

        # Add negative values
        df.loc[20:25, 'units'] = -5

        # Add duplicates
        df = pd.concat([df, df.iloc[0:5]], ignore_index=True)

        return df

    def test_valid_data_passes(self, valid_data):
        """Valid data should pass all checks."""
        checker = DataQualityChecker()
        report = checker.check_all(valid_data)

        assert report.total_rows == len(valid_data)
        assert report.total_columns == len(valid_data.columns)
        # Should only have info-level issues (if any)
        critical_issues = [i for i in report.issues if i.severity == 'critical']
        assert len(critical_issues) == 0

    def test_missing_schema_columns(self):
        """Should detect missing required columns."""
        df = pd.DataFrame({'col1': [1, 2, 3]})
        checker = DataQualityChecker()
        report = checker.check_all(df)

        assert not report.passed
        schema_issues = [i for i in report.issues if i.category == 'schema']
        assert len(schema_issues) > 0

    def test_detect_missing_values(self, data_with_issues):
        """Should detect missing values."""
        checker = DataQualityChecker(missing_threshold=0.05)
        report = checker.check_all(data_with_issues)

        missing_issues = [i for i in report.issues if i.category == 'missing']
        assert len(missing_issues) > 0

    def test_detect_negative_values(self, data_with_issues):
        """Should detect negative values in units column."""
        checker = DataQualityChecker()
        report = checker.check_all(data_with_issues)

        numeric_issues = [i for i in report.issues if i.category == 'numeric_range']
        assert len(numeric_issues) > 0

        negative_issue = [i for i in numeric_issues if 'negative' in i.message.lower()]
        assert len(negative_issue) > 0

    def test_detect_duplicates(self, data_with_issues):
        """Should detect duplicate rows."""
        checker = DataQualityChecker()
        report = checker.check_all(data_with_issues)

        dup_issues = [i for i in report.issues if i.category == 'duplicates']
        assert len(dup_issues) > 0

    def test_detect_outliers(self):
        """Should detect outliers."""
        # Create data with outliers
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=100, freq='D'),
            'sku_id': ['SKU001'] * 100,
            'store_id': ['S01'] * 100,
            'units': np.concatenate([
                np.random.randint(10, 20, 95),
                [100, 150, 200, 250, 300]  # Outliers
            ]),
        })

        checker = DataQualityChecker(outlier_std=2.0)
        report = checker.check_all(df)

        outlier_issues = [i for i in report.issues if i.category == 'outlier']
        assert len(outlier_issues) > 0

    def test_report_to_dict(self, valid_data):
        """Report should convert to dictionary."""
        checker = DataQualityChecker()
        report = checker.check_all(valid_data)

        report_dict = report.to_dict()

        assert 'timestamp' in report_dict
        assert 'total_rows' in report_dict
        assert 'issues' in report_dict
        assert 'summary' in report_dict

    def test_report_summary(self, data_with_issues):
        """Report should have correct summary."""
        checker = DataQualityChecker()
        report = checker.check_all(data_with_issues)

        assert 'total_issues' in report.summary
        assert 'issues_by_severity' in report.summary
        assert 'issues_by_category' in report.summary


class TestValidators:
    """Tests for individual validator functions."""

    def test_validate_schema_valid(self):
        """Should pass with all required columns."""
        df = pd.DataFrame({'col1': [1], 'col2': [2], 'col3': [3]})
        is_valid, missing = validate_schema(df, ['col1', 'col2'])

        assert is_valid
        assert len(missing) == 0

    def test_validate_schema_missing_columns(self):
        """Should detect missing columns."""
        df = pd.DataFrame({'col1': [1]})
        is_valid, missing = validate_schema(df, ['col1', 'col2', 'col3'])

        assert not is_valid
        assert 'col2' in missing
        assert 'col3' in missing

    def test_validate_date_range_valid(self):
        """Should pass with valid date range."""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10, freq='D')
        })

        is_valid, details = validate_date_range(
            df,
            min_date=datetime(2023, 1, 1),
            max_date=datetime(2025, 1, 1)
        )

        assert is_valid
        assert len(details) == 0

    def test_validate_date_range_future_dates(self):
        """Should detect future dates."""
        df = pd.DataFrame({
            'date': pd.date_range('2030-01-01', periods=10, freq='D')
        })

        is_valid, details = validate_date_range(df)

        assert not is_valid
        assert 'future_dates' in details

    def test_validate_numeric_range_valid(self):
        """Should pass with valid numeric range."""
        df = pd.DataFrame({'values': [10, 20, 30, 40, 50]})

        is_valid, details = validate_numeric_range(
            df, 'values', min_value=0, max_value=100
        )

        assert is_valid
        assert len(details) == 0

    def test_validate_numeric_range_negative(self):
        """Should detect negative values when not allowed."""
        df = pd.DataFrame({'values': [-5, 10, 20, 30]})

        is_valid, details = validate_numeric_range(
            df, 'values', allow_negative=False
        )

        assert not is_valid
        assert 'negative_values' in details

    def test_check_duplicates_none(self):
        """Should return 0 for data without duplicates."""
        df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})

        count, dups = check_duplicates(df)

        assert count == 0
        assert len(dups) == 0

    def test_check_duplicates_found(self):
        """Should detect duplicates."""
        df = pd.DataFrame({
            'col1': [1, 2, 3, 1],
            'col2': [4, 5, 6, 4]
        })

        count, dups = check_duplicates(df)

        assert count == 1
        assert len(dups) == 1

    def test_check_missing_values_none(self):
        """Should return empty dict for data without missing values."""
        df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})

        result = check_missing_values(df)

        assert len(result) == 0

    def test_check_missing_values_found(self):
        """Should detect missing values."""
        df = pd.DataFrame({
            'col1': [1, np.nan, 3],
            'col2': [4, 5, np.nan]
        })

        result = check_missing_values(df, threshold=0.2)

        assert 'col1' in result
        assert 'col2' in result
        assert result['col1']['count'] == 1
        assert result['col2']['count'] == 1

    def test_detect_outliers_zscore(self):
        """Should detect outliers using Z-score method."""
        df = pd.DataFrame({
            'values': [10, 11, 12, 13, 14, 15, 100]  # 100 is outlier
        })

        count, mask = detect_outliers_zscore(df, 'values', threshold=2.0)

        assert count >= 1
        assert mask.iloc[-1] == True  # Last value should be outlier

    def test_detect_outliers_iqr(self):
        """Should detect outliers using IQR method."""
        df = pd.DataFrame({
            'values': [10, 11, 12, 13, 14, 15, 100]  # 100 is outlier
        })

        count, mask = detect_outliers_iqr(df, 'values', multiplier=1.5)

        assert count >= 1
        assert mask.iloc[-1] == True  # Last value should be outlier


class TestQualityIssue:
    """Tests for QualityIssue class."""

    def test_create_issue(self):
        """Should create quality issue."""
        issue = QualityIssue(
            severity='critical',
            category='missing',
            message='Test issue',
            details={'test': 'value'},
            affected_rows=10
        )

        assert issue.severity == 'critical'
        assert issue.category == 'missing'
        assert issue.message == 'Test issue'
        assert issue.details == {'test': 'value'}
        assert issue.affected_rows == 10
        assert issue.timestamp is not None


class TestQualityReport:
    """Tests for QualityReport class."""

    def test_create_report(self):
        """Should create quality report."""
        report = QualityReport(
            timestamp=datetime.now().isoformat(),
            total_rows=100,
            total_columns=5
        )

        assert report.total_rows == 100
        assert report.total_columns == 5
        assert len(report.issues) == 0
        assert report.passed == True

    def test_add_issue_changes_passed_status(self):
        """Adding critical issue should set passed to False."""
        report = QualityReport(
            timestamp=datetime.now().isoformat(),
            total_rows=100,
            total_columns=5
        )

        assert report.passed == True

        report.add_issue(QualityIssue(
            severity='critical',
            category='test',
            message='Critical issue'
        ))

        assert report.passed == False

    def test_add_warning_doesnt_change_passed(self):
        """Adding warning should not change passed status."""
        report = QualityReport(
            timestamp=datetime.now().isoformat(),
            total_rows=100,
            total_columns=5
        )

        report.add_issue(QualityIssue(
            severity='warning',
            category='test',
            message='Warning issue'
        ))

        assert report.passed == True
