"""
Example: Using Forecastly API with Python.

This script demonstrates how to interact with the Forecastly API
using the requests library.
"""

import requests
import pandas as pd
from typing import Optional, Dict, List


class ForecastlyClient:
    """Client for interacting with Forecastly API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the Forecastly API client.

        Args:
            base_url: Base URL of the Forecastly API
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()

    def health_check(self) -> Dict:
        """
        Check if the API is healthy.

        Returns:
            Health check response
        """
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def get_skus(self) -> List[str]:
        """
        Get list of available SKUs.

        Returns:
            List of SKU IDs
        """
        response = self.session.get(f"{self.base_url}/api/v1/skus")
        response.raise_for_status()
        data = response.json()
        return data['skus']

    def get_forecast(
        self,
        sku_id: str,
        horizon: int = 14
    ) -> pd.DataFrame:
        """
        Get forecast for a specific SKU.

        Args:
            sku_id: SKU identifier
            horizon: Forecast horizon in days

        Returns:
            DataFrame with forecast data
        """
        params = {'sku_id': sku_id, 'horizon': horizon}
        response = self.session.get(
            f"{self.base_url}/api/v1/predict",
            params=params
        )
        response.raise_for_status()

        data = response.json()
        df = pd.DataFrame(data['predictions'])
        df['date'] = pd.to_datetime(df['date'])
        return df

    def get_metrics(self) -> pd.DataFrame:
        """
        Get model performance metrics.

        Returns:
            DataFrame with metrics
        """
        response = self.session.get(f"{self.base_url}/api/v1/metrics")
        response.raise_for_status()

        data = response.json()
        return pd.DataFrame(data['metrics'])

    def rebuild_forecasts(
        self,
        horizon: int = 14,
        save_to_db: bool = False
    ) -> Dict:
        """
        Rebuild forecasts for all SKUs.

        Args:
            horizon: Forecast horizon in days
            save_to_db: Whether to save to database

        Returns:
            Rebuild response
        """
        params = {'horizon': horizon, 'save_to_db': save_to_db}
        response = self.session.post(
            f"{self.base_url}/api/v1/predict/rebuild",
            params=params
        )
        response.raise_for_status()
        return response.json()

    def get_system_status(self) -> Dict:
        """
        Get system status.

        Returns:
            System status information
        """
        response = self.session.get(f"{self.base_url}/api/v1/status")
        response.raise_for_status()
        return response.json()


# ==============================================================================
# Usage Examples
# ==============================================================================

def main():
    """Main example function."""

    # Initialize client
    client = ForecastlyClient("http://localhost:8000")

    print("=" * 60)
    print("Forecastly API - Python Client Example")
    print("=" * 60)
    print()

    # 1. Health Check
    print("1. Health Check")
    print("-" * 60)
    health = client.health_check()
    print(f"Status: {health['status']}")
    print(f"Service: {health['service']} v{health['version']}")
    print()

    # 2. Get Available SKUs
    print("2. Available SKUs")
    print("-" * 60)
    skus = client.get_skus()
    print(f"Found {len(skus)} SKUs: {skus[:5]}...")
    print()

    # 3. Get Forecast for Specific SKU
    print("3. Forecast for SKU001")
    print("-" * 60)
    forecast = client.get_forecast("SKU001", horizon=14)
    print(forecast.head())
    print()

    # Calculate average forecast
    avg_forecast = forecast['ensemble'].mean()
    print(f"Average forecast (Ensemble): {avg_forecast:.2f} units/day")
    print()

    # 4. Get Metrics
    print("4. Model Performance Metrics")
    print("-" * 60)
    metrics = client.get_metrics()
    print(metrics.head())
    print()

    # Find best performing model
    best_model = metrics.groupby('best_model').size()
    print("\nModel wins:")
    print(best_model)
    print()

    # 5. System Status
    print("5. System Status")
    print("-" * 60)
    status = client.get_system_status()
    print(f"System: {status['system']}")
    print(f"Database mode: {status.get('database_mode', False)}")

    if 'data_available' in status:
        data_avail = status['data_available']
        print(f"Data available:")
        for key, value in data_avail.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {'✓' if v else '✗'}")
            else:
                print(f"  {key}: {'✓' if value else '✗'}")
    print()

    # 6. Export forecast to CSV
    print("6. Export Forecast")
    print("-" * 60)
    output_file = "forecast_SKU001.csv"
    forecast.to_csv(output_file, index=False)
    print(f"Forecast exported to: {output_file}")
    print()

    # 7. Batch Processing (Multiple SKUs)
    print("7. Batch Processing")
    print("-" * 60)
    all_forecasts = []

    for sku in skus[:5]:  # Process first 5 SKUs
        try:
            df = client.get_forecast(sku, horizon=7)
            df['sku_id'] = sku
            all_forecasts.append(df)
            print(f"✓ Processed {sku}")
        except Exception as e:
            print(f"✗ Failed {sku}: {e}")

    # Combine all forecasts
    if all_forecasts:
        combined = pd.concat(all_forecasts, ignore_index=True)
        print(f"\nCombined forecasts shape: {combined.shape}")
        combined.to_csv("forecasts_batch.csv", index=False)
        print("Batch forecasts exported to: forecasts_batch.csv")

    print()
    print("=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
