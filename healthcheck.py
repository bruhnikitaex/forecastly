#!/usr/bin/env python3
"""
Health check script for Docker containers.

This script is used by Docker HEALTHCHECK to verify that the service is running correctly.
Returns exit code 0 if healthy, 1 if unhealthy.
"""

import sys
import os
import requests
from typing import Optional


def check_api_health(host: str = "localhost", port: int = 8000, timeout: int = 5) -> bool:
    """
    Check if the API is healthy.

    Args:
        host: API host
        port: API port
        timeout: Request timeout in seconds

    Returns:
        True if healthy, False otherwise
    """
    url = f"http://{host}:{port}/health"

    try:
        response = requests.get(url, timeout=timeout)

        if response.status_code != 200:
            print(f"Health check failed: HTTP {response.status_code}", file=sys.stderr)
            return False

        data = response.json()

        if data.get("status") != "ok":
            print(f"Health check failed: status is '{data.get('status')}'", file=sys.stderr)
            return False

        # Optionally check database connection if enabled
        if data.get("database_mode"):
            db_connected = data.get("database_connected")
            if db_connected is False:
                print("Health check warning: database not connected", file=sys.stderr)
                # Don't fail on DB connection - service can still work with CSV

        print(f"Health check passed: {data.get('service')} v{data.get('version')}")
        return True

    except requests.exceptions.Timeout:
        print(f"Health check failed: timeout after {timeout}s", file=sys.stderr)
        return False
    except requests.exceptions.ConnectionError:
        print(f"Health check failed: cannot connect to {url}", file=sys.stderr)
        return False
    except requests.exceptions.RequestException as e:
        print(f"Health check failed: {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Health check failed: unexpected error: {e}", file=sys.stderr)
        return False


def check_dashboard_health(host: str = "localhost", port: int = 8501, timeout: int = 5) -> bool:
    """
    Check if the Streamlit dashboard is healthy.

    Args:
        host: Dashboard host
        port: Dashboard port
        timeout: Request timeout in seconds

    Returns:
        True if healthy, False otherwise
    """
    url = f"http://{host}:{port}/_stcore/health"

    try:
        response = requests.get(url, timeout=timeout)

        if response.status_code != 200:
            print(f"Dashboard health check failed: HTTP {response.status_code}", file=sys.stderr)
            return False

        print("Dashboard health check passed")
        return True

    except requests.exceptions.Timeout:
        print(f"Dashboard health check failed: timeout after {timeout}s", file=sys.stderr)
        return False
    except requests.exceptions.ConnectionError:
        print(f"Dashboard health check failed: cannot connect to {url}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Dashboard health check failed: {e}", file=sys.stderr)
        return False


def main() -> int:
    """
    Main health check function.

    Returns:
        0 if healthy, 1 if unhealthy
    """
    # Determine which service to check based on environment
    service_type = os.getenv("SERVICE_TYPE", "api")  # api or dashboard
    host = os.getenv("HEALTH_CHECK_HOST", "localhost")
    timeout = int(os.getenv("HEALTH_CHECK_TIMEOUT", "5"))

    if service_type == "dashboard":
        port = int(os.getenv("STREAMLIT_SERVER_PORT", "8501"))
        is_healthy = check_dashboard_health(host, port, timeout)
    else:  # api
        port = int(os.getenv("API_PORT", "8000"))
        is_healthy = check_api_health(host, port, timeout)

    return 0 if is_healthy else 1


if __name__ == "__main__":
    sys.exit(main())
