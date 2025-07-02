import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "data-team",
    "depends_on_past": False,
    "start_date": datetime(2025, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

CITIES = {
    # Found using https://www.latlong.net/
    "Paris": {"latitude": 48.85, "longitude": 2.35},
    "Monteriggioni": {"latitude": 43.38, "longitude": 11.22},
    "Moscow": {"latitude": 55.75, "longitude": 37.61},
}

DATA_DIR = Path("/opt/airflow/data")
CSV_FILE = DATA_DIR / "weather_data.csv"


def extract_weather_data(**context):
    """Extract weather data from Open-Meteo API for multiple cities."""
    weather_data = []

    for city_name, coords in CITIES.items():
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={coords['latitude']}"
            f"&longitude={coords['longitude']}"
            f"&current_weather=true"
        )

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            current_weather = data.get("current_weather", {})
            weather_record = {
                "city": city_name,
                "temperature": current_weather.get("temperature"),
                "windspeed": current_weather.get("windspeed"),
                "weather_code": current_weather.get("weathercode"),
                "timestamp": current_weather.get("time"),
            }
            weather_data.append(weather_record)

        except requests.RequestException as e:
            print(f"Error fetching data for {city_name}: {e}")
            raise

    return weather_data


def transform_weather_data(**context):
    """Transform and normalize the weather data."""
    weather_data = context["task_instance"].xcom_pull(task_ids="extract_weather")

    if not weather_data:
        raise ValueError("No weather data received from extract task")

    df = pd.DataFrame(weather_data)

    df["processed_at"] = datetime.utcnow().isoformat()

    df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
    df["windspeed"] = pd.to_numeric(df["windspeed"], errors="coerce")
    df["weather_code"] = pd.to_numeric(df["weather_code"], errors="coerce")

    return df.to_dict("records")


def load_weather_data(**context):
    """Load weather data to CSV file with idempotency."""
    transformed_data = context["task_instance"].xcom_pull(task_ids="transform_weather")

    if not transformed_data:
        print("No transformed data to load")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    new_df = pd.DataFrame(transformed_data)

    if CSV_FILE.exists():
        existing_df = pd.read_csv(CSV_FILE)

        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(
            subset=["city", "timestamp"], keep="last"
        )
    else:
        combined_df = new_df

    combined_df.to_csv(CSV_FILE, index=False)
    print(f"Weather data saved to {CSV_FILE}")
    print(f"Total records: {len(combined_df)}")


dag = DAG(
    "daily_weather_pipeline",
    default_args=default_args,
    description="Daily Weather Data ETL Pipeline",
    schedule="0 8 * * *",  # Daily at 8 AM UTC
    catchup=False,
    tags=["weather", "etl", "daily"],
)

extract_task = PythonOperator(
    task_id="extract_weather",
    python_callable=extract_weather_data,
    dag=dag,
)

transform_task = PythonOperator(
    task_id="transform_weather",
    python_callable=transform_weather_data,
    dag=dag,
)

load_task = PythonOperator(
    task_id="load_weather",
    python_callable=load_weather_data,
    dag=dag,
)

extract_task >> transform_task >> load_task
