#!/usr/bin/env python3
"""
CNN Fear and Greed Index Web Scraper using zoom/slider method
Iteratively zooms into 2-month windows to extract daily data
"""

import json
import time
from datetime import datetime, timedelta
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
import pandas as pd


def extract_chart_data(page):
    """Extract current visible data from ECharts instance"""

    chart_data = page.evaluate("""
        () => {
            const chartDiv = document.getElementById('chart');
            if (!chartDiv) return { error: 'Chart not found' };

            let chartInstance = null;

            // Find ECharts instance
            for (let key in chartDiv) {
                if (key.indexOf('_echarts_instance_') !== -1) {
                    chartInstance = chartDiv[key];
                    break;
                }
            }

            if (!chartInstance && typeof echarts !== 'undefined') {
                chartInstance = echarts.getInstanceByDom(chartDiv);
            }

            if (!chartInstance) return { error: 'Instance not found' };

            const option = chartInstance.getOption();
            if (!option || !option.series || option.series.length === 0) {
                return { error: 'No series data' };
            }

            const series = option.series[0];
            const data = series.data || [];

            // Get current dataZoom setting to understand what's visible
            const dataZoom = option.dataZoom ? option.dataZoom[0] : null;

            return {
                success: true,
                data: data,
                dataZoom: dataZoom,
                totalPoints: data.length
            };
        }
    """)

    return chart_data


def set_zoom_range(page, start_percent, end_percent):
    """Set the zoom range using the slider at the bottom"""

    result = page.evaluate("""
        ([startPct, endPct]) => {
            const chartDiv = document.getElementById('chart');
            if (!chartDiv) return { error: 'Chart not found' };

            let chartInstance = null;

            for (let key in chartDiv) {
                if (key.indexOf('_echarts_instance_') !== -1) {
                    chartInstance = chartDiv[key];
                    break;
                }
            }

            if (!chartInstance && typeof echarts !== 'undefined') {
                chartInstance = echarts.getInstanceByDom(chartDiv);
            }

            if (!chartInstance) return { error: 'Instance not found' };

            // Set dataZoom to show only the specified range
            chartInstance.dispatchAction({
                type: 'dataZoom',
                start: startPct,
                end: endPct
            });

            return { success: true };
        }
    """, [start_percent, end_percent])

    time.sleep(1)  # Wait for zoom animation
    return result


def scrape_fng_with_zoom(url="https://www.finhacker.cz/en/fear-and-greed-index-historical-data-and-chart/"):
    """
    Scrape Fear and Greed Index data by iteratively zooming into 2-month windows

    Returns:
        pd.DataFrame: DataFrame with columns ['date', 'value']
    """

    with sync_playwright() as p:
        # Launch browser
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.set_viewport_size({"width": 1920, "height": 1200})

        print("Loading webpage...")
        page.goto(url, wait_until='networkidle')

        print("Waiting for chart to load...")
        page.wait_for_selector('#chart', timeout=30000)
        time.sleep(3)

        # First, get the full data to understand the date range
        print("Extracting full data...")
        full_data = extract_chart_data(page)

        if 'error' in full_data:
            print(f"Error: {full_data['error']}")
            browser.close()
            return None

        print(f"Total data points in chart: {full_data['totalPoints']}")

        # Process the full data to get date range
        all_data_points = []
        for point in full_data['data']:
            if isinstance(point, list) and len(point) >= 2:
                date_val = point[0]
                value = point[1]

                if isinstance(date_val, (int, float)):
                    date_str = datetime.fromtimestamp(date_val / 1000).strftime('%Y-%m-%d')
                else:
                    date_str = str(date_val)

                all_data_points.append({
                    'date': date_str,
                    'value': value,
                    'timestamp': date_val
                })

        if not all_data_points:
            print("No data points extracted from full view")
            browser.close()
            return None

        df_initial = pd.DataFrame(all_data_points)
        df_initial['date'] = pd.to_datetime(df_initial['date'])
        df_initial = df_initial.sort_values('date')

        print(f"\nInitial extraction: {len(df_initial)} points")
        print(f"Date range: {df_initial['date'].min()} to {df_initial['date'].max()}")

        # Now zoom into windows to get more granular data
        # Calculate 2-month windows with 1-month overlap
        all_detailed_data = []

        # We'll use percentage-based zooming
        # Each 2-month window, moving by 1 month (50% of the window)
        window_size = 8.33  # Approximately 2 months as percentage (2/24 years * 100)
        step_size = 4.17    # Approximately 1 month

        current_start = 0

        iteration = 0
        while current_start < 100:
            current_end = min(current_start + window_size, 100)

            print(f"\n--- Iteration {iteration + 1}: Zoom {current_start:.1f}% to {current_end:.1f}% ---")

            # Set zoom range
            zoom_result = set_zoom_range(page, current_start, current_end)

            if 'error' in zoom_result:
                print(f"Zoom error: {zoom_result['error']}")
                current_start += step_size
                iteration += 1
                continue

            # Extract data for this zoom level
            zoomed_data = extract_chart_data(page)

            if 'error' not in zoomed_data:
                data_points = zoomed_data.get('data', [])
                print(f"Extracted {len(data_points)} points in this window")

                for point in data_points:
                    if isinstance(point, list) and len(point) >= 2:
                        date_val = point[0]
                        value = point[1]

                        # Handle both string dates and numeric timestamps
                        if isinstance(date_val, (int, float)):
                            date_obj = datetime.fromtimestamp(date_val / 1000)
                            date_str = date_obj.strftime('%Y-%m-%d')
                        else:
                            date_str = str(date_val)

                        all_detailed_data.append({
                            'date': date_str,
                            'value': value,
                            'iteration': iteration,
                            'zoom_start': current_start,
                            'zoom_end': current_end
                        })

            current_start += step_size
            iteration += 1

        browser.close()

        print(f"\n{'='*60}")
        print(f"Extraction complete!")
        print(f"Total data points collected: {len(all_detailed_data)}")

        # Convert to DataFrame and remove duplicates
        if all_detailed_data:
            df = pd.DataFrame(all_detailed_data)
            df['date'] = pd.to_datetime(df['date'])

            # Remove duplicates, keeping the first occurrence
            df_unique = df.drop_duplicates(subset=['date'], keep='first')
            df_unique = df_unique.sort_values('date')

            print(f"Unique data points after deduplication: {len(df_unique)}")
            print(f"Date range: {df_unique['date'].min()} to {df_unique['date'].max()}")

            return df_unique[['date', 'value']]
        else:
            return None


def main():
    """Main execution function"""
    print("=" * 60)
    print("CNN Fear and Greed Index Scraper (Zoom Method)")
    print("=" * 60)

    df = scrape_fng_with_zoom()

    if df is not None and len(df) > 0:
        # Save to CSV
        output_file = 'data/fng_scraped_zoom.csv'
        df.to_csv(output_file, index=False)
        print(f"\n✓ Data saved to {output_file}")
        print(f"✓ Total unique records: {len(df)}")
        print(f"✓ Date range: {df['date'].min()} to {df['date'].max()}")

        print(f"\nPreview of data:")
        print(df.head(20))
        print("\n...")
        print(df.tail(20))

        # Basic statistics
        print(f"\nBasic Statistics:")
        print(df['value'].describe())

        # Check for date gaps
        df_sorted = df.sort_values('date')
        date_diffs = df_sorted['date'].diff()
        max_gap = date_diffs.max()
        print(f"\nLargest gap between consecutive dates: {max_gap}")

    else:
        print("\n✗ No data extracted")


if __name__ == "__main__":
    main()
