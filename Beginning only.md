```python
# Setup and Configuration


```python
# Import necessary libraries
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime, timedelta

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Environment checks
print(f"Python executable: {sys.executable}")  
print("Current working directory:", os.getcwd())
print("\nFiles in data directory:", os.listdir('data'))
```

    Python executable: C:\Users\josep\miniconda3\envs\CBH_CS2\python.exe
    Current working directory: C:\Users\josep\Documents\programming_projects\GitHub\CBH_CS2
    
    Files in data directory: ['booking_logs_large.csv', 'Booking_logs_sample_1000 - Sheet1.csv', 'Booking_logs_sample_100_rows.csv', 'Booking_logs_sample_header.csv', 'cancel_logs_large.csv', 'Cancel_logs_sample_1000 - Sheet1.csv', 'Cancel_logs_sample_100_rows.csv', 'Cancel_logs_sample_header.csv', 'cleveland_shifts_large.csv', 'Cleveland_shifts_Sample_1000 - Sheet1.csv', 'Cleveland_shifts_Sample_100_rows.csv', 'Cleveland_shifts_Sample_header.csv']
    

# Helper Functions and Classes


```python
# Helper Functions for Data Loading and Cleaning
# update: now takes varied date ranges into account in loading
def load_and_clean_shifts(df):
    """
    Load and clean shifts dataset
    
    Parameters:
        df (pd.DataFrame): Raw shifts dataframe
        
    Returns:
        pd.DataFrame: Cleaned shifts dataframe with proper datatypes
        
    Notes:
        - Makes a copy to avoid modifying original data
        - Converts datetime columns
        - Handles potential errors in datetime conversion
    """
    df = df.copy()
    
    # Convert datetime columns with error handling
    datetime_cols = ['Start', 'End', 'Created At']
    for col in datetime_cols:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], format='mixed')
            except Exception as e:
                print(f"Error converting {col} to datetime: {str(e)}")
                # Log problematic rows for investigation
                problematic_rows = df[pd.to_datetime(df[col], format='mixed', errors='coerce').isna()]
                if not problematic_rows.empty:
                    print(f"Problematic rows in {col}:")
                    print(problematic_rows[col].head())
    
    return df

def load_and_clean_bookings(df):
    """
    Load and clean booking logs dataset
    
    Parameters:
        df (pd.DataFrame): Raw bookings dataframe
        
    Returns:
        pd.DataFrame: Cleaned bookings dataframe with proper datatypes
    """
    df = df.copy()
    # Convert datetime columns
    try:
        df['Created At'] = pd.to_datetime(df['Created At'])
    except Exception as e:
        print(f"Error converting Created At: {str(e)}")
    return df

def load_and_clean_cancellations(df):
    """
    Load and clean cancellation logs dataset
    
    Parameters:
        df (pd.DataFrame): Raw cancellations dataframe
        
    Returns:
        pd.DataFrame: Cleaned cancellations dataframe with proper datatypes
    """
    df = df.copy()
    # Convert datetime columns with flexible parsing
    try:
        df['Created At'] = pd.to_datetime(df['Created At'], format='mixed')
        df['Shift Start Logs'] = pd.to_datetime(df['Shift Start Logs'], format='mixed')
    except Exception as e:
        print(f"Error in datetime conversion: {str(e)}")
        # Try to identify problematic rows
        prob_rows = df[pd.to_datetime(df['Shift Start Logs'], format='mixed', errors='coerce').isna()]
        if not prob_rows.empty:
            print("\nSample of problematic date formats:")
            print(prob_rows['Shift Start Logs'].head())
    
    return df

def categorize_lead_time(hours):
    """
    Categorize lead times based on business rules.
    
    Parameters:
        hours (float): Lead time in hours
        
    Returns:
        str: Category of lead time
    """
    if hours < 0:
        return 'No-Show'  # Cancelled after shift start
    elif hours < 4:
        return 'Late (<4hrs)'
    elif hours < 24:
        return 'Same Day'
    elif hours < 72:
        return 'Advance (<3 days)'
    return 'Early (3+ days)'

def clean_lead_times(cancellations_df):
    """
    Clean and categorize lead times in cancellation data
    
    Parameters:
        cancellations_df (pd.DataFrame): Raw cancellations dataframe
    
    Returns:
        pd.DataFrame: Cleaned cancellations data with categorized lead times
        pd.Series: Statistics about removed records for quality control
    """
    df = cancellations_df.copy()
    
    # Track data quality issues
    quality_stats = {
        'original_rows': len(df),
        'null_lead_times': df['Lead Time'].isnull().sum(),
        'infinite_values': (~np.isfinite(df['Lead Time'])).sum()
    }
    
    # Only remove truly invalid data
    mask = df['Lead Time'].notnull() & np.isfinite(df['Lead Time'])
    df = df[mask]
    
    # Add cleaned lead time without filtering extremes
    df['clean_lead_time'] = df['Lead Time']
    
    # Categorize all lead times
    df['cancellation_category'] = df['clean_lead_time'].apply(categorize_lead_time)
    
    # Add flags for extreme values for analysis
    df['is_extreme_negative'] = df['Lead Time'] < -72  # Flag cancellations >3 days after
    df['is_extreme_positive'] = df['Lead Time'] > 1000 # Flag cancellations >41 days before
    
    quality_stats['final_rows'] = len(df)
    quality_stats['removed_rows'] = quality_stats['original_rows'] - quality_stats['final_rows']
    
    return df, pd.Series(quality_stats)

# Data Summary Storage Class
class DataSummary:
    """Class to store and manage analysis results"""
    def __init__(self):
        self.summaries = {}
    
    def add_summary(self, dataset_name, summary_type, data):
        """Add summary statistics to storage"""
        if dataset_name not in self.summaries:
            self.summaries[dataset_name] = {}
        self.summaries[dataset_name][summary_type] = data
    
    def get_summary(self, dataset_name, summary_type=None):
        """Retrieve stored summary statistics"""
        if summary_type:
            return self.summaries.get(dataset_name, {}).get(summary_type)
        return self.summaries.get(dataset_name)
    
    def print_summary(self, dataset_name):
        """Print stored summaries for a dataset"""
        if dataset_name in self.summaries:
            print(f"\nSummary for {dataset_name}:")
            for summary_type, data in self.summaries[dataset_name].items():
                print(f"\n{summary_type}:")
                print(data)

# Initialize summary storage
summary = DataSummary()
```


```python
# Summary explorer helper functions 
def explore_summary(summary, indent=0, max_display_length=100):
    """
    Recursively explore and display the contents of the DataSummary object.
    
    Parameters:
        summary: The DataSummary object or a nested dictionary/value to explore
        indent: Current indentation level (default: 0)
        max_display_length: Maximum length for displayed values (default: 100)
    """
    def format_value(value):
        """Format a value for display, truncating if too long"""
        str_value = str(value)
        if len(str_value) > max_display_length:
            return str_value[:max_display_length] + '...'
        return str_value
    
    def print_indented(text, indent):
        """Print text with proper indentation"""
        print('    ' * indent + text)

    if isinstance(summary, DataSummary):
        # If we're starting with a DataSummary object, explore its summaries
        print("\n=== Complete Summary Contents ===\n")
        explore_summary(summary.summaries, indent)
    
    elif isinstance(summary, dict):
        # Recursively explore dictionary contents
        for key, value in summary.items():
            if isinstance(value, dict):
                print_indented(f"{key}:", indent)
                explore_summary(value, indent + 1)
            elif isinstance(value, pd.Series) or isinstance(value, pd.DataFrame):
                print_indented(f"{key}: [pandas {type(value).__name__}]", indent)
                str_repr = str(value)
                for line in str_repr.split('\n')[:5]:  # Show first 5 lines
                    print_indented(line, indent + 1)
                if len(str_repr.split('\n')) > 5:
                    print_indented('...', indent + 1)
            else:
                print_indented(f"{key}: {format_value(value)}", indent)
    
    else:
        # Base case: print the value
        print_indented(format_value(summary), indent)

def get_summary_structure(summary):
    """
    Print just the structure of the summary without all the data values.
    
    Parameters:
        summary: The DataSummary object
    """
    print("\n=== Summary Structure ===\n")
    for dataset_name in summary.summaries:
        print(f"\nDataset: {dataset_name}")
        for summary_type in summary.summaries[dataset_name]:
            print(f"  └── {summary_type}")

# Example usage:
print("First, let's see the overall structure of your summary:")
get_summary_structure(summary)

print("\nWould you like to see the complete contents of any specific dataset?")
print("You can view them using:")
print("explore_summary(summary.get_summary('dataset_name'))")

# To view everything:
print("\nOr view all contents with:")
print("explore_summary(summary)")
```

    First, let's see the overall structure of your summary:
    
    === Summary Structure ===
    
    
    Dataset: data_quality
      └── audit_results
      └── coverage_analysis
      └── missing_data_impact
    
    Dataset: bookings
      └── time_to_fill
      └── role_patterns
      └── rebooking_stats
      └── shape
      └── dtypes
      └── missing_values
      └── date_range
      └── unique_ids
    
    Dataset: economic
      └── overall_impact
      └── role_impact
      └── type_impact
    
    Dataset: complete_analysis
      └── shifts
      └── behavior
      └── missing_agents
    
    Dataset: cancellations
      └── quality_stats
      └── extreme_values
      └── action_types
      └── role_patterns
      └── shift_patterns
      └── role_impact
      └── shape
      └── dtypes
      └── missing_values
      └── date_range
      └── unique_ids
    
    Dataset: data_filtering
      └── overlap_period
    
    Dataset: shifts
      └── shape
      └── dtypes
      └── missing_values
      └── numeric_stats
      └── shift_types
      └── agent_types
      └── hour_distribution
      └── day_distribution
      └── facility_stats
      └── date_range
      └── unique_ids
    
    Dataset: cross_validation
      └── id_overlaps
    
    Would you like to see the complete contents of any specific dataset?
    You can view them using:
    explore_summary(summary.get_summary('dataset_name'))
    
    Or view all contents with:
    explore_summary(summary)
    

# Initial Data Loading and Validation


```python
# === Load and Prepare All Datasets ===
print("Loading and preparing all datasets...")

# Load all datasets
shifts_df = pd.read_csv('data/cleveland_shifts_large.csv')
bookings_df = pd.read_csv('data/booking_logs_large.csv')
cancellations_df = pd.read_csv('data/cancel_logs_large.csv')

def get_overlapping_date_range(shifts_df, bookings_df, cancellations_df):
    """
    Determine the overlapping date range across all three datasets.
    Returns the start and end dates that represent the period where we have complete data.
    
    The overlapping range is determined by:
    - Latest start date among all datasets (to ensure we have data from all sources)
    - Earliest end date among all datasets (to ensure we don't exceed any dataset's range)
    """
    # Get date ranges for each dataset
    shifts_range = {
        'start': shifts_df['Created At'].min(),
        'end': shifts_df['Created At'].max()
    }
    bookings_range = {
        'start': bookings_df['Created At'].min(),
        'end': bookings_df['Created At'].max()
    }
    cancellations_range = {
        'start': cancellations_df['Created At'].min(),
        'end': cancellations_df['Created At'].max()
    }
    
    # Find overlapping range
    overlap_start = max(
        shifts_range['start'],
        bookings_range['start'],
        cancellations_range['start']
    )
    
    overlap_end = min(
        shifts_range['end'],
        bookings_range['end'],
        cancellations_range['end']
    )
    
    return overlap_start, overlap_end

def filter_to_overlap_period(df, start_date, end_date):
    """
    Filter a dataframe to only include rows within the overlapping date range.
    """
    return df[
        (df['Created At'] >= start_date) & 
        (df['Created At'] <= end_date)
    ]

# After your existing data loading code, add:
# Find overlapping period
overlap_start, overlap_end = get_overlapping_date_range(shifts_df, bookings_df, cancellations_df)

# Filter all datasets to overlapping period
shifts_df = filter_to_overlap_period(shifts_df, overlap_start, overlap_end)
bookings_df = filter_to_overlap_period(bookings_df, overlap_start, overlap_end)
cancellations_df = filter_to_overlap_period(cancellations_df, overlap_start, overlap_end)

# Print information about the filtering
print("\n=== Data Filtering Summary ===")
print(f"Analysis Period: {overlap_start} to {overlap_end}")
print("\nDataset sizes after filtering:")
print(f"Shifts: {len(shifts_df):,} records")
print(f"Bookings: {len(bookings_df):,} records")
print(f"Cancellations: {len(cancellations_df):,} records")

# Store filtering info in summary
summary.add_summary('data_filtering', 'overlap_period', {
    'start': overlap_start,
    'end': overlap_end,
    'original_sizes': {
        'shifts': len(shifts_df),
        'bookings': len(bookings_df),
        'cancellations': len(cancellations_df)
    }
})
```

    Loading and preparing all datasets...
    
    === Data Filtering Summary ===
    Analysis Period: 2021-09-06 11:06:36 to 2022-04-04 19:50:32
    
    Dataset sizes after filtering:
    Shifts: 40,989 records
    Bookings: 126,849 records
    Cancellations: 78,056 records
    


```python
# Clean and prepare the data
shifts_df = load_and_clean_shifts(shifts_df)
bookings_df = load_and_clean_bookings(bookings_df)
cancellations_df = load_and_clean_cancellations(cancellations_df)

# Function to analyze and summarize a dataset
def analyze_dataset(df, dataset_name, summary):
    print(f"\n=== {dataset_name} Data Overview ===")
    print("Dataset Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nData Types:\n", df.dtypes)
    
    # Missing value analysis
    missing_values = df.isnull().sum()
    print("\nMissing Values:\n", missing_values)
    
    # Display sample data
    print("\nFirst few rows:")
    print(df.head())
    
    # Store findings in summary
    summary.add_summary(dataset_name.lower(), 'shape', df.shape)
    summary.add_summary(dataset_name.lower(), 'dtypes', df.dtypes)
    summary.add_summary(dataset_name.lower(), 'missing_values', missing_values)
    
    # Additional temporal analysis
    date_range = {
        'start_date': pd.to_datetime(df['Created At']).min(),
        'end_date': pd.to_datetime(df['Created At']).max(),
        'total_days': (pd.to_datetime(df['Created At']).max() - pd.to_datetime(df['Created At']).min()).days
    }
    summary.add_summary(dataset_name.lower(), 'date_range', date_range)
    
    # Record unique IDs count
    unique_ids = df['ID'].nunique()
    summary.add_summary(dataset_name.lower(), 'unique_ids', unique_ids)
    
    return date_range, unique_ids

# Analyze each dataset
datasets = {
    'Shifts': shifts_df,
    'Bookings': bookings_df,
    'Cancellations': cancellations_df
}

print("\n=== Dataset Analysis ===")
for name, df in datasets.items():
    date_range, unique_ids = analyze_dataset(df, name, summary)
    print(f"\n{name} Dataset Summary:")
    print(f"Date Range: {date_range['start_date']} to {date_range['end_date']} ({date_range['total_days']} days)")
    print(f"Unique IDs: {unique_ids}")

# Cross-dataset validation
print("\n=== Cross-Dataset Validation ===")
for name, df in datasets.items():
    print(f"\n{name} Dataset:")
    print(f"Total Records: {len(df)}")
    print(f"Records per Day: {len(df) / (pd.to_datetime(df['Created At']).max() - pd.to_datetime(df['Created At']).min()).days:.2f}")

# Check for overlapping IDs between datasets
print("\n=== ID Overlap Analysis ===")
shifts_ids = set(shifts_df['ID'])
bookings_ids = set(bookings_df['ID'])
cancellations_ids = set(cancellations_df['ID'])

overlap_analysis = {
    'shifts_bookings': len(shifts_ids.intersection(bookings_ids)),
    'shifts_cancellations': len(shifts_ids.intersection(cancellations_ids)),
    'bookings_cancellations': len(bookings_ids.intersection(cancellations_ids))
}

summary.add_summary('cross_validation', 'id_overlaps', overlap_analysis)

print("ID Overlaps:")
print(f"Shifts-Bookings: {overlap_analysis['shifts_bookings']}")
print(f"Shifts-Cancellations: {overlap_analysis['shifts_cancellations']}")
print(f"Bookings-Cancellations: {overlap_analysis['bookings_cancellations']}")
```

    
    === Dataset Analysis ===
    
    === Shifts Data Overview ===
    Dataset Shape: (40989, 16)
    
    Columns: ['ID', 'Agent ID', 'Facility ID', 'Start', 'Agent Req', 'End', 'Deleted', 'Shift Type', 'Created At', 'Verified', 'Charge', 'Time', 'Hour', 'Day', 'Month', 'Shift_Length']
    
    Data Types:
     ID                      object
    Agent ID                object
    Facility ID             object
    Start           datetime64[ns]
    Agent Req               object
    End             datetime64[ns]
    Deleted                 object
    Shift Type              object
    Created At      datetime64[ns]
    Verified                  bool
    Charge                 float64
    Time                   float64
    Hour                     int32
    Day                     object
    Month                    int32
    Shift_Length           float64
    dtype: object
    
    Missing Values:
     ID                  0
    Agent ID        20010
    Facility ID         0
    Start               0
    Agent Req           0
    End                 0
    Deleted         27028
    Shift Type          0
    Created At          0
    Verified            0
    Charge              0
    Time                0
    Hour                0
    Day                 0
    Month               0
    Shift_Length        0
    dtype: int64
    
    First few rows:
                             ID                  Agent ID  \
    0  61732a2ad690c401690cf273  614627661afb050166fecd99   
    1  61732a4fd690c401690cf307  60d5f4c8a9b88a0166aedaca   
    2  61732ab7d6e633016acd05e2  5d7fb6319b671100167be1f1   
    3  61732c33d6e633016acd11fc  613503b78b28c60166060efe   
    4  61759081801438016ae23301  6099c93f2a957601669549c3   
    
                    Facility ID               Start Agent Req                 End  \
    0  5f9c169622c5c50016d5ba32 2021-10-27 23:00:00       LVN 2021-10-28 03:00:00   
    1  5f9c169622c5c50016d5ba32 2021-10-28 11:00:00       CNA 2021-10-28 19:00:00   
    2  5f9c169622c5c50016d5ba32 2021-10-30 19:00:00       LVN 2021-10-31 03:00:00   
    3  5e6bd68c8d921f0016325d09 2021-10-25 18:30:00       CNA 2021-10-26 03:00:00   
    4  61704cc1f19e4d0169ec4ac5 2021-11-03 02:00:00       CNA 2021-11-03 10:00:00   
    
      Deleted Shift Type          Created At  Verified  Charge  Time  Hour  \
    0     NaN         pm 2021-10-22 21:16:26      True  43.000 4.170    23   
    1     NaN         am 2021-10-22 21:17:04      True  25.000 5.000    11   
    2     NaN         pm 2021-10-22 21:18:48      True  64.750 8.000    19   
    3     NaN         pm 2021-10-22 21:25:08      True  28.000 8.440    18   
    4     NaN        noc 2021-10-24 16:57:37      True  30.000 7.500     2   
    
             Day  Month  Shift_Length  
    0  Wednesday     10         4.000  
    1   Thursday     10         8.000  
    2   Saturday     10         8.000  
    3     Monday     10         8.500  
    4  Wednesday     11         8.000  
    
    Shifts Dataset Summary:
    Date Range: 2021-09-07 01:25:57 to 2022-04-04 19:50:32 (209 days)
    Unique IDs: 40989
    
    === Bookings Data Overview ===
    Dataset Shape: (126849, 7)
    
    Columns: ['ID', 'Created At', 'Shift ID', 'Action', 'Worker ID', 'Facility ID', 'Lead Time']
    
    Data Types:
     ID                     object
    Created At     datetime64[ns]
    Shift ID               object
    Action                 object
    Worker ID              object
    Facility ID            object
    Lead Time             float64
    dtype: object
    
    Missing Values:
     ID               0
    Created At       0
    Shift ID         0
    Action           0
    Worker ID      140
    Facility ID      0
    Lead Time        0
    dtype: int64
    
    First few rows:
                             ID          Created At                  Shift ID  \
    0  615f58b997538d018b1163e0 2021-10-07 20:29:46  615e1de54502b9016c9a5af1   
    1  61608ce36790e5016acaf149 2021-10-08 18:24:35  616060338917ec016965f9a0   
    2  61677eea7dce1d016a9b7f95 2021-10-14 00:50:51  61645d1dfdd396016a697466   
    3  6166159a8186c5016a54dd6d 2021-10-12 23:09:15  616093710daae0016926f60d   
    4  616391cec03fa3016a12a316 2021-10-11 01:22:23  614b6df64cd99701667bb099   
    
            Action                 Worker ID               Facility ID  Lead Time  
    0  SHIFT_CLAIM  5c993c695d096c00167e6845  615b46a1c7135401876cdd06    144.504  
    1  SHIFT_CLAIM  61587d5a2ced390187554556  5fbd2a01582f640016a4cb09    200.590  
    2  SHIFT_CLAIM  60898a05cd5d9c016675a996  5c3cc2e3a2ae5c0016bef82c     73.653  
    3  SHIFT_CLAIM  60ba35153c4b4f016610c5cd  60c1495470d2440166e23cb0    118.846  
    4  SHIFT_CLAIM  60de429441198701669b529a  614900f003c26b0166e08737    178.627  
    
    Bookings Dataset Summary:
    Date Range: 2021-09-06 01:19:57 to 2022-02-18 20:29:52 (165 days)
    Unique IDs: 126849
    
    === Cancellations Data Overview ===
    Dataset Shape: (78056, 8)
    
    Columns: ['ID', 'Created At', 'Shift ID', 'Action', 'Worker ID', 'Shift Start Logs', 'Facility ID', 'Lead Time']
    
    Data Types:
     ID                          object
    Created At          datetime64[ns]
    Shift ID                    object
    Action                      object
    Worker ID                   object
    Shift Start Logs    datetime64[ns]
    Facility ID                 object
    Lead Time                  float64
    dtype: object
    
    Missing Values:
     ID                    0
    Created At            0
    Shift ID              0
    Action                0
    Worker ID           191
    Shift Start Logs      0
    Facility ID           0
    Lead Time             0
    dtype: int64
    
    First few rows:
                             ID          Created At                  Shift ID  \
    0  61a07227f36d1e0186381d10 2021-11-26 05:35:36  619d2f2a5db209018533a507   
    1  617bb2e2ae50230185b05985 2021-10-29 08:37:55  614fa1ed22aa37018320e6ed   
    2  61c90bb026c4c8018adc0820 2021-12-27 00:41:20  61ba75ff55eb55018586568b   
    3  6186ce5dba1046018596162c 2021-11-06 18:50:06  6169aebc4dcbb2016aad106c   
    4  618987925daf8001857ffd84 2021-11-08 20:24:50  6179b80d9c79750169c98f54   
    
                Action                 Worker ID    Shift Start Logs  \
    0  NO_CALL_NO_SHOW  5e1fa8d8170f34001633e511 2021-11-25 06:00:00   
    1    WORKER_CANCEL  5cf573381648900016c41377 2021-10-29 13:00:00   
    2    WORKER_CANCEL  61c24b6132278b018511e941 2021-12-27 23:45:00   
    3    WORKER_CANCEL  60f9a88678a10501661b36d0 2021-11-13 12:00:00   
    4    WORKER_CANCEL  615d413f21f03c016c7523ff 2021-11-23 20:00:00   
    
                    Facility ID  Lead Time  
    0  6182c3fb79773801854c081d    -23.593  
    1  5ff4f626909f7a00160d06fd      4.368  
    2  5f9888997f5dee0016f777d4     23.061  
    3  5f9b189a7ecb880016516a52    161.165  
    4  5f9ad22ae3a95f0016090f97    359.586  
    
    Cancellations Dataset Summary:
    Date Range: 2021-09-06 11:06:36 to 2022-04-01 19:06:44 (207 days)
    Unique IDs: 78056
    
    === Cross-Dataset Validation ===
    
    Shifts Dataset:
    Total Records: 40989
    Records per Day: 196.12
    
    Bookings Dataset:
    Total Records: 126849
    Records per Day: 768.78
    
    Cancellations Dataset:
    Total Records: 78056
    Records per Day: 377.08
    
    === ID Overlap Analysis ===
    ID Overlaps:
    Shifts-Bookings: 0
    Shifts-Cancellations: 0
    Bookings-Cancellations: 0
    


```python

```

```
