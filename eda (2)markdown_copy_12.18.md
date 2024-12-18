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
from datetime import datetime

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

# Initial Data Loading and Validation


```python
# to do note this only previews shifts data not cancels and bookings 
# we should add other datasets to summary 
# check how unique ids is defined 
# note the date ranges for each dataset are different 
# 
# 41k unique shifts, 127k unique bookings, 78k unique cancels -> 41 + 78 ~ 120 seems reasonable but lots of 
# this means each shift is booked 3 times and canceled twice? 

# === Load and Prepare All Datasets ===
print("Loading and preparing all datasets...")

# Load all datasets
shifts_df = pd.read_csv('data/cleveland_shifts_large.csv')
bookings_df = pd.read_csv('data/booking_logs_large.csv')
cancellations_df = pd.read_csv('data/cancel_logs_large.csv')

# Clean and prepare the data
shifts_df = load_and_clean_shifts(shifts_df)
bookings_df = load_and_clean_bookings(bookings_df)
cancellations_df = load_and_clean_cancellations(cancellations_df)

# Initial shifts data exploration
print("\n=== Shifts Data Overview ===")
print("Dataset Shape:", shifts_df.shape)
print("\nColumns:", shifts_df.columns.tolist())
print("\nData Types:\n", shifts_df.dtypes)

# Missing value analysis
missing_values = shifts_df.isnull().sum()
print("\nMissing Values:\n", missing_values)

# Display sample data
print("\nFirst few rows:")
print(shifts_df.head())

# Store initial findings
summary.add_summary('shifts', 'shape', shifts_df.shape)
summary.add_summary('shifts', 'dtypes', shifts_df.dtypes)
summary.add_summary('shifts', 'missing_values', missing_values)

# Cross-dataset validation
print("\n=== Dataset Cross-Validation ===")
for name, df in [('Shifts', shifts_df), ('Bookings', bookings_df), ('Cancellations', cancellations_df)]:
    print(f"\n=== {name} Dataset ===")
    print(f"Shape: {df.shape}")
    print(f"Date Range: {pd.to_datetime(df['Created At']).min()} to {pd.to_datetime(df['Created At']).max()}")
    print("\nSample of unique IDs:")
    print(df['ID'].nunique())
```

    Loading and preparing all datasets...
    
    === Shifts Data Overview ===
    Dataset Shape: (41040, 12)
    
    Columns: ['ID', 'Agent ID', 'Facility ID', 'Start', 'Agent Req', 'End', 'Deleted', 'Shift Type', 'Created At', 'Verified', 'Charge', 'Time']
    
    Data Types:
     ID                     object
    Agent ID               object
    Facility ID            object
    Start          datetime64[ns]
    Agent Req              object
    End            datetime64[ns]
    Deleted                object
    Shift Type             object
    Created At     datetime64[ns]
    Verified                 bool
    Charge                float64
    Time                  float64
    dtype: object
    
    Missing Values:
     ID                 0
    Agent ID       20035
    Facility ID        0
    Start              0
    Agent Req          0
    End                0
    Deleted        27058
    Shift Type         0
    Created At         0
    Verified           0
    Charge             0
    Time               0
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
    
      Deleted Shift Type          Created At  Verified  Charge  Time  
    0     NaN         pm 2021-10-22 21:16:26      True  43.000 4.170  
    1     NaN         am 2021-10-22 21:17:04      True  25.000 5.000  
    2     NaN         pm 2021-10-22 21:18:48      True  64.750 8.000  
    3     NaN         pm 2021-10-22 21:25:08      True  28.000 8.440  
    4     NaN        noc 2021-10-24 16:57:37      True  30.000 7.500  
    
    === Dataset Cross-Validation ===
    
    === Shifts Dataset ===
    Shape: (41040, 12)
    Date Range: 2021-08-16 18:09:54 to 2022-04-04 19:50:32
    
    Sample of unique IDs:
    41040
    
    === Bookings Dataset ===
    Shape: (127005, 7)
    Date Range: 2021-08-13 07:34:17 to 2022-04-12 22:16:25
    
    Sample of unique IDs:
    127005
    
    === Cancellations Dataset ===
    Shape: (78073, 8)
    Date Range: 2021-09-06 11:06:36 to 2022-07-09 05:05:31
    
    Sample of unique IDs:
    78073
    

# Data Dive (formerly Data Quality Analysis) 


```python
def check_duplicates(df, dataset_name):
    """
    Check for duplicate IDs in a DataFrame
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame to check for duplicates
    dataset_name : str
        Name of the dataset for reporting
    
    Returns:
    --------
    dict : A dictionary with duplicate analysis results
    """
    # Identify the ID column (adjust if different in your specific datasets)
    id_column = 'ID'
    
    # Count total rows
    total_rows = len(df)
    
    # Count unique IDs
    unique_ids = df[id_column].nunique()
    
    # Find duplicate IDs
    duplicate_ids = df[df.duplicated(subset=[id_column], keep=False)]
    num_duplicate_ids = len(duplicate_ids)
    
    # Group duplicates to see how many times each duplicate appears
    duplicate_counts = duplicate_ids[id_column].value_counts()
    
    # Print detailed results
    print(f"\n=== Duplicate Analysis for {dataset_name} ===")
    print(f"Total rows: {total_rows}")
    print(f"Unique IDs: {unique_ids}")
    print(f"Number of duplicate IDs: {num_duplicate_ids}")
    
    # If there are duplicates, show more details
    if num_duplicate_ids > 0:
        print("\nDuplicate ID Frequency:")
        print(duplicate_counts.head())  # Show top duplicates
        
        print("\nSample of rows with duplicate IDs:")
        print(duplicate_ids.groupby(id_column).first())
    
    return {
        'total_rows': total_rows,
        'unique_ids': unique_ids,
        'num_duplicate_ids': num_duplicate_ids
    }

# Apply the function to each dataset
shifts_duplicate_analysis = check_duplicates(shifts_df, 'Shifts Dataset')
bookings_duplicate_analysis = check_duplicates(booking_logs_df, 'Bookings Dataset')
cancellations_duplicate_analysis = check_duplicates(cancel_logs_df, 'Cancellations Dataset')
```

    
    === Duplicate Analysis for Shifts Dataset ===
    Total rows: 41040
    Unique IDs: 41040
    Number of duplicate IDs: 0
    


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[108], line 54
         52 # Apply the function to each dataset
         53 shifts_duplicate_analysis = check_duplicates(shifts_df, 'Shifts Dataset')
    ---> 54 bookings_duplicate_analysis = check_duplicates(booking_logs_df, 'Bookings Dataset')
         55 cancellations_duplicate_analysis = check_duplicates(cancel_logs_df, 'Cancellations Dataset')
    

    NameError: name 'booking_logs_df' is not defined



```python
# === Lead Time Analysis ===
print("Analyzing lead times and cancellation patterns...")

# Clean lead times and get quality stats
clean_cancellations, quality_stats = clean_lead_times(cancellations_df)

# Basic cancellation metrics 
# to do - this may be incorrect due to date overlap 
shifts_with_cancellations = len(set(shifts_df['ID']) & set(cancellations_df['Shift ID']))
print(f"Shifts with cancellations: {shifts_with_cancellations}")
print(f"Percentage of shifts cancelled: {(shifts_with_cancellations/len(shifts_df))*100:.2f}%")

print("\n=== Data Quality Statistics ===")
print(quality_stats)

print("\n=== Lead Time Distribution ===")
print("Overall Lead Time Statistics:")
print(cancellations_df['Lead Time'].describe().round(2))

print("\n=== Cancellation Categories ===")
print(clean_cancellations['cancellation_category'].value_counts().sort_index())

print("\n=== Extreme Values Analysis ===")
print(f"Very Late Cancellations (>3 days after): {clean_cancellations['is_extreme_negative'].sum()}")
print(f"Very Early Cancellations (>41 days before): {clean_cancellations['is_extreme_positive'].sum()}")

# Distribution of lead times for extreme cases
if clean_cancellations['is_extreme_negative'].any():
    print("\nVery Late Cancellation Stats:")
    print(clean_cancellations[clean_cancellations['is_extreme_negative']]['Lead Time'].describe())

if clean_cancellations['is_extreme_positive'].any():
    print("\nVery Early Cancellation Stats:")
    print(clean_cancellations[clean_cancellations['is_extreme_positive']]['Lead Time'].describe())

# Store results in summary
summary.add_summary('cancellations', 'quality_stats', quality_stats.to_dict())
summary.add_summary('cancellations', 'extreme_values', {
    'very_late': clean_cancellations['is_extreme_negative'].sum(),
    'very_early': clean_cancellations['is_extreme_positive'].sum()
})
```

    Analyzing lead times and cancellation patterns...
    Shifts with cancellations: 6535
    Percentage of shifts cancelled: 15.92%
    
    === Data Quality Statistics ===
    original_rows      78073
    null_lead_times        0
    infinite_values        0
    final_rows         78073
    removed_rows           0
    dtype: int64
    
    === Lead Time Distribution ===
    Overall Lead Time Statistics:
    count   78073.000
    mean      105.910
    std       215.990
    min     -4210.100
    25%         2.250
    50%        36.300
    75%       200.660
    max      1854.240
    Name: Lead Time, dtype: float64
    
    === Cancellation Categories ===
    cancellation_category
    Advance (<3 days)     7933
    Early (3+ days)      33853
    Late (<4hrs)         11102
    No-Show              11966
    Same Day             13219
    Name: count, dtype: int64
    
    === Extreme Values Analysis ===
    Very Late Cancellations (>3 days after): 4960
    Very Early Cancellations (>41 days before): 135
    
    Very Late Cancellation Stats:
    count    4960.000
    mean     -277.947
    std       338.618
    min     -4210.100
    25%      -289.484
    50%      -160.323
    75%      -104.718
    max       -72.018
    Name: Lead Time, dtype: float64
    
    Very Early Cancellation Stats:
    count    135.000
    mean    1218.288
    std      195.949
    min     1000.745
    25%     1063.788
    50%     1139.868
    75%     1319.083
    max     1854.237
    Name: Lead Time, dtype: float64
    


```python
# === Comprehensive Cancellation Analysis ===
print("=== Cancellation Type Analysis ===")
# Basic cancellation types
cancellation_types = cancellations_df['Action'].value_counts()
print("\nCancellation Action Types:")
print(cancellation_types)

# Detailed Pattern Analysis
def analyze_cancellation_patterns(clean_cancellations, shifts_df):
    """
    Analyze patterns in cancellations, including:
    - Types of cancellations (NCNS vs Regular)
    - Role-based patterns
    - Shift type patterns
    """
    # Merge with shifts to analyze by role
    cancellations_with_shifts = pd.merge(
        clean_cancellations,
        shifts_df[['ID', 'Agent Req', 'Shift Type', 'Charge']],
        left_on='Shift ID',
        right_on='ID',
        how='left'
    )
    
    print("\n=== Cancellations by Role ===")
    role_cancels = pd.crosstab(
        cancellations_with_shifts['Agent Req'],
        cancellations_with_shifts['cancellation_category'],
        normalize='index'
    ).round(3) * 100
    print(role_cancels)
    
    print("\n=== Cancellations by Shift Type ===")
    shift_cancels = pd.crosstab(
        cancellations_with_shifts['Shift Type'],
        cancellations_with_shifts['cancellation_category'],
        normalize='index'
    ).round(3) * 100
    print(shift_cancels)
    
    # Role-based impact analysis
    role_impact = cancellations_with_shifts.groupby('Agent Req').agg({
        'Shift ID': 'count',
        'Lead Time': ['mean', 'std'],
        'Action': lambda x: (x == 'NO_CALL_NO_SHOW').mean() * 100
    }).round(2)
    role_impact.columns = ['Total Cancellations', 'Avg Lead Time', 'Lead Time Std', 'NCNS Rate']
    print("\n=== Role-Based Impact ===")
    print(role_impact)
    
    # Store results
    summary.add_summary('cancellations', 'action_types', cancellation_types.to_dict())
    summary.add_summary('cancellations', 'role_patterns', role_cancels.to_dict())
    summary.add_summary('cancellations', 'shift_patterns', shift_cancels.to_dict())
    summary.add_summary('cancellations', 'role_impact', role_impact.to_dict())
    
    return cancellations_with_shifts

# Run the analysis
cancellations_with_shifts = analyze_cancellation_patterns(clean_cancellations, shifts_df)
```

    === Cancellation Type Analysis ===
    
    Cancellation Action Types:
    Action
    WORKER_CANCEL      66108
    NO_CALL_NO_SHOW    11965
    Name: count, dtype: int64
    
    === Cancellations by Role ===
    cancellation_category  Advance (<3 days)  Early (3+ days)  Late (<4hrs)  \
    Agent Req                                                                 
    CAREGIVER                          0.000            0.000        25.000   
    CNA                                6.700           44.100        18.700   
    LVN                                8.700           40.900        16.800   
    NURSE                              7.400           23.800        30.500   
    RN                                11.500           59.800         9.400   
    
    cancellation_category  No-Show  Same Day  
    Agent Req                                 
    CAREGIVER               75.000     0.000  
    CNA                     17.200    13.200  
    LVN                     12.500    21.100  
    NURSE                   18.100    20.200  
    RN                       5.700    13.500  
    
    === Cancellations by Shift Type ===
    cancellation_category  Advance (<3 days)  Early (3+ days)  Late (<4hrs)  \
    Shift Type                                                                
    am                                 6.700           42.400        15.300   
    custom                            11.400           34.200        13.400   
    noc                                6.900           45.100        19.300   
    pm                                 7.800           43.200        21.400   
    
    cancellation_category  No-Show  Same Day  
    Shift Type                                
    am                      19.400    16.100  
    custom                  24.200    16.800  
    noc                     13.400    15.400  
    pm                      15.100    12.500  
    
    === Role-Based Impact ===
               Total Cancellations  Avg Lead Time  Lead Time Std  NCNS Rate
    Agent Req                                                              
    CAREGIVER                    4        -62.810         45.730     75.000
    CNA                       6066        101.970        232.930     17.230
    LVN                       1184         98.490        177.570     12.500
    NURSE                      282         44.380        132.820     18.090
    RN                         244        142.140        188.720      5.740
    


```python
# === Numerical Analysis ===
print("\n=== Numerical Analysis ===")
# Basic statistics for numerical columns
numeric_stats = shifts_df[['Charge', 'Time']].describe()
print("\nNumerical Statistics:")
print(numeric_stats)

# Additional numeric insights
print("\nCharge Rate Analysis:")
print(f"Shifts with zero charge: {(shifts_df['Charge'] == 0).sum()}")
print(f"Average charge by agent type:")
print(shifts_df.groupby('Agent Req')['Charge'].mean().round(2))

# === Categorical Analysis ===
print("\n=== Categorical Analysis ===")
# Shift type distribution
print("\nShift Type Distribution:")
shift_type_dist = shifts_df['Shift Type'].value_counts(dropna=True)
print(shift_type_dist)

# Agent requirements
print("\nAgent Requirement Distribution:")
agent_req_dist = shifts_df['Agent Req'].value_counts(dropna=True)
print(agent_req_dist)

# Cross-tabulation of shift types and agent requirements
print("\nShift Types by Agent Requirements:")
print(pd.crosstab(shifts_df['Shift Type'], shifts_df['Agent Req']))

# === Data Completeness Analysis ===
print("\n=== Data Completeness Analysis ===")
complete_rows = shifts_df.dropna().shape[0]
print(f"Complete rows: {complete_rows} out of {shifts_df.shape[0]}")
print(f"Completion rate: {(complete_rows/shifts_df.shape[0]*100):.2f}%")

# === Time-Based Analysis ===
print("\n=== Time-Based Analysis ===")
# Extract time components
shifts_df['Hour'] = shifts_df['Start'].dt.hour
shifts_df['Day'] = shifts_df['Start'].dt.day_name()
shifts_df['Month'] = shifts_df['Start'].dt.month
shifts_df['Shift_Length'] = (shifts_df['End'] - shifts_df['Start']).dt.total_seconds() / 3600
# Time patterns
print("\nShifts by Hour:")
hour_dist = shifts_df['Hour'].value_counts().sort_index()
print(hour_dist)

print("\nShifts by Day of Week:")
day_dist = shifts_df['Day'].value_counts()
print(day_dist)

print("\nShift Length Distribution:")
print(shifts_df['Shift_Length'].describe().round(2))

# === Facility Analysis ===
print("\n=== Facility Analysis ===")
facility_stats = shifts_df.groupby('Facility ID').agg({
    'ID': 'count',
    'Charge': 'mean',
    'Time': 'mean'
}).rename(columns={
    'ID': 'Number of Shifts',
    'Charge': 'Average Charge',
    'Time': 'Average Shift Length'
})
print("\nFacility Statistics:")
print(facility_stats.head())
print(f"\nTotal unique facilities: {shifts_df['Facility ID'].nunique()}")

# Store all results
summary.add_summary('shifts', 'numeric_stats', numeric_stats)
summary.add_summary('shifts', 'shift_types', shift_type_dist.to_dict())
summary.add_summary('shifts', 'agent_types', agent_req_dist.to_dict())
summary.add_summary('shifts', 'hour_distribution', hour_dist.to_dict())
summary.add_summary('shifts', 'day_distribution', day_dist.to_dict())
summary.add_summary('shifts', 'facility_stats', facility_stats.to_dict())

# Optional: Create visualizations
# We can add matplotlib/seaborn plots here if you'd like
```

    
    === Numerical Analysis ===
    
    Numerical Statistics:
             Charge      Time
    count 41040.000 41040.000
    mean     36.353     7.626
    std      18.245     2.872
    min       0.000    -0.500
    25%      28.000     7.500
    50%      37.000     7.500
    75%      47.000     8.000
    max     120.000    17.010
    
    Charge Rate Analysis:
    Shifts with zero charge: 5019
    Average charge by agent type:
    Agent Req
    CAREGIVER            30.770
    CNA                  30.710
    LVN                  47.300
    Medical Aide         10.670
    Medical Technician    9.080
    NURSE                 0.000
    PT                   10.110
    RN                   63.710
    Name: Charge, dtype: float64
    
    === Categorical Analysis ===
    
    Shift Type Distribution:
    Shift Type
    pm        14232
    am        13951
    noc       11446
    custom     1411
    Name: count, dtype: int64
    
    Agent Requirement Distribution:
    Agent Req
    CNA                   24662
    LVN                   11277
    RN                     3139
    NURSE                  1907
    CAREGIVER                31
    Medical Technician       12
    PT                        9
    Medical Aide              3
    Name: count, dtype: int64
    
    Shift Types by Agent Requirements:
    Agent Req   CAREGIVER   CNA   LVN  Medical Aide  Medical Technician  NURSE  \
    Shift Type                                                                   
    am                  7  8329  3961             2                   1    560   
    custom              0   609   620             0                   0    106   
    noc                20  6881  3230             0                   1    407   
    pm                  4  8843  3466             1                  10    834   
    
    Agent Req   PT    RN  
    Shift Type            
    am           1  1090  
    custom       3    73  
    noc          5   902  
    pm           0  1074  
    
    === Data Completeness Analysis ===
    Complete rows: 4270 out of 41040
    Completion rate: 10.40%
    
    === Time-Based Analysis ===
    
    Shifts by Hour:
    Hour
    0     2876
    1       13
    2      560
    3     2960
    4     4836
    7       10
    8       15
    9        4
    10    1283
    11    5210
    12    7336
    13      88
    14      86
    15     115
    16      78
    17      38
    18     947
    19    4114
    20    6559
    21     113
    22    1114
    23    2685
    Name: count, dtype: int64
    
    Shifts by Day of Week:
    Day
    Sunday       6639
    Saturday     6529
    Monday       6370
    Friday       6184
    Tuesday      5194
    Thursday     5139
    Wednesday    4985
    Name: count, dtype: int64
    
    Shift Length Distribution:
    count   41040.000
    mean        8.840
    std         2.150
    min         0.000
    25%         8.000
    50%         8.000
    75%         8.500
    max        16.000
    Name: Shift_Length, dtype: float64
    
    === Facility Analysis ===
    
    Facility Statistics:
                              Number of Shifts  Average Charge  \
    Facility ID                                                  
    5c3cc2e3a2ae5c0016bef82c              1639          24.743   
    5c4783699679f70016ea793d               347          25.058   
    5cae2b12b2e65a001646dc85                14          38.143   
    5d55c3cfd07e7a0016327d01               109          37.991   
    5e5849e1e0adc90016335e0e                23          42.348   
    
                              Average Shift Length  
    Facility ID                                     
    5c3cc2e3a2ae5c0016bef82c                 6.800  
    5c4783699679f70016ea793d                 6.670  
    5cae2b12b2e65a001646dc85                10.929  
    5d55c3cfd07e7a0016327d01                10.397  
    5e5849e1e0adc90016335e0e                11.022  
    
    Total unique facilities: 67
    


```python
# Relationship analysis
# First, let's see how many shifts had cancellations
shifts_with_cancellations = len(set(shifts_df['ID']) & set(cancellations_df['Shift ID']))
print(f"Shifts with cancellations: {shifts_with_cancellations}")
print(f"Percentage of shifts cancelled: {(shifts_with_cancellations/len(shifts_df))*100:.2f}%")

# Analyze cancellation lead times
cancellations_df['Lead Time'].describe()
```

    Shifts with cancellations: 6535
    Percentage of shifts cancelled: 15.92%
    




    count   78073.000
    mean      105.908
    std       215.991
    min     -4210.100
    25%         2.254
    50%        36.303
    75%       200.658
    max      1854.237
    Name: Lead Time, dtype: float64




```python
# === Booking Pattern Analysis ===
def analyze_booking_patterns(bookings_df, shifts_df, clean_cancellations):
    """
    Analyze patterns in shift bookings, including:
    - Time from posting to booking
    - Successful vs cancelled bookings
    - Rebooking patterns after cancellations
    
    Parameters:
        bookings_df (pd.DataFrame): Booking logs data
        shifts_df (pd.DataFrame): Shifts data
        clean_cancellations (pd.DataFrame): Cleaned cancellations data
    """
    print("=== Booking Success Analysis ===")

    # Calculate time to fill (from shift creation to booking)
    bookings_with_shifts = pd.merge(
        bookings_df,
        shifts_df[['ID', 'Created At', 'Agent Req', 'Shift Type', 'Charge']],
        left_on='Shift ID',
        right_on='ID',
        how='left',
        suffixes=('_booking', '_shift')
    )
    
    bookings_with_shifts['time_to_fill'] = (
        pd.to_datetime(bookings_with_shifts['Created At_booking']) - 
        pd.to_datetime(bookings_with_shifts['Created At_shift'])
    ).dt.total_seconds() / 3600  # Convert to hours
    
    print("\nTime to Fill Statistics (hours):")
    print(bookings_with_shifts['time_to_fill'].describe().round(2))
    
    # Analyze bookings by role and shift type
    print("\n=== Bookings by Role ===")
    role_bookings = bookings_with_shifts.groupby('Agent Req').agg({
        'Shift ID': 'count',  # Changed from 'ID' to 'Shift ID'
        'time_to_fill': 'mean',
        'Charge': 'mean'
    }).round(2)
    role_bookings.columns = ['Number of Bookings', 'Avg Time to Fill', 'Avg Charge']
    print(role_bookings)
    
    # Look at shifts that got cancelled and rebooked
    rebooked_cancellations = clean_cancellations['Shift ID'].value_counts()
    
    print("\n=== Rebooking Analysis ===")
    print(f"Shifts cancelled multiple times: {(rebooked_cancellations > 1).sum()}")
    print(f"Maximum cancellations for a single shift: {rebooked_cancellations.max()}")
    
    # Additional timing analysis
    print("\n=== Booking Time Patterns ===")
    bookings_with_shifts['booking_hour'] = pd.to_datetime(bookings_with_shifts['Created At_booking']).dt.hour
    bookings_with_shifts['booking_day'] = pd.to_datetime(bookings_with_shifts['Created At_booking']).dt.day_name()
    
    print("\nBookings by Hour of Day:")
    print(bookings_with_shifts['booking_hour'].value_counts().sort_index())
    
    print("\nBookings by Day of Week:")
    print(bookings_with_shifts['booking_day'].value_counts())
    
    # Store results
    summary.add_summary('bookings', 'time_to_fill', 
                       bookings_with_shifts['time_to_fill'].describe().to_dict())
    summary.add_summary('bookings', 'role_patterns', role_bookings.to_dict())
    summary.add_summary('bookings', 'rebooking_stats', {
        'multiple_cancellations': (rebooked_cancellations > 1).sum(),
        'max_cancellations': rebooked_cancellations.max()
    })
    
    return bookings_with_shifts

# Run the analysis
bookings_with_shifts = analyze_booking_patterns(bookings_df, shifts_df, clean_cancellations)    
```

    === Booking Success Analysis ===
    
    Time to Fill Statistics (hours):
    count   11162.000
    mean      156.410
    std       197.200
    min         0.000
    25%         8.470
    50%        70.410
    75%       232.580
    max      1082.830
    Name: time_to_fill, dtype: float64
    
    === Bookings by Role ===
               Number of Bookings  Avg Time to Fill  Avg Charge
    Agent Req                                                  
    CNA                      8578           148.390      28.150
    LVN                      2086           179.640      42.700
    NURSE                     122           181.890       0.000
    RN                        376           202.190      59.350
    
    === Rebooking Analysis ===
    Shifts cancelled multiple times: 10302
    Maximum cancellations for a single shift: 18
    
    === Booking Time Patterns ===
    
    Bookings by Hour of Day:
    booking_hour
    0     6538
    1     5642
    2     5113
    3     4067
    4     3437
    5     2998
    6     2390
    7     1852
    8     1636
    9     1395
    10    1343
    11    1973
    12    2871
    13    3830
    14    5287
    15    6717
    16    7731
    17    8574
    18    9527
    19    9899
    20    9763
    21    9161
    22    7972
    23    7289
    Name: count, dtype: int64
    
    Bookings by Day of Week:
    booking_day
    Tuesday      22052
    Wednesday    21991
    Thursday     20951
    Friday       19956
    Monday       18201
    Saturday     13467
    Sunday       10387
    Name: count, dtype: int64
    


```python
# === Economic Impact Analysis ===
def analyze_economic_impact(shifts_df, cancellations_with_shifts):
    """
    Analyze economic impact of cancellations, including:
    - Revenue loss from cancellations
    - Impact by facility and role type
    - Patterns in high-cost cancellations
    """
    print("=== Economic Impact Analysis ===")
    
    # First, ensure we have all needed columns by merging with shifts data if needed
    if 'Time' not in cancellations_with_shifts.columns:
        cancellations_with_shifts = pd.merge(
            cancellations_with_shifts,
            shifts_df[['ID', 'Time', 'Charge']],
            left_on='Shift ID',
            right_on='ID',
            how='left',
            suffixes=('', '_shift')
        )

    # Calculate baseline metrics
    total_revenue = (shifts_df['Charge'] * shifts_df['Time']).sum()
    avg_hourly_revenue = shifts_df['Charge'].mean()
    
    # Analyze cancelled shifts
    cancelled_revenue = (cancellations_with_shifts['Charge'] * 
                        cancellations_with_shifts['Time']).sum()
    
    print("\nBaseline Metrics:")
    print(f"Total Potential Revenue: ${total_revenue:,.2f}")
    print(f"Average Hourly Rate: ${avg_hourly_revenue:.2f}")
    print(f"Lost Revenue from Cancellations: ${cancelled_revenue:,.2f}")
    if total_revenue > 0:  # Avoid division by zero
        print(f"Percentage of Revenue Lost: {(cancelled_revenue/total_revenue)*100:.2f}%")

    # Analysis by role type
    print("\n=== Impact by Role Type ===")
    role_impact = cancellations_with_shifts.groupby('Agent Req').agg({
        'Shift ID': 'count',
        'Charge': ['mean', 'sum'],
        'Time': 'sum'
    }).round(2)
    role_impact.columns = ['Cancellations', 'Avg Rate', 'Total Charge', 'Total Hours']
    role_impact['Est. Revenue Loss'] = role_impact['Avg Rate'] * role_impact['Total Hours']
    print(role_impact.sort_values('Est. Revenue Loss', ascending=False))

    # Analysis by cancellation type
    print("\n=== Impact by Cancellation Type ===")
    type_impact = cancellations_with_shifts.groupby('cancellation_category').agg({
        'Shift ID': 'count',
        'Charge': ['mean', 'sum'],
        'Time': 'sum'
    }).round(2)
    type_impact.columns = ['Cancellations', 'Avg Rate', 'Total Charge', 'Total Hours']
    type_impact['Est. Revenue Loss'] = type_impact['Avg Rate'] * type_impact['Total Hours']
    print(type_impact.sort_values('Est. Revenue Loss', ascending=False))

    # Calculate impact by facility
    print("\n=== Top 5 Facilities by Revenue Loss ===")
    facility_impact = cancellations_with_shifts.groupby('Facility ID').agg({
        'Shift ID': 'count',
        'Charge': ['mean', 'sum'],
        'Time': 'sum'
    }).round(2)
    facility_impact.columns = ['Cancellations', 'Avg Rate', 'Total Charge', 'Total Hours']
    facility_impact['Est. Revenue Loss'] = facility_impact['Avg Rate'] * facility_impact['Total Hours']
    print(facility_impact.nlargest(5, 'Est. Revenue Loss'))

    # Store results
    summary.add_summary('economic', 'overall_impact', {
        'total_revenue': total_revenue,
        'cancelled_revenue': cancelled_revenue,
        'avg_hourly_rate': avg_hourly_revenue
    })
    summary.add_summary('economic', 'role_impact', role_impact.to_dict())
    summary.add_summary('economic', 'type_impact', type_impact.to_dict())

    return role_impact, type_impact, facility_impact

# Run the analysis
role_impact, type_impact, facility_impact = analyze_economic_impact(shifts_df, cancellations_with_shifts)
```

    === Economic Impact Analysis ===
    
    Baseline Metrics:
    Total Potential Revenue: $12,306,970.09
    Average Hourly Rate: $36.35
    Lost Revenue from Cancellations: $2,169,704.35
    Percentage of Revenue Lost: 17.63%
    
    === Impact by Role Type ===
               Cancellations  Avg Rate  Total Charge  Total Hours  \
    Agent Req                                                       
    CNA                 6066    32.110    194775.250    44462.680   
    LVN                 1184    48.550     57480.250    10244.350   
    RN                   244    64.770     15805.000     2012.000   
    CAREGIVER              4    35.000       140.000       30.000   
    NURSE                282     0.000         0.000     2527.560   
    
               Est. Revenue Loss  
    Agent Req                     
    CNA              1427696.655  
    LVN               497363.193  
    RN                130317.240  
    CAREGIVER           1050.000  
    NURSE                  0.000  
    
    === Impact by Cancellation Type ===
                           Cancellations  Avg Rate  Total Charge  Total Hours  \
    cancellation_category                                                       
    Early (3+ days)                33853    32.510    109670.500    24117.000   
    Late (<4hrs)                   11102    35.170     50815.500    11653.600   
    No-Show                        11966    35.510     44781.500    10111.360   
    Same Day                       13219    36.980     42157.500     9015.990   
    Advance (<3 days)               7933    37.030     20775.500     4378.640   
    
                           Est. Revenue Loss  
    cancellation_category                     
    Early (3+ days)               784043.670  
    Late (<4hrs)                  409857.112  
    No-Show                       359054.394  
    Same Day                      333411.310  
    Advance (<3 days)             162141.039  
    
    === Top 5 Facilities by Revenue Loss ===
                              Cancellations  Avg Rate  Total Charge  Total Hours  \
    Facility ID                                                                    
    5f9b189a7ecb880016516a52            585    33.230     19442.000     4049.310   
    5f9ad22ae3a95f0016090f97            480    37.610     18052.000     3520.440   
    6137c340b7995a01665c51df            551    32.580     17952.000     3916.620   
    5f91b184cb91b40016e1e183            362    37.550     13591.750     3055.280   
    61422c7f81f2950166d5f881            264    42.060     11105.000     2583.170   
    
                              Est. Revenue Loss  
    Facility ID                                  
    5f9b189a7ecb880016516a52         134558.571  
    5f9ad22ae3a95f0016090f97         132403.748  
    6137c340b7995a01665c51df         127603.480  
    5f91b184cb91b40016e1e183         114725.764  
    61422c7f81f2950166d5f881         108648.130  
    


```python
def audit_data_quality(shifts_df, cancellations_df, bookings_df):
    """
    Comprehensive data quality audit focusing on business-critical issues
    
    Parameters:
    - shifts_df: DataFrame containing shift data
    - cancellations_df: DataFrame containing cancellation logs
    - bookings_df: DataFrame containing booking logs
    
    Returns:
    - Dictionary containing quality issues by dataset
    """
    quality_issues = {
        'shifts': {},
        'cancellations': {},
        'bookings': {}
    }
    
    # Shifts Analysis
    shifts_issues = {
        # Financial data issues
        'zero_charge': (shifts_df['Charge'] == 0).sum(),
        'negative_charge': (shifts_df['Charge'] < 0).sum(),
        
        # Time-related issues
        'zero_time': (shifts_df['Time'] == 0).sum(),
        'negative_time': (shifts_df['Time'] < 0).sum(),
        'end_before_start': (shifts_df['End'] < shifts_df['Start']).sum(),
        
        # Missing data
        'missing_agent': shifts_df['Agent ID'].isnull().sum(),
        'missing_facility': shifts_df['Facility ID'].isnull().sum(),
        'missing_shift_type': shifts_df['Shift Type'].isnull().sum(),
        
        # Verification issues
        'unverified_completed': ((shifts_df['End'] < pd.Timestamp.now()) & 
                                (shifts_df['Verified'].isnull())).sum(),
        
        # Invalid shift types
        'invalid_shift_types': shifts_df[~shifts_df['Shift Type'].isin(['am', 'pm', 'noc', 'custom'])].shape[0]
    }
    
    # Cancellations Analysis
    cancel_issues = {
        # Lead time issues
        'invalid_lead_time': (cancellations_df['Lead Time'].isnull() | 
                            ~np.isfinite(cancellations_df['Lead Time'])).sum(),
        'extreme_negative_lead': (cancellations_df['Lead Time'] < -72).sum(),  # More than 3 days after start
        'extreme_positive_lead': (cancellations_df['Lead Time'] > 720).sum(),  # More than 30 days before
        
        # Missing data
        'missing_worker': cancellations_df['Worker ID'].isnull().sum(),
        'missing_facility': cancellations_df['Facility ID'].isnull().sum(),
        
        # Duplicate issues
        'duplicate_cancels': cancellations_df.groupby('Shift ID').size().gt(1).sum(),
        
        # Action type validation
        'invalid_actions': cancellations_df[~cancellations_df['Action'].isin(
            ['WORKER_CANCEL', 'NO_CALL_NO_SHOW'])].shape[0]
    }
    
    # Bookings Analysis
    bookings_issues = {
        # Missing data
        'missing_worker': bookings_df['Worker ID'].isnull().sum(),
        'missing_facility': bookings_df['Facility ID'].isnull().sum(),
        
        # Lead time issues (time between booking and shift start)
        'invalid_lead_time': (bookings_df['Lead Time'].isnull() | 
                            ~np.isfinite(bookings_df['Lead Time'])).sum(),
        
        # Action validation
        'invalid_actions': bookings_df[bookings_df['Action'] != 'SHIFT_CLAIM'].shape[0]
    }
    
    # Cross-dataset validation
    cross_validation = {
        'orphaned_cancels': cancellations_df[~cancellations_df['Shift ID'].isin(shifts_df['ID'])].shape[0],
        'orphaned_bookings': bookings_df[~bookings_df['Shift ID'].isin(shifts_df['ID'])].shape[0],
        'multiple_workers': shifts_df.groupby('ID')['Agent ID'].nunique().gt(1).sum(),
        'booking_cancel_mismatch': len(
            set(cancellations_df[cancellations_df['Action'] == 'WORKER_CANCEL']['Shift ID']) - 
            set(bookings_df['Shift ID'])
        )
    }
    
    quality_issues['shifts'] = shifts_issues
    quality_issues['cancellations'] = cancel_issues
    quality_issues['bookings'] = bookings_issues
    quality_issues['cross_validation'] = cross_validation
    
    return quality_issues

# Function to display audit results in a readable format
def display_audit_results(audit_results):
    """
    Display audit results in a clear, organized format
    """
    for dataset, issues in audit_results.items():
        print(f"\n=== {dataset.upper()} QUALITY ISSUES ===")
        for issue, count in issues.items():
            print(f"{issue}: {count:,}")


# === In Initial Data Loading and Validation Section ===

print("Performing data quality audit...")
# Run the audit
audit_results = audit_data_quality(shifts_df, cancellations_df, bookings_df)

# Display results
display_audit_results(audit_results)

# Store results in summary
summary.add_summary('data_quality', 'audit_results', audit_results)

# Optional: Display specific issues that need attention
significant_issues = {
    dataset: {issue: count for issue, count in issues.items() if count > 0}
    for dataset, issues in audit_results.items()
}

print("\nSignificant issues requiring attention:")
for dataset, issues in significant_issues.items():
    if issues:  # Only show datasets with issues
        print(f"\n{dataset}:")
        for issue, count in issues.items():
            print(f"- {issue}: {count:,}")
```

    Performing data quality audit...
    
    === SHIFTS QUALITY ISSUES ===
    zero_charge: 5,019
    negative_charge: 0
    zero_time: 79
    negative_time: 22
    end_before_start: 0
    missing_agent: 20,035
    missing_facility: 0
    missing_shift_type: 0
    unverified_completed: 0
    invalid_shift_types: 0
    
    === CANCELLATIONS QUALITY ISSUES ===
    invalid_lead_time: 0
    extreme_negative_lead: 4,960
    extreme_positive_lead: 741
    missing_worker: 191
    missing_facility: 0
    duplicate_cancels: 10,302
    invalid_actions: 0
    
    === BOOKINGS QUALITY ISSUES ===
    missing_worker: 140
    missing_facility: 0
    invalid_lead_time: 0
    invalid_actions: 0
    
    === CROSS_VALIDATION QUALITY ISSUES ===
    orphaned_cancels: 70,293
    orphaned_bookings: 115,843
    multiple_workers: 0
    booking_cancel_mismatch: 36,160
    
    Significant issues requiring attention:
    
    shifts:
    - zero_charge: 5,019
    - zero_time: 79
    - negative_time: 22
    - missing_agent: 20,035
    
    cancellations:
    - extreme_negative_lead: 4,960
    - extreme_positive_lead: 741
    - missing_worker: 191
    - duplicate_cancels: 10,302
    
    bookings:
    - missing_worker: 140
    
    cross_validation:
    - orphaned_cancels: 70,293
    - orphaned_bookings: 115,843
    - booking_cancel_mismatch: 36,160
    

## This suggests that:

Many cancellations and bookings don't link to shifts in our dataset
Could be due to date range mismatches or data completeness issues
Critical for understanding true cancellation rates


Worker/Agent Data Gaps

Copymissing_agent: 20,035 (shifts)
missing_worker: 191 (cancellations)
missing_worker: 140 (bookings)
This matches what we saw earlier but gives us a more complete picture. Particularly important because:

About half of shifts are missing agent IDs
Affects our ability to analyze worker patterns
Could impact our ability to track repeat cancellations


Lead Time Issues

Copyextreme_negative_lead: 4,960
extreme_positive_lead: 741
This provides more granular insight than our earlier analysis. Important because:

Shows significant number of very late cancellations (>3 days after start)
Identifies early cancellations that might need different handling
Relevant to the attendance policy analysis


Financial Data Quality

Copyzero_charge: 5,019
zero_time: 79
negative_time: 22
Matches our earlier findings but gives more context about potential revenue impact.


```python
# at this point realized the data doesn't overlap in dates 
def analyze_data_coverage():
    """
    Analyze the time coverage and relationships between datasets
    
    Returns:
    - Dictionary containing date ranges and overlap analysis for each dataset
    """
    # Get date ranges for each dataset
    print("Analyzing dataset date coverage...")
    
    shift_dates = shifts_df['Start'].dt.date.value_counts().sort_index()
    cancel_dates = cancellations_df['Created At'].dt.date.value_counts().sort_index()
    booking_dates = bookings_df['Created At'].dt.date.value_counts().sort_index()
    
    # Analyze overlap periods
    date_ranges = {
        'shifts': {
            'start': shift_dates.index.min(),
            'end': shift_dates.index.max(),
            'total_days': len(shift_dates),
            'avg_shifts_per_day': shift_dates.mean()
        },
        'cancellations': {
            'start': cancel_dates.index.min(),
            'end': cancel_dates.index.max(),
            'total_days': len(cancel_dates),
            'avg_cancels_per_day': cancel_dates.mean()
        },
        'bookings': {
            'start': booking_dates.index.min(),
            'end': booking_dates.index.max(),
            'total_days': len(booking_dates),
            'avg_bookings_per_day': booking_dates.mean()
        }
    }
    
    return date_ranges

def analyze_missing_data_impact():
    """
    Assess how missing data affects our key metrics
    
    Returns:
    - Dictionary containing comparative analysis of shifts with/without missing data
    """
    print("\nAnalyzing impact of missing data...")
    
    # Analyze shifts with/without missing agent IDs
    missing_agent_shifts = shifts_df[shifts_df['Agent ID'].isnull()]
    complete_shifts = shifts_df[shifts_df['Agent ID'].notnull()]
    
    # Get cancellation rates
    missing_cancels = len(set(missing_agent_shifts['ID']) & set(cancellations_df['Shift ID']))
    complete_cancels = len(set(complete_shifts['ID']) & set(cancellations_df['Shift ID']))
    
    comparison = {
        'missing_agent': {
            'count': len(missing_agent_shifts),
            'avg_charge': missing_agent_shifts['Charge'].mean(),
            'avg_duration': missing_agent_shifts['Time'].mean(),
            'cancellation_count': missing_cancels,
            'cancellation_rate': missing_cancels / len(missing_agent_shifts) if len(missing_agent_shifts) > 0 else 0,
            'verified_rate': missing_agent_shifts['Verified'].mean()
        },
        'complete_data': {
            'count': len(complete_shifts),
            'avg_charge': complete_shifts['Charge'].mean(),
            'avg_duration': complete_shifts['Time'].mean(),
            'cancellation_count': complete_cancels,
            'cancellation_rate': complete_cancels / len(complete_shifts) if len(complete_shifts) > 0 else 0,
            'verified_rate': complete_shifts['Verified'].mean()
        }
    }
    
    return comparison

# Run both analyses
print("Running additional data quality analyses...\n")

# Analyze data coverage
coverage_results = analyze_data_coverage()
print("\n=== Dataset Coverage Analysis ===")
for dataset, info in coverage_results.items():
    print(f"\n{dataset.upper()} Coverage:")
    for metric, value in info.items():
        print(f"{metric}: {value}")

# Analyze missing data impact
impact_results = analyze_missing_data_impact()
print("\n=== Missing Data Impact Analysis ===")
for category, metrics in impact_results.items():
    print(f"\n{category.replace('_', ' ').title()}:")
    for metric, value in metrics.items():
        if 'rate' in metric:
            print(f"{metric}: {value:.2%}")
        else:
            print(f"{metric}: {value:,.2f}")

# Store results in summary
summary.add_summary('data_quality', 'coverage_analysis', coverage_results)
summary.add_summary('data_quality', 'missing_data_impact', impact_results)
```

    Running additional data quality analyses...
    
    Analyzing dataset date coverage...
    
    === Dataset Coverage Analysis ===
    
    SHIFTS Coverage:
    start: 2021-10-01
    end: 2022-01-31
    total_days: 123
    avg_shifts_per_day: 333.6585365853659
    
    CANCELLATIONS Coverage:
    start: 2021-09-06
    end: 2022-07-09
    total_days: 199
    avg_cancels_per_day: 392.32663316582915
    
    BOOKINGS Coverage:
    start: 2021-08-13
    end: 2022-04-12
    total_days: 130
    avg_bookings_per_day: 976.9615384615385
    
    Analyzing impact of missing data...
    
    === Missing Data Impact Analysis ===
    
    Missing Agent:
    count: 20,035.00
    avg_charge: 38.48
    avg_duration: 8.20
    cancellation_count: 3,561.00
    cancellation_rate: 17.77%
    verified_rate: 0.80%
    
    Complete Data:
    count: 21,005.00
    avg_charge: 34.32
    avg_duration: 7.08
    cancellation_count: 2,974.00
    cancellation_rate: 14.16%
    verified_rate: 77.31%
    


```python
def analyze_data_completeness(shifts_df, cancellations_df, bookings_df, verbose=True):
    """
    Analyze dataset completeness and coverage periods.
    
    Parameters:
    ----------
    shifts_df : pandas.DataFrame
        Shift data containing columns: 'Start', 'Agent ID', etc.
    cancellations_df : pandas.DataFrame
        Cancellation data containing columns: 'Created At', etc.
    bookings_df : pandas.DataFrame
        Booking data containing columns: 'Created At', etc.
    verbose : bool, default=True
        If True, prints detailed analysis results
        
    Returns:
    --------
    dict
        Dictionary containing coverage analysis and completeness metrics
    """
    results = {}
    
    # Dataset Coverage Analysis
    coverage = {
        'shifts': {
            'date_range': (shifts_df['Start'].min(), shifts_df['End'].max()),
            'total_records': len(shifts_df),
            'daily_average': len(shifts_df) / shifts_df['Start'].dt.date.nunique()
        },
        'cancellations': {
            'date_range': (cancellations_df['Created At'].min(), 
                         cancellations_df['Created At'].max()),
            'total_records': len(cancellations_df),
            'daily_average': len(cancellations_df) / cancellations_df['Created At'].dt.date.nunique()
        },
        'bookings': {
            'date_range': (bookings_df['Created At'].min(), 
                         bookings_df['Created At'].max()),
            'total_records': len(bookings_df),
            'daily_average': len(bookings_df) / bookings_df['Created At'].dt.date.nunique()
        }
    }
    results['coverage'] = coverage
    
    # Data Completeness Analysis
    completeness = {
        'shifts': {
            'missing_agent_id': {
                'count': shifts_df['Agent ID'].isnull().sum(),
                'percentage': (shifts_df['Agent ID'].isnull().sum() / len(shifts_df)) * 100
            },
            'verified_shifts': {
                'count': shifts_df['Verified'].sum(),
                'percentage': (shifts_df['Verified'].sum() / len(shifts_df)) * 100
            }
        },
        'cancellations': {
            'missing_worker_id': {
                'count': cancellations_df['Worker ID'].isnull().sum(),
                'percentage': (cancellations_df['Worker ID'].isnull().sum() / len(cancellations_df)) * 100
            }
        },
        'bookings': {
            'missing_worker_id': {
                'count': bookings_df['Worker ID'].isnull().sum(),
                'percentage': (bookings_df['Worker ID'].isnull().sum() / len(bookings_df)) * 100
            }
        }
    }
    results['completeness'] = completeness
    
    if verbose:
        print("=== Dataset Coverage Analysis ===")
        print("\nTime Periods:")
        for dataset, info in coverage.items():
            print(f"\n{dataset.upper()}:")
            print(f"Date Range: {info['date_range'][0].date()} to {info['date_range'][1].date()}")
            print(f"Total Records: {info['total_records']:,}")
            print(f"Daily Average: {info['daily_average']:.2f}")
        
        print("\n=== Data Completeness Analysis ===")
        for dataset, metrics in completeness.items():
            print(f"\n{dataset.upper()} Completeness:")
            for field, values in metrics.items():
                print(f"{field}:")
                print(f"  Count: {values['count']:,}")
                print(f"  Percentage: {values['percentage']:.2f}%")
    
    return results

# Run the analysis
completeness_results = analyze_data_completeness(shifts_df, cancellations_df, bookings_df)
```

    === Dataset Coverage Analysis ===
    
    Time Periods:
    
    SHIFTS:
    Date Range: 2021-10-01 to 2022-02-01
    Total Records: 41,040
    Daily Average: 333.66
    
    CANCELLATIONS:
    Date Range: 2021-09-06 to 2022-07-09
    Total Records: 78,073
    Daily Average: 392.33
    
    BOOKINGS:
    Date Range: 2021-08-13 to 2022-04-12
    Total Records: 127,005
    Daily Average: 976.96
    
    === Data Completeness Analysis ===
    
    SHIFTS Completeness:
    missing_agent_id:
      Count: 20,035
      Percentage: 48.82%
    verified_shifts:
      Count: 16,400
      Percentage: 39.96%
    
    CANCELLATIONS Completeness:
    missing_worker_id:
      Count: 191
      Percentage: 0.24%
    
    BOOKINGS Completeness:
    missing_worker_id:
      Count: 140
      Percentage: 0.11%
    

# Business Critical Questions


```python
def analyze_missing_agent_patterns(shifts_df):
    """
    Analyze patterns in shifts with missing Agent IDs
    
    Parameters:
    shifts_df: DataFrame containing shift data
    
    Returns:
    Dictionary containing analysis results
    """
    # Separate shifts with/without Agent IDs
    missing_agent = shifts_df[shifts_df['Agent ID'].isnull()]
    has_agent = shifts_df[shifts_df['Agent ID'].notnull()]
    
    analysis = {
        'temporal_patterns': {
            'missing_by_month': missing_agent['Start'].dt.to_period('M').value_counts().sort_index(),
            'missing_by_dow': missing_agent['Start'].dt.day_name().value_counts(),
            'missing_by_shift_type': missing_agent['Shift Type'].value_counts()
        },
        
        'verification_status': {
            'missing_verified': missing_agent['Verified'].value_counts(),
            'has_agent_verified': has_agent['Verified'].value_counts()
        },
        
        'facility_patterns': {
            'facilities_missing': missing_agent['Facility ID'].value_counts(),
            'missing_rate_by_facility': (
                missing_agent.groupby('Facility ID').size() / 
                shifts_df.groupby('Facility ID').size()
            ).sort_values(ascending=False)
        },
        
        'charge_comparison': {
            'missing_charges': missing_agent['Charge'].describe(),
            'has_agent_charges': has_agent['Charge'].describe()
        }
    }
    
    print("=== Analysis of Shifts with Missing Agent IDs ===\n")
    print(f"Total Shifts: {len(shifts_df):,}")
    print(f"Shifts Missing Agent ID: {len(missing_agent):,} ({len(missing_agent)/len(shifts_df):.1%})")
    print(f"Shifts with Agent ID: {len(has_agent):,} ({len(has_agent)/len(shifts_df):.1%})")
    
    print("\n=== Verification Status ===")
    print("\nShifts Missing Agent ID:")
    print(analysis['verification_status']['missing_verified'])
    print("\nShifts with Agent ID:")
    print(analysis['verification_status']['has_agent_verified'])
    
    print("\n=== Shift Type Distribution (Missing Agent ID) ===")
    print(analysis['temporal_patterns']['missing_by_shift_type'])
    
    print("\n=== Top 5 Facilities with Missing Agent IDs ===")
    print("Count:")
    print(analysis['facility_patterns']['facilities_missing'].head())
    print("\nRate:")
    print(analysis['facility_patterns']['missing_rate_by_facility'].head())
    
    return analysis

"""This analysis should help us:

Identify patterns in missing Agent IDs
See if certain facilities have more missing IDs
Compare verification rates
Understand if missing IDs are random or systematic """
# Run the analysis
missing_agent_analysis = analyze_missing_agent_patterns(shifts_df)
```

    === Analysis of Shifts with Missing Agent IDs ===
    
    Total Shifts: 41,040
    Shifts Missing Agent ID: 20,035 (48.8%)
    Shifts with Agent ID: 21,005 (51.2%)
    
    === Verification Status ===
    
    Shifts Missing Agent ID:
    Verified
    False    19875
    True       160
    Name: count, dtype: int64
    
    Shifts with Agent ID:
    Verified
    True     16240
    False     4765
    Name: count, dtype: int64
    
    === Shift Type Distribution (Missing Agent ID) ===
    Shift Type
    pm        7318
    am        6971
    noc       4818
    custom     928
    Name: count, dtype: int64
    
    === Top 5 Facilities with Missing Agent IDs ===
    Count:
    Facility ID
    5f9ad22ae3a95f0016090f97    2087
    5f9c119522c5c50016d5b89e    1704
    5f9c169622c5c50016d5ba32    1476
    5c3cc2e3a2ae5c0016bef82c    1115
    5fa0a33daaa281001684ade2    1112
    Name: count, dtype: int64
    
    Rate:
    Facility ID
    61b396bdc6963001856bbda8   1.000
    5e727d32759cf60016dd70ca   1.000
    5e72825e759cf60016e11e5a   1.000
    6093f5c353844901664752b9   1.000
    5fa1853b6144d70016b72848   1.000
    dtype: float64
    


```python
# First, let's verify the data structure
overlap_start = shifts_df['Start'].dt.date.min()
overlap_end = shifts_df['Start'].dt.date.max()

# Print shapes and types for debugging
print("Overlap period:", overlap_start, "to", overlap_end)
print("\nDataset shapes:")
print(f"Shifts: {overlap_shifts.shape}")
print(f"Cancellations: {overlap_cancels.shape}")
print(f"Bookings: {overlap_bookings.shape}")

# Let's check a simpler version of the worker calculation first
def analyze_worker_basic(shifts, cancels):
    """Simplified version to debug the core calculation"""
    # Get workers with shifts
    workers = shifts[shifts['Agent ID'].notnull()]['Agent ID'].unique()
    
    # Create base metrics
    metrics = pd.DataFrame(index=workers)
    
    # Calculate basic stats
    shift_counts = shifts[shifts['Agent ID'].notnull()].groupby('Agent ID')['ID'].count()
    cancel_counts = cancels.groupby('Worker ID').size()
    
    # Ensure matching indices
    metrics['total_shifts'] = shift_counts
    metrics['cancellations'] = cancel_counts.reindex(workers).fillna(0)
    metrics['reliability'] = 1 - (metrics['cancellations'] / metrics['total_shifts'])
    
    return metrics

# Try the simplified version
test_metrics = analyze_worker_basic(overlap_shifts, overlap_cancels)
print("\nTest metrics head:")
print(test_metrics.head())
```

    Overlap period: 2021-10-01 to 2022-01-31
    
    Dataset shapes:
    Shifts: (41040, 16)
    Cancellations: (76316, 8)
    Bookings: (112770, 7)
    
    Test metrics head:
                              total_shifts  cancellations  reliability
    614627661afb050166fecd99            84         10.000        0.881
    60d5f4c8a9b88a0166aedaca            31          7.000        0.774
    5d7fb6319b671100167be1f1            49         13.000        0.735
    613503b78b28c60166060efe            56          9.000        0.839
    6099c93f2a957601669549c3           156         39.000        0.750
    


```python
def analyze_business_patterns(start_date='2021-10-01', end_date='2022-01-31'):
    """
    Analyze business patterns within the core overlapping period
    
    Parameters:
    - start_date: Beginning of analysis period
    - end_date: End of analysis period
    """
    # Filter to overlapping period
    mask_period = lambda df: (
        df['Created At'].dt.date >= pd.to_datetime(start_date).date() &
        df['Created At'].dt.date <= pd.to_datetime(end_date).date()
    )
    
    shifts_filtered = shifts_df[shifts_df['Start'].dt.date.between(start_date, end_date)]
    cancels_filtered = cancellations_df[mask_period(cancellations_df)]
    bookings_filtered = bookings_df[mask_period(bookings_df)]
    
    # Time-based patterns
    time_patterns = {
        'hourly_patterns': pd.DataFrame({
            'cancellations': cancels_filtered['Created At'].dt.hour.value_counts().sort_index(),
            'bookings': bookings_filtered['Created At'].dt.hour.value_counts().sort_index()
        }).fillna(0),
        
        'daily_patterns': pd.DataFrame({
            'shifts': shifts_filtered['Start'].dt.day_name().value_counts(),
            'cancellations': cancels_filtered['Created At'].dt.day_name().value_counts(),
            'bookings': bookings_filtered['Created At'].dt.day_name().value_counts()
        }).fillna(0),
        
        'lead_time_success': bookings_filtered[
            ~bookings_filtered['Shift ID'].isin(cancels_filtered['Shift ID'])
        ]['Lead Time'].describe()
    }
    
    # Worker reliability - separating by data completeness
    worker_patterns = {
        'complete_data': analyze_worker_patterns(
            shifts_filtered[shifts_filtered['Agent ID'].notnull()],
            cancels_filtered,
            bookings_filtered
        ),
        'missing_data': analyze_worker_patterns(
            shifts_filtered[shifts_filtered['Agent ID'].isnull()],
            cancels_filtered,
            bookings_filtered
        )
    }
    
    # Facility analysis
    facility_patterns = {
        'cancel_rates': calculate_facility_metrics(
            shifts_filtered, cancels_filtered, bookings_filtered
        ),
        'ncns_impact': analyze_ncns_impact(
            shifts_filtered, cancels_filtered
        )
    }
    
    return {
        'time_patterns': time_patterns,
        'worker_patterns': worker_patterns,
        'facility_patterns': facility_patterns
    }

def analyze_worker_patterns(shifts, cancels, bookings):
    """
    Analyze comprehensive booking and cancellation patterns at the worker level
    
    Parameters:
    -----------
    shifts : pd.DataFrame
        Shift data including verified status and worker information
    cancels : pd.DataFrame
        Cancellation data with timing and reason information
    bookings : pd.DataFrame
        Booking data with lead times and worker details
        
    Returns:
    -----------
    dict
        Dictionary containing detailed worker behavior analysis
    """
    # Start with valid workers only
    active_workers = shifts[shifts['Agent ID'].notnull()]['Agent ID'].unique()
    
    # Initialize success metrics DataFrame first
    success_metrics = pd.DataFrame(index=active_workers)
    
    # Basic counts
    shift_counts = shifts[shifts['Agent ID'].notnull()].groupby('Agent ID')['ID'].count()
    cancel_counts = cancels.groupby('Worker ID').size()
    ncns_counts = cancels[cancels['Action'] == 'NO_CALL_NO_SHOW'].groupby('Worker ID').size()
    
    # Add basic metrics with proper index alignment
    success_metrics['total_shifts'] = shift_counts
    success_metrics['cancellations'] = cancel_counts.reindex(active_workers).fillna(0)
    success_metrics['ncns'] = ncns_counts.reindex(active_workers).fillna(0)
    
    # Calculate derived metrics
    success_metrics['reliability'] = 1 - (success_metrics['cancellations'] / success_metrics['total_shifts'])
    success_metrics['ncns_rate'] = success_metrics['ncns'] / success_metrics['cancellations'].replace(0, 1)
    
    # Add verification rate
    verify_rate = shifts[shifts['Agent ID'].notnull()].groupby('Agent ID')['Verified'].mean()
    success_metrics['completion_rate'] = verify_rate
    
    # Add charge and time metrics
    charge_stats = shifts[shifts['Agent ID'].notnull()].groupby('Agent ID')['Charge'].agg(['mean', 'std'])
    time_stats = shifts[shifts['Agent ID'].notnull()].groupby('Agent ID')['Time'].agg(['mean', 'std'])
    
    success_metrics['avg_charge'] = charge_stats['mean']
    success_metrics['consistency'] = 1 - (time_stats['std'] / time_stats['mean'].replace(0, np.inf))
    
    # Pre-calculate datetime features for patterns
    bookings = bookings.copy()
    bookings['booking_hour'] = bookings['Created At'].dt.hour
    bookings['booking_day'] = bookings['Created At'].dt.day_name()
    
    # Analyze booking patterns
    booking_patterns = {
        'time_preferences': {
            'booking_hours': bookings.groupby('Worker ID')['booking_hour'].value_counts(),
            'booking_days': bookings.groupby('Worker ID')['booking_day'].value_counts(),
            'lead_time_stats': bookings.groupby('Worker ID')['Lead Time'].describe()
        },
        'shift_preferences': {
            'shift_types': shifts[shifts['Agent ID'].isin(active_workers)].groupby(
                ['Agent ID', 'Shift Type']).size().unstack(fill_value=0),
            'facility_choices': shifts[shifts['Agent ID'].isin(active_workers)].groupby(
                ['Agent ID', 'Facility ID']).size().unstack(fill_value=0)
        }
    }
    
    # Calculate overall score
    weights = {
        'completion_rate': 0.4,
        'reliability': 0.3,
        'consistency': 0.2,
        'avg_charge': 0.1
    }
    
    # Normalize any missing columns
    available_metrics = [m for m in weights.keys() if m in success_metrics.columns]
    weight_sum = sum(weights[m] for m in available_metrics)
    
    success_metrics['overall_score'] = sum(
        success_metrics[metric] * (weights[metric] / weight_sum)
        for metric in available_metrics
    )
    
    return {
        'success_metrics': success_metrics,
        'booking_patterns': booking_patterns
    }
"""Worker Profiles:

Creates a baseline profile for each worker using shift data
Captures basic metrics like total shifts, verification rates, and pricing patterns


Booking Patterns:

Analyzes when workers prefer to book shifts (time of day, day of week)
Examines lead time patterns
Identifies preferences for shift types and facilities


Cancellation Patterns:

Studies when cancellations typically occur
Calculates cancellation rates and no-show rates
Analyzes lead times for cancellations


Success Metrics:

Combines multiple factors into an overall worker score
Uses weighted metrics for completion, reliability, consistency, and earnings
Allows for customization of weights based on business priorities"""

def calculate_facility_metrics(shifts, cancels, bookings):
    """Calculate key facility metrics"""
    return {
        'cancel_rates': (cancels.groupby('Facility ID').size() / 
                        shifts.groupby('Facility ID').size()),
        'rebooking_success': calculate_rebooking_rates(shifts, cancels, bookings),
        'shift_fulfillment': calculate_fulfillment_rates(shifts, bookings)
    }


def calculate_worker_reliability(shifts, cancels):
    """
    Calculate reliability scores for workers based on their history
    
    Parameters:
    -----------
    shifts : pd.DataFrame
        Shift data with worker information
    cancels : pd.DataFrame
        Cancellation data
        
    Returns:
    -----------
    pd.DataFrame
        Worker reliability metrics
    """
    worker_metrics = pd.DataFrame()
    
    # Only analyze workers with valid IDs
    valid_workers = shifts[shifts['Agent ID'].notnull()]
    
    # Calculate basic metrics per worker
    worker_metrics = valid_workers.groupby('Agent ID').agg({
        'ID': 'count',  # Total shifts
        'Verified': 'mean',  # Verification rate
        'Charge': 'mean'  # Average charge rate
    }).rename(columns={
        'ID': 'total_shifts',
        'Verified': 'verification_rate',
        'Charge': 'avg_charge'
    })
    
    # Add cancellation metrics
    cancellation_rates = (
        cancels.groupby('Worker ID')
        .agg({
            'Shift ID': 'count',
            'Action': lambda x: (x == 'NO_CALL_NO_SHOW').mean()
        })
        .rename(columns={
            'Shift ID': 'cancellations',
            'Action': 'ncns_rate'
        })
    )
    
    worker_metrics = worker_metrics.join(
        cancellation_rates, 
        how='left'
    ).fillna(0)
    
    # Calculate reliability score (you can adjust the formula)
    worker_metrics['reliability_score'] = (
        worker_metrics['verification_rate'] * 0.4 +
        (1 - worker_metrics['ncns_rate']) * 0.4 +
        (1 - worker_metrics['cancellations']/worker_metrics['total_shifts']) * 0.2
    )
    
    return worker_metrics.sort_values('reliability_score', ascending=False)

def analyze_cancel_timing(cancels):
    """
    Analyze cancellation timing patterns including day/hour distribution,
    lead times, and seasonal patterns.
    
    Parameters:
    -----------
    cancels : pd.DataFrame
        Cancellation data with datetime columns and lead times
        
    Returns:
    -----------
    dict : Dictionary containing timing analysis results
    """
    timing_analysis = {
        # Time of day patterns
        'hourly_distribution': cancels['Created At'].dt.hour.value_counts().sort_index(),
        'daily_distribution': cancels['Created At'].dt.day_name().value_counts(),
        
        # Lead time analysis
        'lead_time_stats': cancels['Lead Time'].describe(),
        'lead_time_buckets': pd.cut(
            cancels['Lead Time'],
            bins=[-float('inf'), 0, 4, 24, 72, float('inf')],
            labels=['After Start', 'Under 4hrs', '4-24hrs', '1-3 days', 'Over 3 days']
        ).value_counts().sort_index(),
        
        # Action type by timing
        'timing_by_action': pd.crosstab(
            pd.cut(cancels['Lead Time'], 
                  bins=[-float('inf'), 0, 4, 24, 72, float('inf')],
                  labels=['After Start', 'Under 4hrs', '4-24hrs', '1-3 days', 'Over 3 days']),
            cancels['Action']
        )
    }
    
    return timing_analysis

def calculate_rebooking_rates(shifts, cancels, bookings):
    """
    Calculate how successfully cancelled shifts get rebooked
    
    Parameters:
    -----------
    shifts : pd.DataFrame
        Shift data
    cancels : pd.DataFrame
        Cancellation data
    bookings : pd.DataFrame
        Booking data
        
    Returns:
    -----------
    dict : Dictionary containing rebooking analysis
    """
    # Get cancelled shifts
    cancelled_shifts = cancels['Shift ID'].unique()
    
    # Look at subsequent bookings for cancelled shifts
    rebooking_analysis = {
        'total_cancellations': len(cancelled_shifts),
        'rebooked_count': sum(
            bookings['Shift ID'].isin(cancelled_shifts) &
            (bookings['Created At'] > cancels.groupby('Shift ID')['Created At'].first())
        ),
        'rebooking_lead_times': bookings[
            bookings['Shift ID'].isin(cancelled_shifts)
        ]['Lead Time'].describe(),
        
        # Facility level analysis
        'facility_rebooking_rates': pd.DataFrame({
            'cancellations': cancels.groupby('Facility ID').size(),
            'rebookings': bookings[
                bookings['Shift ID'].isin(cancelled_shifts)
            ].groupby('Facility ID').size()
        }).fillna(0)
    }
    
    # Calculate success rate
    rebooking_analysis['overall_rebooking_rate'] = (
        rebooking_analysis['rebooked_count'] / 
        rebooking_analysis['total_cancellations']
    )
    
    return rebooking_analysis

def calculate_fulfillment_rates(shifts, bookings):
    """
    Calculate shift fulfillment rates and patterns
    
    Parameters:
    -----------
    shifts : pd.DataFrame
        Shift data including verification status
    bookings : pd.DataFrame
        Booking data
        
    Returns:
    -----------
    dict : Dictionary containing fulfillment analysis
    """
    fulfillment_analysis = {
        # Overall fulfillment
        'total_shifts': len(shifts),
        'booked_shifts': len(shifts[shifts['Agent ID'].notnull()]),
        'verified_shifts': shifts['Verified'].sum(),
        
        # Fulfillment by type
        'fulfillment_by_type': pd.DataFrame({
            'total': shifts.groupby('Shift Type').size(),
            'booked': shifts[shifts['Agent ID'].notnull()].groupby('Shift Type').size(),
            'verified': shifts[shifts['Verified']].groupby('Shift Type').size()
        }).fillna(0),
        
        # Fulfillment by role
        'fulfillment_by_role': pd.DataFrame({
            'total': shifts.groupby('Agent Req').size(),
            'booked': shifts[shifts['Agent ID'].notnull()].groupby('Agent Req').size(),
            'verified': shifts[shifts['Verified']].groupby('Agent Req').size()
        }).fillna(0)
    }
    
    # Calculate rates
    fulfillment_analysis['overall_booking_rate'] = (
        fulfillment_analysis['booked_shifts'] / 
        fulfillment_analysis['total_shifts']
    )
    fulfillment_analysis['overall_verification_rate'] = (
        fulfillment_analysis['verified_shifts'] / 
        fulfillment_analysis['total_shifts']
    )
    
    return fulfillment_analysis
```


```python
# Run the analysis
worker_analysis = analyze_worker_patterns(
    overlap_shifts,
    overlap_cancels,
    overlap_bookings
)

# Display results
print("\n=== Worker Analysis ===")
print("\nMetrics Summary:")
print(worker_analysis['success_metrics'].describe())

print("\nTop 5 Most Reliable Workers:")
print(worker_analysis['success_metrics'].nlargest(5, 'reliability')[
    ['total_shifts', 'cancellations', 'reliability', 'completion_rate']
])
```

    
    === Worker Analysis ===
    
    Metrics Summary:
           total_shifts  cancellations    ncns  reliability  ncns_rate  \
    count       997.000        997.000 997.000      997.000    997.000   
    mean         21.068          7.958   1.228        0.344      0.179   
    std          25.057         12.928   2.047        1.609      0.283   
    min           1.000          0.000   0.000      -31.000      0.000   
    25%           3.000          1.000   0.000        0.300      0.000   
    50%          11.000          4.000   0.000        0.727      0.000   
    75%          30.000          9.000   2.000        0.917      0.250   
    max         156.000        111.000  16.000        1.000      1.000   
    
           completion_rate  avg_charge  consistency  overall_score  
    count          997.000     997.000      870.000        870.000  
    mean             0.742      36.546        0.599          4.187  
    std              0.259      14.442        0.207          1.428  
    min              0.000       0.000       -0.121         -1.450  
    25%              0.643      28.146        0.478          3.399  
    50%              0.795      34.167        0.594          4.004  
    75%              0.945      44.275        0.717          4.969  
    max              1.000      87.000        1.000          8.920  
    
    Top 5 Most Reliable Workers:
                              total_shifts  cancellations  reliability  \
    614d2bace60669018587ffd5             5          0.000        1.000   
    614ebc1db03c7c0187bc2ed6             3          0.000        1.000   
    5d3f26fadde430001692153b            15          0.000        1.000   
    60621f61cc38d30016fb3fd5             3          0.000        1.000   
    5fe36e208c148400162895f3            12          0.000        1.000   
    
                              completion_rate  
    614d2bace60669018587ffd5            1.000  
    614ebc1db03c7c0187bc2ed6            0.333  
    5d3f26fadde430001692153b            0.800  
    60621f61cc38d30016fb3fd5            0.667  
    5fe36e208c148400162895f3            1.000  
    


```python
# Example usage
overlap_start = shifts_df['Start'].dt.date.min()
overlap_end = shifts_df['Start'].dt.date.max()

# Filter data to overlapping period
overlap_shifts = shifts_df[shifts_df['Start'].dt.date.between(overlap_start, overlap_end)]
overlap_cancels = cancellations_df[
    cancellations_df['Created At'].dt.date.between(overlap_start, overlap_end)
]
overlap_bookings = bookings_df[
    bookings_df['Created At'].dt.date.between(overlap_start, overlap_end)
]

# Run the analysis
worker_analysis = analyze_worker_patterns(
    overlap_shifts,
    overlap_cancels,
    overlap_bookings
)

# Display summary results
print("\n=== Worker Pattern Analysis ===")
print("\nTop Performing Workers:")
print(worker_analysis['success_metrics'].nlargest(5, 'overall_score'))

print("\nBooking Time Preferences (Top 3 Hours):")
print(worker_analysis['booking_patterns']['time_preferences']['booking_hours']
      .groupby(level=0).nlargest(3))
```

    
    === Worker Pattern Analysis ===
    
    Top Performing Workers:
                              total_shifts  cancellations  ncns  reliability  \
    61ed847f85f245018b18689b             4          0.000 0.000        1.000   
    61ded3788a1b8f018bc396b9             2          1.000 0.000        0.500   
    619ea1b00a49580186317efc             2          3.000 2.000       -0.500   
    618e58ef0aa60a0185e5f9e3             2          2.000 0.000        0.000   
    619e67041c194c0185ae4000             2          0.000 0.000        1.000   
    
                              ncns_rate  completion_rate  avg_charge  consistency  \
    61ed847f85f245018b18689b      0.000            1.000      80.500        0.851   
    61ded3788a1b8f018bc396b9      0.000            1.000      80.000        0.999   
    619ea1b00a49580186317efc      0.667            0.500      84.500        0.798   
    618e58ef0aa60a0185e5f9e3      0.000            1.000      78.000        0.999   
    619e67041c194c0185ae4000      0.000            1.000      75.000        0.933   
    
                              overall_score  
    61ed847f85f245018b18689b          8.920  
    61ded3788a1b8f018bc396b9          8.750  
    619ea1b00a49580186317efc          8.660  
    618e58ef0aa60a0185e5f9e3          8.400  
    619e67041c194c0185ae4000          8.387  
    
    Booking Time Preferences (Top 3 Hours):
    Worker ID                 Worker ID                 booking_hour
    5b4630b61549700014f96191  5b4630b61549700014f96191  1               1
    5b5635b34afc8f0014a40433  5b5635b34afc8f0014a40433  4               6
                                                        9               3
                                                        10              2
    5b70f386740bd10014f18a0d  5b70f386740bd10014f18a0d  23              3
                                                                       ..
    619f018645d2120186b285ce  619f018645d2120186b285ce  16              2
                                                        5               1
                                                        7               1
    61a131290ff19f018773973b  61a131290ff19f018773973b  16              1
    61a3ea9362cee601850a15b8  61a3ea9362cee601850a15b8  18              1
    Name: count, Length: 16924, dtype: int64
    

Let's think critically about this:

Understanding the Data Context
From the proposal:


They explicitly say booking data is a subset
They state it's "OK" because it's meant to show booking behavior patterns
However, they don't mention if cancellation data is complete or a subset
The shifts data appears to be the "source of truth" (Oct 2021 - Jan 2022)


Analysis Implications
We should split our analysis into two categories:

A. Full Period Analysis (Using Shifts Data)


```python
"""Lets think critically about this:

Understanding the Data Context
From the proposal:


They explicitly say booking data is a subset
They state it's "OK" because it's meant to show booking behavior patterns
However, they don't mention if cancellation data is complete or a subset
The shifts data appears to be the "source of truth" (Oct 2021 - Jan 2022)


Analysis Implications
We should split our analysis into two categories: """
# A. Full Period Analysis (Using Shifts Data)
#  B: Behavioral Analysis (Using Overlap Period)

"""The key insight is that we should:

Use shifts data for absolute metrics
Use overlap periods for behavioral analysis
Be clear about limitations in our findings
Focus on patterns rather than absolute numbers for booking/cancellation behavior

This matches their intent while making the best use of available data."""
```




    'The key insight is that we should:\n\nUse shifts data for absolute metrics\nUse overlap periods for behavioral analysis\nBe clear about limitations in our findings\nFocus on patterns rather than absolute numbers for booking/cancellation behavior\n\nThis matches their intent while making the best use of available data.'




```python


# A. Full Period Analysis (Using Shifts Data)
def analyze_shifts_complete():
    """
    Analyze the complete shifts dataset for overall marketplace health
    
    Note: This analysis uses only the shifts dataset which appears to be 
    complete for Oct 2021 - Jan 2022.
    """
    shifts_analysis = {
        # Basic marketplace metrics
        'total_shifts': len(shifts_df),
        'shifts_by_type': shifts_df['Shift Type'].value_counts(),
        'verification_rate': shifts_df['Verified'].mean(),
        
        # Financial metrics
        'charge_patterns': shifts_df.groupby('Agent Req')['Charge'].describe(),
        
        # Time patterns
        'shift_distribution': {
            'by_day': shifts_df['Start'].dt.day_name().value_counts(),
            'by_hour': shifts_df['Start'].dt.hour.value_counts().sort_index()
        },
        
        # Facility metrics
        'facility_patterns': shifts_df.groupby('Facility ID').agg({
            'ID': 'count',
            'Verified': 'mean',
            'Charge': 'mean'
        }).rename(columns={'ID': 'total_shifts'})
    }
    return shifts_analysis
```


```python
#  B: Behavioral Analysis (Using Overlap Period)
def analyze_booking_behavior(start_date=None, end_date=None):
    """
    Analyze HCP booking and cancellation behavior where we have all datasets
    
    Notes:
    - This analysis uses the period where we have overlapping data
    - Focus is on understanding behavioral patterns rather than absolute numbers
    """
    # Filter to overlap period if dates provided
    if start_date and end_date:
        shifts_subset = shifts_df[shifts_df['Start'].dt.date.between(start_date, end_date)]
        cancels_subset = cancellations_df[
            cancellations_df['Created At'].dt.date.between(start_date, end_date)
        ]
        bookings_subset = bookings_df[
            bookings_df['Created At'].dt.date.between(start_date, end_date)
        ]
    else:
        shifts_subset = shifts_df
        cancels_subset = cancellations_df
        bookings_subset = bookings_df
    
    # Cross-reference data
    shifts_with_outcomes = shifts_subset.copy()
    shifts_with_outcomes['was_booked'] = shifts_subset['ID'].isin(bookings_subset['Shift ID'])
    shifts_with_outcomes['was_cancelled'] = shifts_subset['ID'].isin(cancels_subset['Shift ID'])
    
    behavior_analysis = {
        # Booking patterns
        'booking_behavior': {
            'lead_times': bookings_subset['Lead Time'].describe(),
            'booking_times': bookings_subset['Created At'].dt.hour.value_counts().sort_index()
        },
        
        # Cancellation patterns
        'cancellation_behavior': {
            'cancel_types': cancels_subset['Action'].value_counts(),
            'lead_times': cancels_subset['Lead Time'].describe(),
            'cancel_times': cancels_subset['Created At'].dt.hour.value_counts().sort_index()
        },
        
        # Shift outcomes
        'shift_outcomes': {
            'total_shifts': len(shifts_subset),
            'booked_count': shifts_with_outcomes['was_booked'].sum(),
            'cancelled_count': shifts_with_outcomes['was_cancelled'].sum(),
            'booking_rate': shifts_with_outcomes['was_booked'].mean(),
            'cancellation_rate': shifts_with_outcomes['was_cancelled'].mean()
        }
    }
    return behavior_analysis
```


```python
# For Missing Agent IDs Let's cross-check with both datasets:


def analyze_missing_agents_behavior():
    """
    Analyze what happens to shifts with missing Agent IDs
    """
    # Get shifts with/without agents
    missing_agent = shifts_df[shifts_df['Agent ID'].isnull()]
    has_agent = shifts_df[shifts_df['Agent ID'].notnull()]
    
    # Cross reference with bookings and cancellations
    missing_outcomes = {
        'booked': missing_agent['ID'].isin(bookings_df['Shift ID']).mean(),
        'cancelled': missing_agent['ID'].isin(cancellations_df['Shift ID']).mean(),
        'verified': missing_agent['Verified'].mean()
    }
    
    has_agent_outcomes = {
        'booked': has_agent['ID'].isin(bookings_df['Shift ID']).mean(),
        'cancelled': has_agent['ID'].isin(cancellations_df['Shift ID']).mean(),
        'verified': has_agent['Verified'].mean()
    }
    
    return {
        'missing_agent_outcomes': missing_outcomes,
        'has_agent_outcomes': has_agent_outcomes
    }


```


```python
# Run all analyses
print("Running comprehensive analyses...")

# 1. Full Shifts Analysis
print("\n=== COMPLETE SHIFTS ANALYSIS (Oct 2021 - Jan 2022) ===")
shifts_analysis = analyze_shifts_complete()
print("\nBasic Marketplace Metrics:")
print(f"Total Shifts: {shifts_analysis['total_shifts']:,}")
print("\nShift Types:")
print(shifts_analysis['shifts_by_type'])
print(f"\nOverall Verification Rate: {shifts_analysis['verification_rate']:.2%}")

# 2. Behavioral Analysis 
# Using the overlap period (focusing on patterns rather than absolute numbers)
print("\n=== BEHAVIORAL ANALYSIS (Overlap Period) ===")
start_date = shifts_df['Start'].dt.date.min()  # Oct 1, 2021
end_date = shifts_df['Start'].dt.date.max()    # Jan 31, 2022
behavior_analysis = analyze_booking_behavior(start_date, end_date)

print("\nBooking Patterns:")
print("Lead Times (hours):")
print(behavior_analysis['booking_behavior']['lead_times'])

print("\nCancellation Types:")
print(behavior_analysis['cancellation_behavior']['cancel_types'])

print("\nShift Outcomes:")
for metric, value in behavior_analysis['shift_outcomes'].items():
    if 'rate' in metric:
        print(f"{metric}: {value:.2%}")
    else:
        print(f"{metric}: {value:,}")

# 3. Missing Agent ID Analysis
print("\n=== MISSING AGENT ID ANALYSIS ===")
agent_behavior = analyze_missing_agents_behavior()

print("\nShifts with Missing Agent IDs:")
for metric, value in agent_behavior['missing_agent_outcomes'].items():
    print(f"{metric}: {value:.2%}")

print("\nShifts with Agent IDs:")
for metric, value in agent_behavior['has_agent_outcomes'].items():
    print(f"{metric}: {value:.2%}")

# Store results in summary object for later use
summary.add_summary('complete_analysis', 'shifts', shifts_analysis)
summary.add_summary('complete_analysis', 'behavior', behavior_analysis)
summary.add_summary('complete_analysis', 'missing_agents', agent_behavior)


"""
This code:

Analyzes the complete shifts dataset first
Looks at booking/cancellation behavior in the overlap period
Specifically examines shifts with/without Agent IDs
Stores all results in our summary object

The output will help us understand:

Overall marketplace metrics from shifts data
Behavioral patterns where we have complete data
What missing Agent IDs might mean

Each section is clearly labeled, and results are formatted for easy reading. We can use these results to:

Identify key patterns
Support our findings
Guide additional analysis
"""

```

    Running comprehensive analyses...
    
    === COMPLETE SHIFTS ANALYSIS (Oct 2021 - Jan 2022) ===
    
    Basic Marketplace Metrics:
    Total Shifts: 41,040
    
    Shift Types:
    Shift Type
    pm        14232
    am        13951
    noc       11446
    custom     1411
    Name: count, dtype: int64
    
    Overall Verification Rate: 39.96%
    
    === BEHAVIORAL ANALYSIS (Overlap Period) ===
    
    Booking Patterns:
    Lead Times (hours):
    count   112770.000
    mean       182.633
    std        181.176
    min      -1425.318
    25%         53.497
    50%        135.142
    75%        255.955
    max       1267.915
    Name: Lead Time, dtype: float64
    
    Cancellation Types:
    Action
    WORKER_CANCEL      64893
    NO_CALL_NO_SHOW    11423
    Name: count, dtype: int64
    
    Shift Outcomes:
    total_shifts: 41,040
    booked_count: 8,292
    cancelled_count: 6,358
    booking_rate: 20.20%
    cancellation_rate: 15.49%
    
    === MISSING AGENT ID ANALYSIS ===
    
    Shifts with Missing Agent IDs:
    booked: 7.75%
    cancelled: 17.77%
    verified: 0.80%
    
    Shifts with Agent IDs:
    booked: 37.30%
    cancelled: 14.16%
    verified: 77.31%
    




    '\nThis code:\n\nAnalyzes the complete shifts dataset first\nLooks at booking/cancellation behavior in the overlap period\nSpecifically examines shifts with/without Agent IDs\nStores all results in our summary object\n\nThe output will help us understand:\n\nOverall marketplace metrics from shifts data\nBehavioral patterns where we have complete data\nWhat missing Agent IDs might mean\n\nEach section is clearly labeled, and results are formatted for easy reading. We can use these results to:\n\nIdentify key patterns\nSupport our findings\nGuide additional analysis\n'




```python
# PHASE 1: SUCCESS PATH ANALYSIS
#A. Define and Validate Success Metrics
#Deep dive into successful shifts 

class ShiftSuccessAnalysis:
    """
    Analyzes the complete lifecycle of shifts from posting to completion.
    
    Core metrics tracked:
    - Booking success: Did the shift get booked?
    - Retention success: Did the booking stick (no cancellation)?
    - Completion success: Was the shift verified as worked?
    """
    
    def __init__(self, shifts_df, bookings_df, cancellations_df):
        """Initialize with our three core datasets."""
        self.shifts_df = shifts_df.copy()
        self.bookings_df = bookings_df.copy()
        self.cancellations_df = cancellations_df.copy()
        self.success_journey = None
        
        # Verify data compatibility
        self._validate_data()
        
        # Create enhanced dataset
        self._create_success_journey()
    
    def _validate_data(self):
        """
        Ensure data quality and compatibility across datasets.
        """
        # Check required columns
        required_columns = {
            'shifts': ['ID', 'Start', 'End', 'Verified', 'Agent ID', 
                      'Facility ID', 'Agent Req', 'Shift Type', 'Charge'],
            'bookings': ['Shift ID', 'Created At', 'Worker ID'],
            'cancellations': ['Shift ID', 'Created At', 'Action', 'Lead Time']
        }
        
        for df_name, columns in required_columns.items():
            df = getattr(self, f"{df_name}_df")
            missing_cols = [col for col in columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing columns in {df_name}: {missing_cols}")
        
        # Print coverage analysis
        self._analyze_coverage()
    
    def _analyze_coverage(self):
        """Analyze data coverage and overlap."""
        shifts_ids = set(self.shifts_df['ID'])
        booking_ids = set(self.bookings_df['Shift ID'])
        cancel_ids = set(self.cancellations_df['Shift ID'])
        
        print("\n=== Data Coverage Analysis ===")
        print(f"\nTotal Shifts: {len(shifts_ids):,}")
        print(f"Shifts with Bookings: {len(shifts_ids & booking_ids):,} "
              f"({len(shifts_ids & booking_ids)/len(shifts_ids):.1%})")
        print(f"Shifts with Cancellations: {len(shifts_ids & cancel_ids):,} "
              f"({len(shifts_ids & cancel_ids)/len(shifts_ids):.1%})")
        
        # Analyze potential data quality issues
        orphaned_bookings = len(booking_ids - shifts_ids)
        orphaned_cancels = len(cancel_ids - shifts_ids)
        
        if orphaned_bookings or orphaned_cancels:
            print("\nPotential Data Quality Issues:")
            print(f"Orphaned Bookings: {orphaned_bookings:,}")
            print(f"Orphaned Cancellations: {orphaned_cancels:,}")
    
    def _create_success_journey(self):
        """
        Creates enhanced dataset tracking complete shift lifecycle.
        """
        journey = self.shifts_df.copy()
        
        # Add booking information
        booking_times = self.bookings_df.groupby('Shift ID').agg({
            'Created At': ['first', 'count']
        }).reset_index()
        booking_times.columns = ['Shift ID', 'First Booking', 'Booking Count']
        
        journey = journey.merge(
            booking_times, 
            left_on='ID', 
            right_on='Shift ID', 
            how='left'
        )
        
        # Add cancellation information
        cancel_info = self.cancellations_df.groupby('Shift ID').agg({
            'Created At': 'first',
            'Action': 'first',
            'Lead Time': 'first'
        }).reset_index()
        
        journey = journey.merge(
            cancel_info,
            left_on='ID',
            right_on='Shift ID',
            how='left',
            suffixes=('_booking', '_cancel')
        )
        
        # Calculate success metrics
        journey['was_booked'] = journey['First Booking'].notnull()
        journey['was_cancelled'] = journey['Created At_cancel'].notnull()
        journey['was_completed'] = journey['Verified']
        
        # Calculate time to shift start (from booking)
        journey['lead_time'] = (
            journey['Start'] - journey['First Booking']
        ).dt.total_seconds() / 3600  # Convert to hours
        
        self.success_journey = journey
        
        # Print initial success metrics
        self._print_success_metrics()
    
    def _print_success_metrics(self):
        """Print key success metrics from the journey data."""
        metrics = self.success_journey.agg({
            'was_booked': 'mean',
            'was_cancelled': 'mean',
            'was_completed': 'mean'
        })
        
        print("\n=== Success Metrics ===")
        print(f"Booking Rate: {metrics['was_booked']:.1%}")
        print(f"Cancellation Rate: {metrics['was_cancelled']:.1%}")
        print(f"Completion Rate: {metrics['was_completed']:.1%}")
    
    def analyze_verification_discrepancy(self):
        """
        Investigates why shifts might be verified without appearing in booking logs.
        """
        verified_shifts = self.success_journey[self.success_journey['Verified']]
        unbooked_verified = verified_shifts[~verified_shifts['was_booked']]
        
        results = {
            'overview': {
                'total_shifts': len(self.success_journey),
                'verified_shifts': len(verified_shifts),
                'unbooked_verified': len(unbooked_verified),
                'verification_rate': len(verified_shifts) / len(self.success_journey),
                'unbooked_verified_rate': len(unbooked_verified) / len(verified_shifts)
            },
            'unbooked_verified_patterns': {
                'by_role': unbooked_verified['Agent Req'].value_counts(),
                'by_shift_type': unbooked_verified['Shift Type'].value_counts(),
                'by_facility': unbooked_verified['Facility ID'].value_counts().head()
            }
        }
        
        agent_patterns = self.success_journey.groupby(
            self.success_journey['Agent ID'].isnull()
        ).agg({
            'was_booked': 'mean',
            'Verified': 'mean',
            'was_cancelled': 'mean'
        }).round(3)
        
        results['agent_id_patterns'] = agent_patterns
        
        return results
    
    def get_success_patterns(self):
        """
        Analyzes patterns in successfully completed shifts.
        """
        successful = self.success_journey[
            (self.success_journey['Verified']) & 
            (~self.success_journey['was_cancelled'])
        ]
        
        patterns = {
            'timing': {
                'hour_distribution': successful['Start'].dt.hour.value_counts().sort_index(),
                'day_distribution': successful['Start'].dt.day_name().value_counts(),
                'lead_times': successful['lead_time'].describe()
            },
            'characteristics': {
                'role_distribution': successful['Agent Req'].value_counts(),
                'shift_types': successful['Shift Type'].value_counts(),
                'charge_rates': successful.groupby('Agent Req')['Charge'].agg(['mean', 'std'])
            },
            'facility_patterns': {
                'success_rates': (
                    self.success_journey.groupby('Facility ID')['Verified'].agg(['mean', 'count'])
                    .sort_values('mean', ascending=False)
                    .query('count >= 10')  # Only facilities with sufficient data
                )
            }
        }
        
        return patterns
```


```python
# Run the analysis
analyzer = ShiftSuccessAnalysis(shifts_df, bookings_df, cancellations_df)

# Analyze verification discrepancy
discrepancy_results = analyzer.analyze_verification_discrepancy()

print("\n=== Verification Discrepancy Analysis ===")
print("\nOverview:")
for metric, value in discrepancy_results['overview'].items():
    if 'rate' in metric:
        print(f"{metric}: {value:.1%}")
    else:
        print(f"{metric}: {value:,}")

print("\nUnbooked Verified Shifts by Role:")
print(discrepancy_results['unbooked_verified_patterns']['by_role'])

print("\nAgent ID Impact:")
print(discrepancy_results['agent_id_patterns'])

# Get success patterns
success_patterns = analyzer.get_success_patterns()

print("\n=== Success Patterns ===")
print("\nMost Successful Shift Types:")
print(success_patterns['characteristics']['shift_types'])

print("\nAverage Charge Rates for Successful Shifts:")
print(success_patterns['characteristics']['charge_rates'])

print("\nTop 5 Facilities by Success Rate (min 10 shifts):")
print(success_patterns['facility_patterns']['success_rates'].head())
```

    
    === Data Coverage Analysis ===
    
    Total Shifts: 41,040
    Shifts with Bookings: 9,387 (22.9%)
    Shifts with Cancellations: 6,535 (15.9%)
    
    Potential Data Quality Issues:
    Orphaned Bookings: 99,275
    Orphaned Cancellations: 59,027
    
    === Success Metrics ===
    Booking Rate: 22.9%
    Cancellation Rate: 15.9%
    Completion Rate: 40.0%
    
    === Verification Discrepancy Analysis ===
    
    Overview:
    total_shifts: 41,040
    verified_shifts: 16,400
    unbooked_verified: 10,283
    verification_rate: 40.0%
    unbooked_verified_rate: 62.7%
    
    Unbooked Verified Shifts by Role:
    Agent Req
    CNA          7815
    LVN          1913
    RN            549
    CAREGIVER       3
    NURSE           3
    Name: count, dtype: int64
    
    Agent ID Impact:
              was_booked  Verified  was_cancelled
    Agent ID                                     
    False          0.373     0.773          0.142
    True           0.077     0.008          0.178
    
    === Success Patterns ===
    
    Most Successful Shift Types:
    Shift Type
    am        4688
    pm        4630
    noc       4280
    custom     301
    Name: count, dtype: int64
    
    Average Charge Rates for Successful Shifts:
                mean   std
    Agent Req             
    CAREGIVER 31.000 8.544
    CNA       35.174 7.315
    LVN       52.317 9.448
    RN        67.642 8.344
    
    Top 5 Facilities by Success Rate (min 10 shifts):
                              mean  count
    Facility ID                          
    61cac3f8e9cdd0018a278531 0.782    133
    61c3ad9db706a101842fb154 0.690     29
    5f9b189a7ecb880016516a52 0.650   1033
    61422c7f81f2950166d5f881 0.622   1126
    6064aa481c4bf40016be967a 0.596    592
    

# Choosing a path: backup pool 


```python
# 1. Backup Pool Estimation
# The goal is to calculate the size of the backup pool needed to cover 75% of late cancellations.
# Backup Pool Estimation

# Assumptions
BACKUP_COVERAGE_TARGET = 0.75  # Cover 75% of late cancellations

def estimate_backup_pool(shifts_df, cancellations_df):
    """
    Estimate the size of a backup pool needed to cover late cancellations.
    
    Parameters:
    - shifts_df (pd.DataFrame): Shift data.
    - cancellations_df (pd.DataFrame): Cancellations data with lead times.
    
    Returns:
    - Estimated pool size needed for target coverage.
    - Contextual insights into late cancellations.
    """
    # Step 1: Focus on Late Cancellations (<4 hours)
    late_cancellations = cancellations_df[
        cancellations_df['Lead Time'] < 4
    ]
    total_late_cancels = len(late_cancellations)
    
    print("=== Backup Pool Estimation for Late Cancellations ===")
    print(f"Total Late Cancellations (<4hrs): {total_late_cancels:,}")
    
    # Step 2: Estimate coverage required
    target_coverage = int(total_late_cancels * BACKUP_COVERAGE_TARGET)
    print(f"Target Coverage (75%): {target_coverage:,} shifts")

    # Step 3: Calculate HCP Availability and Estimate Pool Size
    late_cancel_hcps = late_cancellations['Worker ID'].value_counts()
    avg_shifts_per_hcp = late_cancel_hcps.mean()
    
    if avg_shifts_per_hcp > 0:
        pool_size = int(np.ceil(target_coverage / avg_shifts_per_hcp))
    else:
        pool_size = 0
    
    print(f"Average Late Cancellations per HCP: {avg_shifts_per_hcp:.2f}")
    print(f"Estimated Backup Pool Size: {pool_size} HCPs")
    
    print("\nContext:")
    print("To meet 75% coverage of late cancellations, we estimate needing a pool of pre-vetted,")
    print(f"reliable HCPs who can cover approximately {avg_shifts_per_hcp:.2f} late cancellations on average.")
    print("This estimate assumes that reliable HCPs are distributed evenly across cancellations.")
    
    return pool_size

def identify_reliable_hcps(bookings_df, cancellations_df, threshold=0.1):
    """
    Identify reliable HCPs with cancellation rates below a given threshold.
    
    Parameters:
    - bookings_df (pd.DataFrame): Booking logs with Worker IDs.
    - cancellations_df (pd.DataFrame): Cancellations data with Worker IDs.
    - threshold (float): Maximum cancellation rate for reliability.
    
    Returns:
    - Reliable HCPs as a DataFrame.
    """
    print("\n=== Reliable HCP Identification ===")
    
    # Step 1: Calculate Total Shifts and Cancellations per Worker
    total_shifts = bookings_df['Worker ID'].value_counts()
    total_cancellations = cancellations_df['Worker ID'].value_counts()
    
    # Step 2: Calculate Cancellation Rate
    reliability_df = pd.DataFrame({
        'Total Shifts': total_shifts,
        'Cancellations': total_cancellations
    }).fillna(0)
    reliability_df['Cancellation Rate'] = reliability_df['Cancellations'] / reliability_df['Total Shifts']
    
    # Step 3: Identify Reliable Workers
    reliable_hcps = reliability_df[reliability_df['Cancellation Rate'] <= threshold]
    reliable_hcps_sorted = reliable_hcps.sort_values(by='Cancellation Rate')
    
    print(f"Total Workers Analyzed: {len(reliability_df):,}")
    print(f"Workers with Cancellation Rate ≤ {threshold*100:.0f}%: {len(reliable_hcps):,}")
    print("\nTop 5 Most Reliable Workers:")
    print(reliable_hcps_sorted.head())
    
    print("\nContext:")
    print("Reliable HCPs are defined as those with a cancellation rate ≤ 10%.")
    print("This pool represents our most dependable workers, making them ideal candidates")
    print("for participation in the backup program. They are prioritized based on:")
    print("1. Total shifts worked.")
    print("2. Low cancellation counts.")
    
    return reliable_hcps_sorted

# Run the analyses
backup_pool_size = estimate_backup_pool(shifts_df, cancellations_df)
reliable_hcps = identify_reliable_hcps(bookings_df, cancellations_df)

# Display final summary
print("\n=== Summary for WBD ===")
print(f"Estimated Backup Pool Size (75% Late Cancel Coverage): {backup_pool_size} HCPs")
print(f"Reliable HCPs Identified (Cancellation Rate ≤ 10%): {len(reliable_hcps)} workers")



```

    === Backup Pool Estimation for Late Cancellations ===
    Total Late Cancellations (<4hrs): 23,068
    Target Coverage (75%): 17,301 shifts
    Average Late Cancellations per HCP: 3.36
    Estimated Backup Pool Size: 5154 HCPs
    
    Context:
    To meet 75% coverage of late cancellations, we estimate needing a pool of pre-vetted,
    reliable HCPs who can cover approximately 3.36 late cancellations on average.
    This estimate assumes that reliable HCPs are distributed evenly across cancellations.
    
    === Reliable HCP Identification ===
    Total Workers Analyzed: 10,915
    Workers with Cancellation Rate ≤ 10%: 1,343
    
    Top 5 Most Reliable Workers:
                              Total Shifts  Cancellations  Cancellation Rate
    Worker ID                                                               
    5bebb6bf19a24e000424b1a0         3.000          0.000              0.000
    61315e6d2ae392016696e305        40.000          0.000              0.000
    61315e251b06be01660d5c03        40.000          0.000              0.000
    6131539303cefc0166c2aee3        39.000          0.000              0.000
    6131354bda6f5301669f032f         1.000          0.000              0.000
    
    Context:
    Reliable HCPs are defined as those with a cancellation rate ≤ 10%.
    This pool represents our most dependable workers, making them ideal candidates
    for participation in the backup program. They are prioritized based on:
    1. Total shifts worked.
    2. Low cancellation counts.
    
    === Summary for WBD ===
    Estimated Backup Pool Size (75% Late Cancel Coverage): 5154 HCPs
    Reliable HCPs Identified (Cancellation Rate ≤ 10%): 1343 workers
    


```python

```


```python

```
