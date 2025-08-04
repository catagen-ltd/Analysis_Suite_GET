# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 11:07:08 2021

@author: Home
"""

import pandas 
import numpy as np
import sys
import os

# Add progress indication and suppress potential pandas output
import warnings
warnings.filterwarnings('ignore')
#import matplotlib.pyplot as plt


def load_data_with_column_padding(filename, expected_columns=78):
    """
    Load data ensuring all rows have expected number of columns
    """
    print(f"Loading {filename} with column padding...")
    
    # Read all lines and pad as needed
    padded_data = []
    
    with open(filename, 'r') as file:
        for line_num, line in enumerate(file):
            columns = line.strip().split()
            
            # Pad with zeros if needed
            while len(columns) < expected_columns:
                columns.append('0')
            
            padded_data.append(columns)
    
    # Convert to DataFrame
    df = pandas.DataFrame(padded_data)
    
    # Select only the columns we need and assign names
    selected_cols = [0,1,4,5,9,10,11,27,44,47,49,50,52,53,54,55,58,59,60,61,65,66,67]
    df_selected = df.iloc[:, selected_cols].copy()
    
    # Assign column names
    cols=['Date','Time','lambda_1','lambda_2','Inlet_temp','Bed_temp','Other_temp',"In_test",'CO_1','O2_1','HC_1','NO_1','CO_2','O2_2','HC_2','NO_2','CO_3','O2_3','HC_3','NO_3','MKS_HC','MKS_CO','MKS_NO']
    df_selected.columns = cols
    
    # Convert numeric columns, treating padded zeros and blanks as NaN
    numeric_cols = cols[2:]  # Skip Date and Time
    for col in numeric_cols:
        df_selected[col] = pandas.to_numeric(df_selected[col], errors='coerce')
        # Convert our padded zeros back to NaN (they represent missing data)
        df_selected[col] = df_selected[col].replace(0, np.nan)
    
    return df_selected

def validate_and_clean_mks_data(data_frame):
    """
    Validate and clean MKS analyzer data to handle connection drops and power estimation values
    Only MKS_NO column (column 78) can have power estimation data like 54321
    Only convert zeros to NaN in MKS_NO column - valid zeros in CO/HC should remain
    
    Args:
        data_frame: Pandas DataFrame containing the data
        
    Returns:
        Pandas DataFrame with cleaned MKS data
    """
    # Create a copy to avoid modifying original data
    cleaned_data = data_frame.copy()
    
    # Define power estimation threshold - values like 54321 indicate power estimation mode
    POWER_ESTIMATION_THRESHOLD = 10000  # Any value above this is likely power estimation
    
    # Track cleaning statistics
    cleaning_stats = {
        'MKS_NO_power_values': 0,
        'MKS_NO_nan_filled': 0,
        'MKS_CO_nan_filled': 0,
        'MKS_HC_nan_filled': 0
    }
    
    # ONLY check for power estimation data in MKS_NO (column 78)
    # if 'MKS_NO' in cleaned_data.columns:
    #     # Replace power estimation values with NaN only in MKS_NO column
    #     mask_power = cleaned_data['MKS_NO'] > POWER_ESTIMATION_THRESHOLD
    #     cleaning_stats['MKS_NO_power_values'] = mask_power.sum()
    #     cleaned_data.loc[mask_power, 'MKS_NO'] = np.nan
        #print(f"Replaced {mask_power.sum()} power estimation values in MKS_NO")
    
    # Handle NaN values and blanks in ALL MKS columns (connection drops)
    mks_columns = ['MKS_NO', 'MKS_CO', 'MKS_HC']
    
    for col in mks_columns:
        if col in cleaned_data.columns:
            # print(f"\n=== Processing {col} ===")
            # print(f"First 10 original values: {cleaned_data[col].head(10).tolist()}")
            
            # Convert any blank/empty strings to NaN
            cleaned_data[col] = cleaned_data[col].replace('', np.nan)
            cleaned_data[col] = cleaned_data[col].replace(' ', np.nan)
            cleaned_data[col] = cleaned_data[col].replace('nan', np.nan)
            
            # Handle string representations of blanks
            cleaned_data[col] = pandas.to_numeric(cleaned_data[col], errors='coerce')
            
            # ONLY convert zeros to NaN in MKS_NO column (connection drops)
            # Leave valid zeros in MKS_CO and MKS_HC as they may be legitimate readings
            if col == 'MKS_NO':
                cleaned_data.loc[cleaned_data[col] == 0, col] = np.nan
            
            # Count original NaN values (after converting blanks)
            original_nan_count = cleaned_data[col].isna().sum()
            #print(f"Total NaN/blank values found: {original_nan_count}")
            
            # Forward fill missing values (copy previous valid value)
            cleaned_data[col] = cleaned_data[col].fillna(method='ffill')
            
            # If there are still NaN values at the beginning, backward fill
            cleaned_data[col] = cleaned_data[col].fillna(method='bfill')
            
            # Check if any NaN values remain
            remaining_nan = cleaned_data[col].isna().sum()
            filled_count = original_nan_count - remaining_nan
            cleaning_stats[f'{col}_nan_filled'] = filled_count
            
            # print(f"First 10 cleaned values: {cleaned_data[col].head(10).tolist()}")
            #   print(f"{original_nan_count} NaN/blank values processed, {filled_count} filled, {remaining_nan} remaining")
            
    # Export cleaning report
    export_cleaning_report(data_frame, cleaned_data, cleaning_stats)
    
    return cleaned_data

def export_cleaning_report(original_data, cleaned_data, stats):
    """
    Export a detailed report of the data cleaning process
    
    Args:
        original_data: Original DataFrame before cleaning
        cleaned_data: Cleaned DataFrame after processing
        stats: Dictionary of cleaning statistics
    """
    # Create comparison DataFrame showing original vs cleaned data
    mks_columns = ['MKS_NO', 'MKS_CO', 'MKS_HC']
    
    # Get indices where data exists
    data_length = len(original_data)
    comparison_data = {
        'Index': range(data_length),
        'Date': original_data['Date'].tolist(),
        'Time': original_data['Time'].tolist()
    }
    
    # Add original and cleaned columns for comparison
    for col in mks_columns:
        if col in original_data.columns:
            comparison_data[f'{col}_Original'] = original_data[col].tolist()
            comparison_data[f'{col}_Cleaned'] = cleaned_data[col].tolist()
            
            # Flag rows where changes were made
            # Handle NaN comparisons properly using pandas methods
            original_values = original_data[col].fillna('NaN_ORIGINAL')
            cleaned_values = cleaned_data[col].fillna('NaN_CLEANED')
            changes = original_values != cleaned_values
            comparison_data[f'{col}_Changed'] = changes.tolist()
    
    # Create DataFrame and export
    comparison_df = pandas.DataFrame(comparison_data)
    comparison_df.to_csv("C:/PythonAnalyser/MKS_Data_Cleaning_Report.csv", index=False)
    
    # Create summary statistics
    summary_data = {
        'Statistic': ['Total Rows', 'MKS_NO Power Values Replaced', 'MKS_NO NaN/Blank Values Filled', 
                     'MKS_CO NaN/Blank Values Filled', 'MKS_HC NaN/Blank Values Filled'],
        'Count': [data_length, stats['MKS_NO_power_values'], stats['MKS_NO_nan_filled'],
                 stats['MKS_CO_nan_filled'], stats['MKS_HC_nan_filled']]
    }
    
    summary_df = pandas.DataFrame(summary_data)
    summary_df.to_csv("C:/PythonAnalyser/MKS_Cleaning_Summary.csv", index=False)
    
    print(f"Data cleaning report exported to: C:/PythonAnalyser/MKS_Data_Cleaning_Report.csv")
    print(f"Cleaning summary exported to: C:/PythonAnalyser/MKS_Cleaning_Summary.csv")

# Update the get_analyzer_data function to use the validation
def get_analyzer_data(selected_analyzer, data_frame):
    """
    Get gas concentration data based on selected analyzer number with validation
    
    Args:
        selected_analyzer (int): Analyzer number (1, 2, 3, or 4)
        data_frame: Pandas DataFrame containing the data
        
    Returns:
        tuple: (CO_data, HC_data, NO_data) as numpy arrays
    """

   
    print(f"Selected analyzer: {selected_analyzer}")
    if selected_analyzer == 1:
        return (
            np.array(data_frame.CO_1.tolist()),
            np.array(data_frame.HC_1.tolist()),
            np.array(data_frame.NO_1.tolist())
        )
    elif selected_analyzer == 2:
        return (
            np.array(data_frame.CO_2.tolist()),
            np.array(data_frame.HC_2.tolist()),
            np.array(data_frame.NO_2.tolist())
        )
    elif selected_analyzer == 3:
        return (
            np.array(data_frame.CO_3.tolist()),
            np.array(data_frame.HC_3.tolist()),
            np.array(data_frame.NO_3.tolist())
        )
    elif selected_analyzer == 4:  # MKS analyzer with validation
        # Clean MKS data before returning
        validated_data = validate_and_clean_mks_data(data_frame)
        print("Validating")
        return (
            np.array(validated_data.MKS_CO.tolist()),
            np.array(validated_data.MKS_HC.tolist()),
            np.array(validated_data.MKS_NO.tolist())
        )
    else:
        raise ValueError(f"Invalid analyzer selection: {selected_analyzer}. Must be 1, 2, 3, or 4")

###
def export_raw_mks_data(data_frame):
    """
    Export raw MKS analyzer data before any cleaning for manual inspection
    
    Args:
        data_frame: Pandas DataFrame containing the raw data
    """
    # Check if MKS columns exist
    mks_columns = ['MKS_NO', 'MKS_CO', 'MKS_HC']
    available_columns = [col for col in mks_columns if col in data_frame.columns]
    
    if not available_columns:
        print("WARNING: No MKS columns found in data")
        return
    
    # Create export data with timestamps for reference
    export_data = {
        'Index': range(len(data_frame)),
        'Date': data_frame['Date'].tolist(),
        'Time': data_frame['Time'].tolist()
    }
    
    # Add raw MKS columns
    for col in available_columns:
        export_data[f'{col}_Raw'] = data_frame[col].tolist()
        
        # Add some basic statistics about the raw data
        raw_values = data_frame[col]
        #print(f"\n=== Raw {col} Data Statistics ===")
        #print(f"Total values: {len(raw_values)}")
        #print(f"Non-null values: {raw_values.notna().sum()}")
        #print(f"Null/NaN values: {raw_values.isna().sum()}")
        #print(f"Zero values: {(raw_values == 0).sum()}")
        #print(f"Values > 10000: {(raw_values > 10000).sum()}")
        #print(f"Min value: {raw_values.min()}")
        #print(f"Max value: {raw_values.max()}")
        #print(f"First 10 values: {raw_values.head(10).tolist()}")
    
    # Create DataFrame and export
    raw_data_df = pandas.DataFrame(export_data)
    raw_data_df.to_csv("C:/PythonAnalyser/Raw_MKS_Data_Before_Cleaning.csv", index=False)
    
    print(f"\nRaw MKS data exported to: C:/PythonAnalyser/Raw_MKS_Data_Before_Cleaning.csv")
    #print(f"Columns exported: {available_columns}")

inputdata='C:/PythonAnalyser/LightoffTargetFiles.csv'
tcol=['Fnames']
targetfiles=pandas.read_csv(inputdata,names=tcol,encoding='cp1252')#chwck what encoding should be
tfnames=(targetfiles.Fnames.tolist())

#filename=tfnames[0]
##filename="VOL1001_VOL003_244Hours_SlowLo_211220_1 0"
#cols=['Date','Time','lambda_1','lambda_2','Inlet_temp','Bed_temp','Other_temp',"In_test",'CO_1','O2_1','HC_1','NO_1','CO_2','O2_2','HC_2','NO_2','CO_3','O2_3','HC_3','NO_3','MKS_NO','MKS_CO','MKS_HC']
#DLdata=pandas.read_csv(filename, names=cols, delim_whitespace=True, 
#                       usecols=[0,1,4,5,9,10,11,27,44,47,49,50,52,53,54,55,58,59,60,61,78,79,80],
#                       na_values=['', ' ', 'nan', 'NaN'], 
#                       skip_blank_lines=False,
#                       engine='python')  # Python engine is more flexible with inconsistent data
# Replace your DLdata loading with:
filename = tfnames[0]
DLdata = load_data_with_column_padding(filename, expected_columns=81)



# Export raw MKS data before any processing
print("Exporting raw MKS data before cleaning...")
export_raw_mks_data(DLdata)

DL_Date=np.array(DLdata.Date.tolist())
DL_Time=np.array(DLdata.Time.tolist())
Lambda_1=np.array(DLdata.lambda_1.tolist())
Lambda_2=np.array(DLdata.lambda_2.tolist())
Inlet_temp=np.array(DLdata.Inlet_temp.tolist())
Bed_temp=np.array(DLdata.Bed_temp.tolist())
Other_temp=np.array(DLdata.Other_temp.tolist())
In_test=np.array(DLdata.In_test.tolist())
DLtimesize=np.size(DL_Time)
DL_Timestamp_list = []
for x in range(0, DLtimesize):        
    currentdate = str(DL_Date[x])
    
    # Check if date already has slashes (format: DD/MM/YY)
    if "/" in currentdate:
        # Split by slash and parse
        date_parts = currentdate.split("/")
        if len(date_parts) == 3:
            day = date_parts[0]
            month = date_parts[1] 
            year = date_parts[2]
            
            # Convert 2-digit year to 4-digit year (assuming 20xx)
            if len(year) == 2:
                full_year = "20" + year
            else:
                full_year = year
                
            # Format as DD/MM/YYYY HH:MM:SS
            timestamp_str = f"{day}/{month}/{full_year} {DL_Time[x]}"
        else:
            timestamp_str = f"INVALID_DATE_FORMAT {DL_Time[x]}"
    else:
        # Original logic for format like "170625" (6 digits without slashes)
        if len(currentdate) >= 6:
            day = currentdate[0:2]
            month = currentdate[2:4]
            year = currentdate[4:6]
            full_year = "20" + year
            timestamp_str = f"{day}/{month}/{full_year} {DL_Time[x]}"
        else:
            timestamp_str = f"INVALID_DATE {DL_Time[x]}"
    
   
    DL_Timestamp_list.append(timestamp_str)

DL_Timestamp = DL_Timestamp_list
analyser_selection="C:/PythonAnalyser/analyser_select.csv"
analyser_cols=['Selection']
selection_data=pandas.read_csv(analyser_selection,names=analyser_cols)
selected_analysers=np.array(selection_data.Selection.tolist())
temp_variable=np.empty(DLtimesize)
temp_startpoint=0
test_endpoint=DLtimesize
test_startpoint=0
turningpoint_variable=np.empty(DLtimesize)
for x in range(0,DLtimesize):
    if In_test[x]>146:
        test_startpoint=x
        break

                
test_startpoint=test_startpoint-2

# Replace your existing analyzer data retrieval (around line 106)
print("Getting analyser data")
Co_in, Hc_in, No_in = get_analyzer_data(4, DLdata)
Co_out, Hc_out, No_out = get_analyzer_data(4, DLdata)


mfcfilename=tfnames[1]
mfccols=['Date','Time','CO','HC','Air','NO','N2_1','N2_2','Set_CO','Set_HC','Set_Air','Set_NO','Set1_N2','Set2_N2']
MFCdata=pandas.read_csv(mfcfilename,names=mfccols,delim_whitespace=(True),usecols=[0,1,2,3,4,5,6,7,13,14,15,16,17,18])
MFC_Date=np.array(MFCdata.Date.tolist())
MFC_Time=np.array(MFCdata.Time.tolist())
MFCtimesize=np.size(MFC_Time)
MFC_Timestamp_list = []
for x in range(0, MFCtimesize):        
    currentdate = str(MFC_Date[x])
    
    # Check if date already has slashes (format: DD/MM/YY)
    if "/" in currentdate:
        # Split by slash and parse
        date_parts = currentdate.split("/")
        if len(date_parts) == 3:
            day = date_parts[0]
            month = date_parts[1] 
            year = date_parts[2]
            
            # Convert 2-digit year to 4-digit year (assuming 20xx)
            if len(year) == 2:
                full_year = "20" + year
            else:
                full_year = year
                
            # Format as DD/MM/YYYY HH:MM:SS
            timestamp_str = f"{day}/{month}/{full_year} {MFC_Time[x]}"
        else:
            timestamp_str = f"INVALID_DATE_FORMAT {MFC_Time[x]}"
    else:
        # Original logic for format like "170625" (6 digits without slashes)
        if len(currentdate) >= 6:
            day = currentdate[0:2]
            month = currentdate[2:4]
            year = currentdate[4:6]
            full_year = "20" + year
            timestamp_str = f"{day}/{month}/{full_year} {MFC_Time[x]}"
        else:
            timestamp_str = f"INVALID_DATE {MFC_Time[x]}"
    
    MFC_Timestamp_list.append(timestamp_str)

MFC_Timestamp = MFC_Timestamp_list
MFC_air=np.array(MFCdata.Air.tolist())
MFC_co=np.array(MFCdata.CO.tolist())
MFC_hc=np.array(MFCdata.HC.tolist())
MFC_no=np.array(MFCdata.NO.tolist())
MFC_n2_1=np.array(MFCdata.N2_1.tolist())
MFC_n2_2=np.array(MFCdata.N2_2.tolist())
MFC_n2=MFC_n2_1+MFC_n2_2 
Set_co=np.array(MFCdata.Set_CO.tolist())
Set_hc=np.array(MFCdata.Set_HC.tolist())
Set_air=np.array(MFCdata.Set_Air.tolist())
Set_no=np.array(MFCdata.Set_NO.tolist())
Set1_n2=np.array(MFCdata.Set1_N2.tolist())
Set2_n2=np.array(MFCdata.Set2_N2.tolist())
Set_n2=Set1_n2+Set2_n2 



Air_output=np.empty(DLtimesize)
CO_output=np.empty(DLtimesize)
HC_output=np.empty(DLtimesize)
NO_output=np.empty(DLtimesize)
N2_output=np.empty(DLtimesize)
offset=0

# Add debugging to understand what's happening
for x in range(0, DLtimesize):    
    if (DL_Timestamp[x] in MFC_Timestamp):
        MFCindex = MFC_Timestamp.index(DL_Timestamp[x]) + offset
        if MFCindex <= np.size(MFC_Timestamp) - 1 and MFCindex >= 0:  # Fixed bounds check
            #print(f"Match found at index {x}, MFC index {MFCindex}")
            
            # Your existing logic here...
            if MFC_air[MFCindex] == 0 and Set_air[MFCindex] != 0:
                Air_output[x] = Set_air[MFCindex]
            else:
                Air_output[x] = MFC_air[MFCindex]
                
            if MFC_no[MFCindex] == 0 and Set_no[MFCindex] != 0:
                NO_output[x] = Set_no[MFCindex]
            else:
                NO_output[x] = MFC_no[MFCindex]
                
            if MFC_hc[MFCindex] == 0 and Set_hc[MFCindex] != 0:
                HC_output[x] = Set_hc[MFCindex]
            else:
                HC_output[x] = MFC_hc[MFCindex]
                
            if MFC_co[MFCindex] == 0 and Set_co[MFCindex] != 0:
                CO_output[x] = Set_co[MFCindex]
            else:
                CO_output[x] = MFC_co[MFCindex]
                
            if MFC_n2[MFCindex] == 0 and Set_n2[MFCindex] != 0:
                N2_output[x] = Set_n2[MFCindex]
            else:
                N2_output[x] = MFC_n2[MFCindex]
    else:
        # Count mismatches instead of printing each one
        if x == 0:  # Initialize counter on first iteration
            mismatch_count = 0
        mismatch_count += 1
        if x < 5:  # Show format differences for first few only
            print(f"No match found for DL_Timestamp[{x}]: '{DL_Timestamp[x]}'")
            print(f"Example MFC format: '{MFC_Timestamp[0] if MFC_Timestamp else 'None'}'")
        elif x == DLtimesize - 1:  # Print summary at the end
            print(f"Total timestamp mismatches: {mismatch_count}")

timesize=np.size(DL_Time)
Oxidisation_variable=np.zeros(timesize)
Oxidisation_variable2=np.zeros(timesize)
Reduction_variable=np.zeros(timesize)
Reduction_variable2=np.zeros(timesize)
L1_oxidisationpoint=0
L2_oxidisationpoint=0
L1_reductionpoint=0
L2_reductionpoint=0
PostOxidisation_array=np.empty(10)
PostOxidisation_array2=np.empty(10)
PostReduction_array=np.empty(10)
PostReduction_array2=np.empty(10)
Oxidisation_endpoint=np.array([])
Oxidisation_endpoint2=np.array([])
Reduction_endpoint=np.array([])
Reduction_endpoint2=np.array([])
Oxidisation2_startpoint=0
Reduction2_startpoint=0

for x in range(0,timesize):
    if Lambda_1[x]<0.995:
        Reduction_variable[x]=1
    if Lambda_1[x]>=0.995:
        Reduction_variable[x]=0
    if Reduction_variable[x]==1 and Reduction_variable[x-1]==1 and Reduction_variable[x-2]==0:
        L1_reductionpoint=x-1
        break
for y in range (L1_reductionpoint,timesize): 
    if Lambda_1[y]<0.995:
        Reduction_variable[y]=1
    if Lambda_1[y]>=0.995:
        Reduction_variable[y]=0
    if Reduction_variable[y-1]==1 and Reduction_variable[y]==0:
        Reduction_endpoint=np.append(Reduction_endpoint,(y-1))
        Reduction2_startpoint=y-1
        break 
            
for x in range (0, timesize):
    if Lambda_1[x]>=1.005:
        Oxidisation_variable[x]=1
    if Lambda_1[x]<1.005:
        Oxidisation_variable[x]=0
    if Oxidisation_variable[x]==1 and Oxidisation_variable[x-1]==1 and Oxidisation_variable[x-2]==0:
        L1_oxidisationpoint=x-1
        break        
for y in range (L1_oxidisationpoint,timesize):           
    if Lambda_1[y]>=1.005:
        Oxidisation_variable[y]=1
    if Lambda_1[y]<1.005:
        Oxidisation_variable[y]=0
    if Oxidisation_variable[y-1]==1 and Oxidisation_variable[y]==0 :
        Oxidisation_endpoint=np.append(Oxidisation_endpoint,(y-1))
        Oxidisation2_startpoint=y-1
        break

for x in range(Reduction2_startpoint,timesize):
    if Lambda_1[x]<0.995:
        Reduction_variable[x]=1
    if Lambda_1[x]>=0.995:
        Reduction_variable[x]=0
    if Reduction_variable[x]==1 and Reduction_variable[x-1]==1 and Reduction_variable[x-2]==0:
        L2_reductionpoint=x-1
        break

            
for x in range (Oxidisation2_startpoint, timesize):
    if Lambda_1[x]>=1.005:
        Oxidisation_variable[x]=1
    if Lambda_1[x]<1.005:
        Oxidisation_variable[x]=0
    if Oxidisation_variable[x]==1 and Oxidisation_variable[x-1]==1 and Oxidisation_variable[x-2]==0:
        L2_oxidisationpoint=x-1
        break

            
CO_reductionvariable=np.zeros(timesize)  
COout_reductionvariable=np.zeros(timesize)
   
HC_reductionvariable=np.zeros(timesize)
HCout_reductionvariable=np.zeros(timesize)  
NO_oxidisationvariable=np.zeros(timesize)
NOout_oxidisationvariable=np.zeros(timesize)

HCin_startpoint=0
HCout_startpoint=0
NOin_startpoint=0
NOout_startpoint=0          
COin_endpoint=0

HCin_endpoint=0
NOin_endpoint=0
COout_endpoint=0
O2out_endpoint=0
HCout_endpoint=0
NOout_endpoint=0

CO_reductionpoint=0

HC_reductionpoint=0
NO_oxidisationpoint=0

for x in range (0,timesize):
    if Co_in[x]>=0.3:
        CO_reductionvariable[x]=1
    if Co_in[x]<0.3:
        CO_reductionvariable[x]=0
    if CO_reductionvariable[x]==1 and CO_reductionvariable[x-1]==0:
        CO_reductionpoint=x
        break
for y in range (CO_reductionpoint,timesize):
    if Co_in[y]<0.3:
        COin_endpoint=y
        HCin_startpoint=y
        break



for x in range (0,timesize):
    if Co_out[x]>=0.3:
        COout_reductionvariable[x]=1
    if Co_out[x]<0.3:
        COout_reductionvariable[x]=0
    if COout_reductionvariable[x]==1 and COout_reductionvariable[x-1]==0:
        CO_reductionpoint=x
        break
for y in range (CO_reductionpoint,timesize):
    if Co_out[y]<0.3:
        COout_endpoint=y
        HCout_startpoint=y
        break



CO_in=Co_in
CO_out=Co_out
HC_in=Hc_in
HC_out=Hc_out
NO_in=No_in
NO_out=No_out
if L1_reductionpoint==0:
    COin_latency=0
    COout_latency=0
else:
    COin_latency=COin_endpoint-Reduction_endpoint
    COout_latency=COout_endpoint-Reduction_endpoint
    
COin_aligned=np.empty(DLtimesize)
HCin_aligned=np.empty(DLtimesize)
NOin_aligned=np.empty(DLtimesize)
COout_aligned=np.empty(DLtimesize)
HCout_aligned=np.empty(DLtimesize)
NOout_aligned=np.empty(DLtimesize)

for x in range (0,int(COin_latency)):
    Co_in=np.append(Co_in,Co_in[-1])

for x in range(0,int(COout_latency)):
    Co_out=np.append(Co_out,Co_out[-1])

    
    
for x in range (0,DLtimesize):
    COin_index=DL_Timestamp.index(DL_Timestamp[x])+COin_latency
    
    COout_index=DL_Timestamp.index(DL_Timestamp[x])+COout_latency
    if COin_index>=COin_latency:
        COin_aligned[x]=Co_in[int(COin_index)]

        
    if COout_index>=COout_latency:
        COout_aligned[x]=Co_out[int(COout_index)]
#    if O2out_index>=O2out_latency:
#        O2out_aligned[x]=o2_out[int(O2out_index)]
   
   
if (L1_reductionpoint +100)>=L2_reductionpoint>0:
    for y in range (L2_reductionpoint,timesize):
        if Lambda_1[y]<0.995:
            Reduction_variable[y]=1
        if Lambda_1[y]>=0.995:
            Reduction_variable[y]=0
        if Reduction_variable[y-2]==1 and Reduction_variable[y-1]==0 and Reduction_variable[y]==0:
            Reduction_endpoint2=np.append(Reduction_endpoint2,(y-2))                
            break
    for x in range (HCin_startpoint,timesize):
        if Hc_in[x]>=133:
            HC_reductionvariable[x]=1
        if Hc_in[x]<133:
            HC_reductionvariable[x]=0
        if HC_reductionvariable[x]==1 and HC_reductionvariable[x-1]==0:
            HC_reductionpoint=x
            # print(HC_reductionpoint)
            # print("a")
            break
    for y in range(HC_reductionpoint,timesize):
        if Hc_in[y]<133:
            HCin_endpoint=y
            # print(HCin_endpoint)
            # print("b")
            NO_startpoint=y
            break
    for x in range (HCout_startpoint,timesize):
        if Hc_out[x]>=133:
            HCout_reductionvariable[x]=1
        if Hc_out[x]<133:
            HCout_reductionvariable[x]=0
        if HCout_reductionvariable[x]==1 and HCout_reductionvariable[x-1]==0:
            HC_reductionpoint=x
            # print(HC_reductionpoint)
            # print("c")
            break
    for y in range(HC_reductionpoint,timesize):
        if Hc_out[y]<133:
            HCout_endpoint=y
            # print(HCout_endpoint)
            # print("d")
            NOout_startpoint=y
            break
    if HCin_endpoint>0:
        HCin_latency=HCin_endpoint-Reduction_endpoint2
    else:
        HCin_latency=COin_latency
    if HCout_endpoint>0:
        # print("hc latency")
        # print(HCout_endpoint)
        # print("reduction endpoint")
        # print(Reduction_endpoint2)
        HCout_latency=HCout_endpoint-Reduction_endpoint2
    else:
        HCout_latency=COout_latency

    for x in range(0,int(HCin_latency)):
        Hc_in=np.append(Hc_in,Hc_in[-1])
        
    for x in range(0,int(HCout_latency)):
        Hc_out=np.append(Hc_out,Hc_out[-1])
     
    for x in range (0,DLtimesize):
        HCin_index=DL_Timestamp.index(DL_Timestamp[x])+HCin_latency
        HCout_index=DL_Timestamp.index(DL_Timestamp[x])+HCout_latency
        if HCin_index>=HCin_latency:           
            HCin_aligned[x]=Hc_in[int(HCin_index)]
        if HCout_index>=HCout_latency:
            HCout_aligned[x]=Hc_out[int(HCout_index)]
else:
    HCin_latency=COin_latency
    HCout_latency=COout_latency
    for x in range(0,int(COin_latency)):
        Hc_in=np.append(Hc_in,Hc_in[-1])
        
    for x in range(0,int(COout_latency)):
        Hc_out=np.append(Hc_out,Hc_out[-1])
    for x in range (0,DLtimesize):
        HCin_index=DL_Timestamp.index(DL_Timestamp[x])+COin_latency
        HCout_index=DL_Timestamp.index(DL_Timestamp[x])+COout_latency
        if HCin_index>=COin_latency:           
            HCin_aligned[x]=Hc_in[int(HCin_index)]
        if HCout_index>=COout_latency:
            HCout_aligned[x]=Hc_out[int(HCout_index)]          
            
     
if (L1_oxidisationpoint+100)>=L2_oxidisationpoint>0:
    for y in range (L2_oxidisationpoint,timesize):
        if Lambda_1[y]>=1.005:
            Oxidisation_variable[y]=1
        if Lambda_1[y]<1.005:
            Oxidisation_variable[y]=0
        if Oxidisation_variable[y-2]==1 and Oxidisation_variable[y-1]==0 and Oxidisation_variable[y]==0:
            Oxidisation_endpoint2=np.append(Oxidisation_endpoint2,(y-2))
            break
    for x in range(NOin_startpoint,timesize):
        if No_in[x]>=400:
            NO_oxidisationvariable[x]=1
        if No_in[x]<400:
            NO_oxidisationvariable[x]=0
        if NO_oxidisationvariable[x]==1 and NO_oxidisationvariable[x-1]==0:
            NO_oxidisationpoint=x
            break
    for y in range (NO_oxidisationpoint,timesize):
        if No_in[y]<400:
            NOin_endpoint=y
            break        
    for x in range(NOout_startpoint,timesize):
        if No_out[x]>=400:
            NOout_oxidisationvariable[x]=1
        if No_out[x]<400:
            NOout_oxidisationvariable[x]=0
        if NOout_oxidisationvariable[x]==1 and NOout_oxidisationvariable[x-1]==0:
            NO_oxidisationpoint=x
            break
    for y in range (NO_oxidisationpoint,timesize):
        if No_out[y]<400:
            NOout_endpoint=y
            break 
    if NOin_endpoint>0:
        # print("firrst latency")
        NOin_latency=NOin_endpoint-Oxidisation_endpoint2           
    else:
        #NOin_latency=O2in_latency
        NOin_latency=COin_latency
    if NOout_endpoint>0:
        NOout_latency=NOout_endpoint-Oxidisation_endpoint2
    else:
        #NOout_latency=O2out_latency
        NOout_latency=COout_latency
    for x in range (0,int(NOin_latency)):
        No_in=np.append(No_in,No_in[-1])
        
    for x in range(0,int(NOout_latency)):
        No_out=np.append(No_out,No_out[-1]) 
        
    for x in range (0,DLtimesize): 
        NOin_index=DL_Timestamp.index(DL_Timestamp[x])+NOin_latency
        NOout_index=DL_Timestamp.index(DL_Timestamp[x])+NOout_latency
        if NOin_index>=NOin_latency:            
            NOin_aligned[x]=No_in[int(NOin_index)]
        if NOout_index>=NOout_latency:
            NOout_aligned[x]=No_out[int(NOout_index)]
else:
    #NOin_latency=O2in_latency
    NOin_latency=COin_latency
    #NOout_latency=O2out_latency
    NOout_latency=COout_latency
    #for x in range (0,int(O2in_latency)):
    #    No_in=np.append(No_in,No_in[-1])
        
    #for x in range(0,int(O2out_latency)):
    #    No_out=np.append(No_out,No_out[-1])
    for x in range (0,DLtimesize): 
        NOin_index=DL_Timestamp.index(DL_Timestamp[x])+HCin_latency
        NOout_index=DL_Timestamp.index(DL_Timestamp[x])+HCout_latency
    #    if NOin_index>=O2in_latency:            
    #        NOin_aligned[x]=No_in[int(NOin_index)]
    #    if NOout_index>=O2out_latency:
    #        NOout_aligned[x]=No_out[int(NOout_index)]
                
            

#outputsize=np.size(HC_output)
#CO_Conc=np.empty(outputsize)
#HC_Conc=np.empty(outputsize)
#NO_Conc=np.empty(outputsize)
#MassFlow=np.empty(outputsize)
#for x in range(0,outputsize):
#    if CO_output[x]==0:
#        CO_Conc[x]=0
#    else:
#        CO_Conc[x]=CO_output[x]/(CO_output[x]+HC_output[x]+Air_output[x]+NO_output[x]+N2_output[x])*100
#    if HC_output[x]==0:
#        HC_Conc[x]=0
#    else:
#        HC_Conc[x]=HC_output[x]/(CO_output[x]+HC_output[x]+Air_output[x]+NO_output[x]+N2_output[x])*3000000
#    if NO_output[x]==0:
#        NO_Conc[x]=0
#    else:
#       NO_Conc[x]=NO_output[x]/(CO_output[x]+HC_output[x]+Air_output[x]+NO_output[x]+N2_output[x])*1000000
#    
#    MassFlow[x]=(CO_output[x]+HC_output[x]+Air_output[x]+NO_output[x]+N2_output[x])/60*(101325/288/296)

# Add this right after the concentration calculations (after line ~800)
outputsize=np.size(HC_output)
CO_Conc=np.empty(outputsize)
HC_Conc=np.empty(outputsize)
NO_Conc=np.empty(outputsize)
MassFlow=np.empty(outputsize)

# Track calculation components for debugging
Total_Flow=np.empty(outputsize)
CO_Fraction=np.empty(outputsize)
HC_Fraction=np.empty(outputsize)
NO_Fraction=np.empty(outputsize)

for x in range(0,outputsize):
    # Calculate total flow first
    Total_Flow[x] = CO_output[x] + HC_output[x] + Air_output[x] + NO_output[x] + N2_output[x]
    
    if CO_output[x]==0:
        CO_Conc[x]=0
        CO_Fraction[x]=0
    else:
        CO_Fraction[x] = CO_output[x] / Total_Flow[x]
        CO_Conc[x] = CO_Fraction[x] * 100  # Convert to percentage
        
    if HC_output[x]==0:
        HC_Conc[x]=0
        HC_Fraction[x]=0
    else:
        HC_Fraction[x] = HC_output[x] / Total_Flow[x]
        HC_Conc[x] = HC_Fraction[x] * 3000000  # Convert to ppm
        
    if NO_output[x]==0:
        NO_Conc[x]=0
        NO_Fraction[x]=0
    else:
        NO_Fraction[x] = NO_output[x] / Total_Flow[x]
        NO_Conc[x] = NO_Fraction[x] * 1000000  # Convert to ppm
    
    MassFlow[x] = Total_Flow[x] / 60 * (101325/288/296)
   
ScaleFactor_COin=CO_Conc[test_startpoint]/COin_aligned[test_startpoint]
if ScaleFactor_COin>2:
    ScaleFactor_COin=2
ScaleFactor_COout=CO_Conc[test_startpoint]/COout_aligned[test_startpoint]
if ScaleFactor_COout>2:
    ScaleFactor_COout=2
ScaleFactor_HCin=HC_Conc[test_startpoint]/HCin_aligned[test_startpoint]
if ScaleFactor_HCin>2:
    ScaleFactor_HCin=2
ScaleFactor_HCout=HC_Conc[test_startpoint]/HCout_aligned[test_startpoint]
if ScaleFactor_HCout>2:
    ScaleFactor_HCout=2
ScaleFactor_NOin=NO_Conc[test_startpoint]/NOin_aligned[test_startpoint]
if ScaleFactor_NOin>2:
    ScaleFactor_NOin=2
ScaleFactor_NOout=NO_Conc[test_startpoint]/NOout_aligned[test_startpoint]
if ScaleFactor_NOout>2:
    ScaleFactor_NOout=2
Scaled_COin=np.empty(outputsize)
Scaled_COout=np.empty(outputsize)
Scaled_HCin=np.empty(outputsize)
Scaled_HCout=np.empty(outputsize)
Scaled_NOin=np.empty(outputsize)
Scaled_NOout=np.empty(outputsize)    
for x in range(outputsize):
    Scaled_COin[x]=COin_aligned[x]*ScaleFactor_COin
    Scaled_COout[x]=COout_aligned[x]*ScaleFactor_COout
    Scaled_HCin[x]=HCin_aligned[x]*ScaleFactor_HCin
    Scaled_HCout[x]=HCout_aligned[x]*ScaleFactor_HCout
    Scaled_NOin[x]=NOin_aligned[x]*ScaleFactor_NOin
    Scaled_NOout[x]=NOout_aligned[x]*ScaleFactor_NOout
    
    

# Export concentration calculation breakdown
Concentration_Calc_df = pandas.DataFrame({
    "Index": range(outputsize),
    "Timestamp": DL_Timestamp[:outputsize],
    "CO_output": CO_output,
    "HC_output": HC_output,
    "Air_output": Air_output,
    "NO_output": NO_output,
    "N2_output": N2_output,
    "Total_Flow": Total_Flow,
    "CO_Fraction": CO_Fraction,
    "HC_Fraction": HC_Fraction,
    "NO_Fraction": NO_Fraction,
    "CO_Concentration_Percent": CO_Conc,
    "HC_Concentration_ppm": HC_Conc,
    "NO_Concentration_ppm": NO_Conc,
    "Mass_Flow": MassFlow,
    "CO_Zero_Flag": (CO_output == 0).astype(int),
    "HC_Zero_Flag": (HC_output == 0).astype(int),
    "NO_Zero_Flag": (NO_output == 0).astype(int),
    # Add inlet and outlet analyzer readings
    "CO_Inlet_Raw": CO_in[:outputsize],
    "HC_Inlet_Raw": HC_in[:outputsize], 
    "NO_Inlet_Raw": NO_in[:outputsize],
    "CO_Outlet_Raw": CO_out[:outputsize],
    "HC_Outlet_Raw": HC_out[:outputsize],
    "NO_Outlet_Raw": NO_out[:outputsize],
    # Add aligned (latency corrected) values
    "CO_Inlet_Aligned": COin_aligned[:outputsize],
    "HC_Inlet_Aligned": HCin_aligned[:outputsize],
    "NO_Inlet_Aligned": NOin_aligned[:outputsize],
    "CO_Outlet_Aligned": COout_aligned[:outputsize],
    "HC_Outlet_Aligned": HCout_aligned[:outputsize],
    "NO_Outlet_Aligned": NOout_aligned[:outputsize],
    # Add scaled values
    "CO_Inlet_Scaled": Scaled_COin,
    "HC_Inlet_Scaled": Scaled_HCin,
    "NO_Inlet_Scaled": Scaled_NOin,
    "CO_Outlet_Scaled": Scaled_COout,
    "HC_Outlet_Scaled": Scaled_HCout,
    "NO_Outlet_Scaled": Scaled_NOout,
    # Add scale factors for reference
    "Scale_Factor_CO_In": [ScaleFactor_COin] * outputsize,
    "Scale_Factor_HC_In": [ScaleFactor_HCin] * outputsize,
    "Scale_Factor_NO_In": [ScaleFactor_NOin] * outputsize,
    "Scale_Factor_CO_Out": [ScaleFactor_COout] * outputsize,
    "Scale_Factor_HC_Out": [ScaleFactor_HCout] * outputsize,
    "Scale_Factor_NO_Out": [ScaleFactor_NOout] * outputsize,
    # Add latency values for reference
    "CO_Inlet_Latency": [COin_latency] * outputsize,
    "HC_Inlet_Latency": [HCin_latency] * outputsize,
    "NO_Inlet_Latency": [NOin_latency] * outputsize,
    "CO_Outlet_Latency": [COout_latency] * outputsize,
    "HC_Outlet_Latency": [HCout_latency] * outputsize,
    "NO_Outlet_Latency": [NOout_latency] * outputsize
})

Concentration_Calc_df.to_csv("C:/PythonAnalyser/Concentration_Calculations.csv", index=False)
#print("Concentration calculation breakdown exported to: C:/PythonAnalyser/Concentration_Calculations.csv")

# Simple MFC vs Set point tracking
# Check array lengths before creating DataFrame
# print("=== Array Length Check ===")
# print(f"MFC_Timestamp length: {len(MFC_Timestamp)}")
# print(f"MFC_air length: {len(MFC_air)}")
# print(f"Set_air length: {len(Set_air)}")
# print(f"MFC_co length: {len(MFC_co)}")
# print(f"Set_co length: {len(Set_co)}")
# print(f"MFC_hc length: {len(MFC_hc)}")
# print(f"Set_hc length: {len(Set_hc)}")
# print(f"MFC_no length: {len(MFC_no)}")
# print(f"Set_no length: {len(Set_no)}")
# print(f"MFC_n2 length: {len(MFC_n2)}")
# print(f"Set_n2 length: {len(Set_n2)}")

# Find the minimum length to use for all arrays
all_arrays = [MFC_Timestamp, MFC_air, Set_air, MFC_co, Set_co, 
              MFC_hc, Set_hc, MFC_no, Set_no, MFC_n2, Set_n2]
min_length = min(len(arr) for arr in all_arrays)
print(f"Minimum length: {min_length}")

# Simple MFC vs Set point tracking - using minimum length
MFC_Set_df = pandas.DataFrame({
    "Index": range(min_length),
    "MFC_Timestamp": MFC_Timestamp[:min_length],
    
    # Air values
    "MFC_Air": MFC_air[:min_length],
    "Set_Air": Set_air[:min_length],
    
    # CO values
    "MFC_CO": MFC_co[:min_length],
    "Set_CO": Set_co[:min_length],
    
    # HC values
    "MFC_HC": MFC_hc[:min_length],
    "Set_HC": Set_hc[:min_length],
    
    # NO values
    "MFC_NO": MFC_no[:min_length],
    "Set_NO": Set_no[:min_length],
    
    # N2 values
    "MFC_N2": MFC_n2[:min_length],
    "Set_N2": Set_n2[:min_length]
})

MFC_Set_df.to_csv("C:/PythonAnalyser/MFC_Set_Tracking.csv", index=False)
print("MFC vs Set tracking exported to: C:/PythonAnalyser/MFC_Set_Tracking.csv")


Raw_Data_df=pandas.DataFrame({"Timestamp":DL_Timestamp,"Inlet Lambda":Lambda_1,"Outlet Lambda":Lambda_2,"Inlet temp":Inlet_temp,"Bed Temp":Bed_temp,"Other temp":Other_temp,"Calculated CO IN":CO_Conc,"Calculated HC IN":HC_Conc,"Calculated NO IN":NO_Conc,"Inlet CO":CO_in,"Inlet O2":NO_in,"Inlet HC":HC_in,"Inlet NO":NO_in,"Outlet CO":CO_out,"Outlet O2":NO_out,"Outlet HC":HC_out,"Outlet NO":NO_out,"CO injection":CO_output,"O2 injection":Air_output ,"HC injection":HC_output,"NO injection":NO_output,"Mass Flow":MassFlow})
Raw_Data_df.to_csv("C:/PythonAnalyser/Raw Data.csv",index=False, header=False)

Lightoff_latency_df=pandas.DataFrame({"Test Startpoint":test_startpoint,"Test Endpoint":test_endpoint,"Selected analyser 1":selected_analysers[0],"CO in latency":COin_latency,"O2 in latency":NOin_latency,"HC in latency":HCin_latency,"NO in latency":NOin_latency,"Scale Factor CO in":ScaleFactor_COin,"Scale Factor HC in":ScaleFactor_HCin,"Scale Factor NO in":ScaleFactor_NOin,"Selected Analyser 2":selected_analysers[1],"CO out latency":COout_latency,"O2 out latency":NOout_latency,"HC out latency":HCout_latency,"NO out latency":NOout_latency,"Scale Factor CO out":ScaleFactor_COout,"Scale Factor HC out":ScaleFactor_HCout,"Scale Factor NO out":ScaleFactor_NOout})
Lightoff_latency_df.to_csv("C:/PythonAnalyser/Lightoff Latency.csv",index=False, header=False)

# Test the cleaning function
#cleaned_test = validate_and_clean_mks_data(DLdata)
#print("=== After Cleaning ===") 
#print("MKS_CO first 10:", cleaned_test['MKS_CO'].head(10).tolist())



