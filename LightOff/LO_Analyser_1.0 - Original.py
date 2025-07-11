# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 11:07:08 2021

@author: Home
"""

import pandas 
import numpy as np
#import matplotlib.pyplot as plt

inputdata='C:/PythonAnalyser/LightoffTargetFiles.csv'
tcol=['Fnames']
targetfiles=pandas.read_csv(inputdata,names=tcol,encoding='cp1252')#chwck what encoding should be
tfnames=(targetfiles.Fnames.tolist())

filename=tfnames[0]
cols=['Date','Time','lambda_1','lambda_2','Inlet_temp','Bed_temp','Other_temp',"In_test",'CO_1','O2_1','HC_1','NO_1','CO_2','O2_2','HC_2','NO_2','CO_3','O2_3','HC_3','NO_3']
DLdata=pandas.read_csv(filename,names=cols,delim_whitespace=(True),usecols=[0,1,4,5,9,10,11,27,44,47,49,50,52,53,54,55,58,59,60,61])
DL_Date=np.array(DLdata.Date.tolist())
DL_Time=np.array(DLdata.Time.tolist())
Lambda_1=np.array(DLdata.lambda_1.tolist())
Lambda_2=np.array(DLdata.lambda_2.tolist())
Inlet_temp=np.array(DLdata.Inlet_temp.tolist())
Bed_temp=np.array(DLdata.Bed_temp.tolist())
Other_temp=np.array(DLdata.Other_temp.tolist())
In_test=np.array(DLdata.In_test.tolist())
DLtimesize=np.size(DL_Time)
DL_Timestamp=np.empty(DLtimesize,dtype=object)
for x in range (0,DLtimesize):        
    currentdate=DL_Date[x]
    CurrentdateA=currentdate[0:6]
    CurrentdateB=currentdate[6:8]
    DL_Timestamp[x]= CurrentdateA + "20" + CurrentdateB + " " + DL_Time[x]

DL_Timestamp=DL_Timestamp.tolist()
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
if selected_analysers[0]==1:
    Co_in=np.array(DLdata.CO_1.tolist())
    o2_in=np.array(DLdata.O2_1.tolist())
    Hc_in=np.array(DLdata.HC_1.tolist())
    No_in=np.array(DLdata.NO_1.tolist())
if selected_analysers[0]==2:
    Co_in=np.array(DLdata.CO_2.tolist())
    o2_in=np.array(DLdata.O2_2.tolist())
    Hc_in=np.array(DLdata.HC_2.tolist())
    No_in=np.array(DLdata.NO_2.tolist())
if selected_analysers[0]==3:
    Co_in=np.array(DLdata.CO_3.tolist())
    o2_in=np.array(DLdata.O2_3.tolist())
    Hc_in=np.array(DLdata.HC_3.tolist())
    No_in=np.array(DLdata.NO_3.tolist())
    
if selected_analysers[1]==1:
    Co_out=np.array(DLdata.CO_1.tolist())
    o2_out=np.array(DLdata.O2_1.tolist())
    Hc_out=np.array(DLdata.HC_1.tolist())
    No_out=np.array(DLdata.NO_1.tolist())
if selected_analysers[1]==2:
    Co_out=np.array(DLdata.CO_2.tolist())
    o2_out=np.array(DLdata.O2_2.tolist())
    Hc_out=np.array(DLdata.HC_2.tolist())
    No_out=np.array(DLdata.NO_2.tolist())
if selected_analysers[1]==3:
    Co_out=np.array(DLdata.CO_3.tolist())
    o2_out=np.array(DLdata.O2_3.tolist())
    Hc_out=np.array(DLdata.HC_3.tolist())
    No_out=np.array(DLdata.NO_3.tolist())
    

    
mfcfilename=tfnames[1]
mfccols=['Date','Time','CO','HC','Air','NO','N2_1','N2_2','Set_CO','Set_HC','Set_Air','Set_NO','Set1_N2','Set2_N2']
MFCdata=pandas.read_csv(mfcfilename,names=mfccols,delim_whitespace=(True),usecols=[0,1,2,3,4,5,6,7,13,14,15,16,17,18])
MFC_Date=np.array(MFCdata.Date.tolist())
MFC_Time=np.array(MFCdata.Time.tolist())
MFC_Timestamp=np.empty(np.size(MFC_Time),dtype=object)
for x in range (0,np.size(MFC_Time)-1):
    currentdate=MFC_Date[x]
    CurrentdateA=currentdate[0:6]
    CurrentdateB=currentdate[6:8]
    MFC_Timestamp[x]= CurrentdateA + "20" + CurrentdateB + " " + MFC_Time[x]
MFC_Timestamp=MFC_Timestamp.tolist()
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

for x in range (0,DLtimesize):
    if (DL_Timestamp[x] in MFC_Timestamp):
        MFCindex=MFC_Timestamp.index(DL_Timestamp[x])+offset
        if MFCindex<=np.size(MFC_Timestamp) and MFCindex>=0:
            if MFC_air[MFCindex]==0 and Set_air[MFCindex]!=0:
                Air_output[x]=Set_air[MFCindex]
            else:
                Air_output[x]=MFC_air[MFCindex]
                
            if MFC_no[MFCindex]==0 and Set_no[MFCindex]!=0:
                NO_output[x]=Set_no[MFCindex]
            else:
                NO_output[x]=MFC_no[MFCindex]
                
            if MFC_hc[MFCindex]==0 and Set_hc[MFCindex]!=0:
                HC_output[x]=Set_hc[MFCindex]
            else:
                HC_output[x]=MFC_hc[MFCindex]
                
            if MFC_co[MFCindex]==0 and Set_co[MFCindex]!=0:
                CO_output[x]=Set_co[MFCindex]
            else:
                CO_output[x]=MFC_co[MFCindex]
                
            if MFC_n2[MFCindex]==0 and Set_n2[MFCindex]!=0:
                N2_output[x]=Set_n2[MFCindex]
            else:
                N2_output[x]=MFC_n2[MFCindex]
            
            
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
O2_oxidisationvariable=np.zeros(timesize)
O2out_oxidisationvariable=np.zeros(timesize)    
HC_reductionvariable=np.zeros(timesize)
HCout_reductionvariable=np.zeros(timesize)  
NO_oxidisationvariable=np.zeros(timesize)
NOout_oxidisationvariable=np.zeros(timesize)
O2in_startpoint=0
O2out_startpoint=0
HCin_startpoint=0
HCout_startpoint=0
NOin_startpoint=0
NOout_startpoint=0          
COin_endpoint=0
O2in_endpoint=0
HCin_endpoint=0
NOin_endpoint=0
COout_endpoint=0
O2out_endpoint=0
HCout_endpoint=0
NOout_endpoint=0
CO_reductionpoint=0
O2_oxidisationpoint=0
HC_reductionpoint=0
NO_oxidisationpoint=0
CO_in = Co_in
CO_out = Co_out
O2_in = o2_in
O2_out = o2_out
HC_in = Hc_in
HC_out = Hc_out
NO_in = No_in
NO_out = No_out
COin_aligned = np.empty(DLtimesize)
O2in_aligned = np.empty(DLtimesize)
HCin_aligned = np.empty(DLtimesize)
NOin_aligned = np.empty(DLtimesize)
COout_aligned = np.empty(DLtimesize)
O2out_aligned = np.empty(DLtimesize)
HCout_aligned = np.empty(DLtimesize)
NOout_aligned = np.empty(DLtimesize)
outputsize = np.size(HC_output)
CO_Conc = np.empty(outputsize)
HC_Conc = np.empty(outputsize)
NO_Conc = np.empty(outputsize)
Air_Conc = np.empty(outputsize)
MassFlow = np.empty(outputsize)
o2_oxidisationpoint=0
for x in range(0, timesize):
    if Co_in[x] >= 0.3:
        CO_reductionvariable[x] = 1
    if Co_in[x] < 0.3:
        CO_reductionvariable[x] = 0
    if CO_reductionvariable[x] == 1 and CO_reductionvariable[x-1] == 0:
        CO_reductionpoint = x
        break
if CO_reductionpoint>0:
    for y in range(CO_reductionpoint, timesize):
        if Co_in[y] < 0.3:
            COin_endpoint = y
            O2in_startpoint = y
            break
for x in range(O2in_startpoint, timesize):
    if o2_in[x] >= 0.15:
        O2_oxidisationvariable[x] = 1
    if o2_in[x] < 0.15:
        O2_oxidisationvariable[x] = 0
    if O2_oxidisationvariable[x] == 1 and O2_oxidisationvariable[x-1] == 0:
        o2_oxidisationpoint = x
        break
if o2_oxidisationpoint>0:
    for y in range(o2_oxidisationpoint, timesize):
        if o2_in[y] < 0.15:
            O2in_endpoint = y
            HCin_startpoint = y
            break

if CO_reductionpoint==0 and o2_oxidisationpoint==0:
    COin_latency=0
    COout_latency=0
    O2in_latency=0
    O2out_latency=0
    HCin_latency=0
    HCout_latency=0
    NOin_latency=0
    NOout_latency=0
    COin_aligned = Co_in
    O2in_aligned = o2_in
    HCin_aligned = Hc_in
    NOin_aligned = No_in
    COout_aligned = Co_out
    O2out_aligned = O2_out
    HCout_aligned = Hc_out
    NOout_aligned = No_out
else:   
    for x in range(0, timesize):
        if Co_out[x] >= 0.3:
            COout_reductionvariable[x] = 1
        if Co_out[x] < 0.3:
            COout_reductionvariable[x] = 0
        if COout_reductionvariable[x] == 1 and COout_reductionvariable[x-1] == 0:
            CO_reductionpoint = x
            break
    for y in range(CO_reductionpoint, timesize):
        if Co_out[y] < 0.3:
            COout_endpoint = y
            O2out_startpoint = y
            break
    for x in range(O2out_startpoint, timesize):
        if o2_out[x] >= 0.15:
            O2out_oxidisationvariable[x] = 1
        if o2_out[x] < 0.15:
            O2out_oxidisationvariable[x] = 0
        if O2out_oxidisationvariable[x] == 1 and O2out_oxidisationvariable[x-1] == 0:
            o2_oxidisationpoint = x
            break
    for y in range(o2_oxidisationpoint, timesize):
        if o2_out[y] < 0.15:
            O2out_endpoint = y
            HCout_startpoint = y
            break
    
    
    
    if L1_reductionpoint == 0:
        COin_latency = 0
        COout_latency = 0
    else:
        COin_latency = COin_endpoint-Reduction_endpoint
        COout_latency = COout_endpoint-Reduction_endpoint
    
    if L1_oxidisationpoint == 0:
        O2in_latency = 0
        O2out_latency = 0
    else:
        O2in_latency = O2in_endpoint-Oxidisation_endpoint
        O2out_latency = O2out_endpoint-Oxidisation_endpoint
    
    
    for x in range(0, int(COin_latency)):
        Co_in = np.append(Co_in, Co_in[-1])
    
    for x in range(0, int(O2in_latency)):
        o2_in = np.append(o2_in, o2_in[-1])
    
    for x in range(0, int(COout_latency)):
        Co_out = np.append(Co_out, Co_out[-1])
    
    for x in range(0, int(O2out_latency)):
        o2_out = np.append(o2_out, o2_out[-1])
    
    
    for x in range(0, DLtimesize):
        COin_index = DL_Timestamp.index(DL_Timestamp[x])+COin_latency
        O2in_index = DL_Timestamp.index(DL_Timestamp[x])+O2in_latency
    
        COout_index = DL_Timestamp.index(DL_Timestamp[x])+COout_latency
        O2out_index = DL_Timestamp.index(DL_Timestamp[x])+O2out_latency
        if COin_index >= COin_latency:
            COin_aligned[x] = Co_in[int(COin_index)]
        if O2in_index >= O2in_latency:
            O2in_aligned[x] = o2_in[int(O2in_index)]
    
        if COout_index >= COout_latency:
            COout_aligned[x] = Co_out[int(COout_index)]
        if O2out_index >= O2out_latency:
            O2out_aligned[x] = o2_out[int(O2out_index)]
    
    
    if (L1_reductionpoint + 100) >= L2_reductionpoint > 0:
        for y in range(L2_reductionpoint, timesize):
            if Lambda_1[y] < 0.995:
                Reduction_variable[y] = 1
            if Lambda_1[y] >= 0.995:
                Reduction_variable[y] = 0
            if Reduction_variable[y-2] == 1 and Reduction_variable[y-1] == 0 and Reduction_variable[y] == 0:
                Reduction_endpoint2 = np.append(Reduction_endpoint2, (y-2))
                break
        for x in range(HCin_startpoint, timesize):
            if Hc_in[x] >= 400:
                HC_reductionvariable[x] = 1
            if Hc_in[x] < 400:
                HC_reductionvariable[x] = 0
            if HC_reductionvariable[x] == 1 and HC_reductionvariable[x-1] == 0:
                HC_reductionpoint = x
                break
        for y in range(HC_reductionpoint, timesize):
            if Hc_in[y] < 400:
                HCin_endpoint = y
                NO_startpoint = y
                break
        for x in range(HCout_startpoint, timesize):
            if Hc_out[x] >= 400:
                HCout_reductionvariable[x] = 1
            if Hc_out[x] < 400:
                HCout_reductionvariable[x] = 0
            if HCout_reductionvariable[x] == 1 and HCout_reductionvariable[x-1] == 0:
                HC_reductionpoint = x
                break
        for y in range(HC_reductionpoint, timesize):
            if Hc_out[y] < 400:
                HCout_endpoint = y
                NOout_startpoint = y
                break
        if HCin_endpoint > 0:
            HCin_latency = HCin_endpoint-Reduction_endpoint2
        else:
            HCin_latency = COin_latency
        if HCout_endpoint > 0:
            HCout_latency = HCout_endpoint-Reduction_endpoint2
        else:
            HCout_latency = COout_latency
    
        for x in range(0, int(HCin_latency)):
            Hc_in = np.append(Hc_in, Hc_in[-1])
    
        for x in range(0, int(HCout_latency)):
            Hc_out = np.append(Hc_out, Hc_out[-1])
    
        for x in range(0, DLtimesize):
            HCin_index = DL_Timestamp.index(DL_Timestamp[x])+HCin_latency
            HCout_index = DL_Timestamp.index(DL_Timestamp[x])+HCout_latency
            if HCin_index >= HCin_latency:
                HCin_aligned[x] = Hc_in[int(HCin_index)]
            if HCout_index >= HCout_latency:
                HCout_aligned[x] = Hc_out[int(HCout_index)]
    else:
        HCin_latency = COin_latency
        HCout_latency = COout_latency
        for x in range(0, int(COin_latency)):
            Hc_in = np.append(Hc_in, Hc_in[-1])
    
        for x in range(0, int(COout_latency)):
            Hc_out = np.append(Hc_out, Hc_out[-1])
        for x in range(0, DLtimesize):
            HCin_index = DL_Timestamp.index(DL_Timestamp[x])+COin_latency
            HCout_index = DL_Timestamp.index(DL_Timestamp[x])+COout_latency
            if HCin_index >= COin_latency:
                HCin_aligned[x] = Hc_in[int(HCin_index)]
            if HCout_index >= COout_latency:
                HCout_aligned[x] = Hc_out[int(HCout_index)]
    
    
    if (L1_oxidisationpoint+100) >= L2_oxidisationpoint > 0:
        for y in range(L2_oxidisationpoint, timesize):
            if Lambda_1[y] >= 1.005:
                Oxidisation_variable[y] = 1
            if Lambda_1[y] < 1.005:
                Oxidisation_variable[y] = 0
            if Oxidisation_variable[y-2] == 1 and Oxidisation_variable[y-1] == 0 and Oxidisation_variable[y] == 0:
                Oxidisation_endpoint2 = np.append(Oxidisation_endpoint2, (y-2))
                break
        for x in range(NOin_startpoint, timesize):
            if No_in[x] >= 400:
                NO_oxidisationvariable[x] = 1
            if No_in[x] < 400:
                NO_oxidisationvariable[x] = 0
            if NO_oxidisationvariable[x] == 1 and NO_oxidisationvariable[x-1] == 0:
                NO_oxidisationpoint = x
                break
        for y in range(NO_oxidisationpoint, timesize):
            if No_in[y] < 400:
                NOin_endpoint = y
                break
        for x in range(NOout_startpoint, timesize):
            if No_out[x] >= 400:
                NOout_oxidisationvariable[x] = 1
            if No_out[x] < 400:
                NOout_oxidisationvariable[x] = 0
            if NOout_oxidisationvariable[x] == 1 and NOout_oxidisationvariable[x-1] == 0:
                NO_oxidisationpoint = x
                break
        for y in range(NO_oxidisationpoint, timesize):
            if No_out[y] < 400:
                NOout_endpoint = y
                break
        if NOin_endpoint > 0:
            NOin_latency = NOin_endpoint-Oxidisation_endpoint2
        else:
            NOin_latency = O2in_latency
        if NOout_endpoint > 0:
            NOout_latency = NOout_endpoint-Oxidisation_endpoint2
        else:
            NOout_latency = O2out_latency
        for x in range(0, int(NOin_latency)):
            No_in = np.append(No_in, No_in[-1])
    
        for x in range(0, int(NOout_latency)):
            No_out = np.append(No_out, No_out[-1])
    
        for x in range(0, DLtimesize):
            NOin_index = DL_Timestamp.index(DL_Timestamp[x])+NOin_latency
            NOout_index = DL_Timestamp.index(DL_Timestamp[x])+NOout_latency
            if NOin_index >= NOin_latency:
                NOin_aligned[x] = No_in[int(NOin_index)]
            if NOout_index >= NOout_latency:
                NOout_aligned[x] = No_out[int(NOout_index)]
    else:
        NOin_latency = O2in_latency
        NOout_latency = O2out_latency
        for x in range(0, int(O2in_latency)):
            No_in = np.append(No_in, No_in[-1])
    
        for x in range(0, int(O2out_latency)):
            No_out = np.append(No_out, No_out[-1])
        for x in range(0, DLtimesize):
            NOin_index = DL_Timestamp.index(DL_Timestamp[x])+O2in_latency
            NOout_index = DL_Timestamp.index(DL_Timestamp[x])+O2out_latency
            if NOin_index >= O2in_latency:
                NOin_aligned[x] = No_in[int(NOin_index)]
            if NOout_index >= O2out_latency:
                NOout_aligned[x] = No_out[int(NOout_index)]
#################################
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
        O2in_startpoint=y
        break
for x in range (O2in_startpoint,timesize):
    if o2_in[x]>=0.15:
        O2_oxidisationvariable[x]=1
    if o2_in[x]<0.15:
        O2_oxidisationvariable[x]=0
    if O2_oxidisationvariable[x]==1 and O2_oxidisationvariable[x-1]==0:
        o2_oxidisationpoint=x
        break
for y in range(o2_oxidisationpoint,timesize):
    if o2_in[y]<0.15:
        O2in_endpoint=y
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
        O2out_startpoint=y
        break
for x in range (O2out_startpoint,timesize):
    if o2_out[x]>=0.15:
        O2out_oxidisationvariable[x]=1
    if o2_out[x]<0.15:
        O2out_oxidisationvariable[x]=0
    if O2out_oxidisationvariable[x]==1 and O2out_oxidisationvariable[x-1]==0:
        o2_oxidisationpoint=x
        break
for y in range(o2_oxidisationpoint,timesize):
    if o2_out[y]<0.15:
        O2out_endpoint=y
        HCout_startpoint=y
        break


CO_in=Co_in
CO_out=Co_out
O2_in=o2_in
O2_out=o2_out
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

if L1_oxidisationpoint==0:
    O2in_latency=0
    O2out_latency=0
else:
    O2in_latency=O2in_endpoint-Oxidisation_endpoint
    O2out_latency=O2out_endpoint-Oxidisation_endpoint
    
COin_aligned=np.empty(DLtimesize)
O2in_aligned=np.empty(DLtimesize)
HCin_aligned=np.empty(DLtimesize)
NOin_aligned=np.empty(DLtimesize)
COout_aligned=np.empty(DLtimesize)
O2out_aligned=np.empty(DLtimesize)
HCout_aligned=np.empty(DLtimesize)
NOout_aligned=np.empty(DLtimesize)

for x in range (0,int(COin_latency)):
    Co_in=np.append(Co_in,Co_in[-1])

for x in range(0,int(O2in_latency)):
    o2_in=np.append(o2_in,o2_in[-1])

for x in range(0,int(COout_latency)):
    Co_out=np.append(Co_out,Co_out[-1])

for x in range(0,int(O2out_latency)):
    o2_out=np.append(o2_out,o2_out[-1])
    
    
for x in range (0,DLtimesize):
    COin_index=DL_Timestamp.index(DL_Timestamp[x])+COin_latency
    O2in_index=DL_Timestamp.index(DL_Timestamp[x])+O2in_latency
    
    COout_index=DL_Timestamp.index(DL_Timestamp[x])+COout_latency
    O2out_index=DL_Timestamp.index(DL_Timestamp[x])+O2out_latency
    if COin_index>=COin_latency:
        COin_aligned[x]=Co_in[int(COin_index)]
    if O2in_index>=O2in_latency:
        O2in_aligned[x]=o2_in[int(O2in_index)]
        
    if COout_index>=COout_latency:
        COout_aligned[x]=Co_out[int(COout_index)]
    if O2out_index>=O2out_latency:
        O2out_aligned[x]=o2_out[int(O2out_index)]
   
   
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
        if Hc_in[x]>=400:
            HC_reductionvariable[x]=1
        if Hc_in[x]<400:
            HC_reductionvariable[x]=0
        if HC_reductionvariable[x]==1 and HC_reductionvariable[x-1]==0:
            HC_reductionpoint=x
            break
    for y in range(HC_reductionpoint,timesize):
        if Hc_in[y]<400:
            HCin_endpoint=y
            NO_startpoint=y
            break
    for x in range (HCout_startpoint,timesize):
        if Hc_out[x]>=400:
            HCout_reductionvariable[x]=1
        if Hc_out[x]<400:
            HCout_reductionvariable[x]=0
        if HCout_reductionvariable[x]==1 and HCout_reductionvariable[x-1]==0:
            HC_reductionpoint=x
            break
    for y in range(HC_reductionpoint,timesize):
        if Hc_out[y]<400:
            HCout_endpoint=y
            NOout_startpoint=y
            break
    if HCin_endpoint>0:
        HCin_latency=HCin_endpoint-Reduction_endpoint2
    else:
        HCin_latency=COin_latency
    if HCout_endpoint>0:
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
        NOin_latency=NOin_endpoint-Oxidisation_endpoint2           
    else:
        NOin_latency=O2in_latency
    if NOout_endpoint>0:
        NOout_latency=NOout_endpoint-Oxidisation_endpoint2
    else:
        NOout_latency=O2out_latency
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
    NOin_latency=O2in_latency
    NOout_latency=O2out_latency
    for x in range (0,int(O2in_latency)):
        No_in=np.append(No_in,No_in[-1])
        
    for x in range(0,int(O2out_latency)):
        No_out=np.append(No_out,No_out[-1])
    for x in range (0,DLtimesize): 
        NOin_index=DL_Timestamp.index(DL_Timestamp[x])+O2in_latency
        NOout_index=DL_Timestamp.index(DL_Timestamp[x])+O2out_latency
        if NOin_index>=O2in_latency:            
            NOin_aligned[x]=No_in[int(NOin_index)]
        if NOout_index>=O2out_latency:
            NOout_aligned[x]=No_out[int(NOout_index)]
                
            

outputsize=np.size(HC_output)
CO_Conc=np.empty(outputsize)
HC_Conc=np.empty(outputsize)
NO_Conc=np.empty(outputsize)
MassFlow=np.empty(outputsize)
###################################################
for x in range(0,outputsize):
    if CO_output[x]==0:
        CO_Conc[x]=0
    else:
        CO_Conc[x]=CO_output[x]/(CO_output[x]+HC_output[x]+Air_output[x]+NO_output[x]+N2_output[x])*100
    if HC_output[x]==0:
        HC_Conc[x]=0
    else:
        HC_Conc[x]=HC_output[x]/(CO_output[x]+HC_output[x]+Air_output[x]+NO_output[x]+N2_output[x])*3000000
    if NO_output[x]==0:
        NO_Conc[x]=0
    else:
        NO_Conc[x]=NO_output[x]/(CO_output[x]+HC_output[x]+Air_output[x]+NO_output[x]+N2_output[x])*1000000
    
    MassFlow[x]=(CO_output[x]+HC_output[x]+Air_output[x]+NO_output[x]+N2_output[x])/60*(101325/288/296)

    
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
    
    
    

Raw_Data_df=pandas.DataFrame({"Timestamp":DL_Timestamp,"Inlet Lambda":Lambda_1,"Outlet Lambda":Lambda_2,"Inlet temp":Inlet_temp,"Bed Temp":Bed_temp,"Other temp":Other_temp,"Calculated CO IN":CO_Conc,"Calculated HC IN":HC_Conc,"Calculated NO IN":NO_Conc,"Inlet CO":CO_in,"Inlet O2":O2_in,"Inlet HC":HC_in,"Inlet NO":NO_in,"Outlet CO":CO_out,"Outlet O2":O2_out,"Outlet HC":HC_out,"Outlet NO":NO_out,"CO injection":CO_output,"O2 injection":Air_output ,"HC injection":HC_output,"NO injection":NO_output,"Mass Flow":MassFlow})
Raw_Data_df.to_csv("C:/PythonAnalyser/Raw Data.csv",index=False, header=False)

Lightoff_latency_df=pandas.DataFrame({"Test Startpoint":test_startpoint,"Test Endpoint":test_endpoint,"Selected analyser 1":selected_analysers[0],"CO in latency":COin_latency,"O2 in latency":O2in_latency,"HC in latency":HCin_latency,"NO in latency":NOin_latency,"Scale Factor CO in":ScaleFactor_COin,"Scale Factor HC in":ScaleFactor_HCin,"Scale Factor NO in":ScaleFactor_NOin,"Selected Analyser 2":selected_analysers[1],"CO out latency":COout_latency,"O2 out latency":O2out_latency,"HC out latency":HCout_latency,"NO out latency":NOout_latency,"Scale Factor CO out":ScaleFactor_COout,"Scale Factor HC out":ScaleFactor_HCout,"Scale Factor NO out":ScaleFactor_NOout})
Lightoff_latency_df.to_csv("C:/PythonAnalyser/Lightoff Latency.csv",index=False, header=False)