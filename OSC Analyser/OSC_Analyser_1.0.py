# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 10:53:32 2021

@author: Home
"""
import pandas 
import numpy as np
import statistics

inputdata='C:/PythonAnalyser/TargetFiles.csv'
tcol=['Fnames']
targetfiles=pandas.read_csv(inputdata,names=tcol,encoding='cp1252')
tfnames=(targetfiles.Fnames.tolist())

filename=tfnames[0]
cols=['Timestamp','DLI','DLO']
parse_dates=['Timestamp']
data=pandas.read_csv(filename,names=cols,skiprows=2,parse_dates=parse_dates)
Timestamp=np.array(data.Timestamp.tolist(),dtype=object)
Dli=np.array(data.DLI.tolist())
Dlo=np.array(data.DLO.tolist())
timesize=np.size(Timestamp)
sliced_variable=0 #variable used to slice Timestamp
sliced_variable1=0 #variable used to slice Timestamp[x+1]
sliced_variable2=0 #variable used to slice Timestamp1
hour=np.zeros(timesize,dtype=int)
minute=np.zeros(timesize,dtype=int)
second=np.zeros(timesize,dtype=int)
pointsecond=np.zeros(timesize,dtype=int)
TimevalETAS=np.zeros(timesize,dtype=float)#time value in seconds
Timestamp1=np.empty(timesize,dtype=object)#used to write the correct date and time values to
Hour=np.empty(timesize,dtype=object) #these arrays differ to the previous arrays of a similar name as to allow them to be sliced etc.
Minute=np.empty(timesize,dtype=object)
Second=np.empty(timesize,dtype=object)
Pointsecond=np.empty(timesize,dtype=object)
duplicatesarray=np.empty([],dtype=int)
for x in range (0,timesize-1):
    sliced_variable=Timestamp[x]
    hour[x]=sliced_variable[11:13]
    minute[x]=sliced_variable[14:16]
    second[x]=sliced_variable[17:19]
    pointsecond[x]=sliced_variable[20:21]
    Hour[x]=hour[x]
    Minute[x]=minute[x]
    Second[x]=second[x]
    Pointsecond[x]=pointsecond[x]    
    TimevalETAS[x]=(hour[x]*3600+minute[x]*60+second[x]+pointsecond[x]/10)
    Timestamp1=Timestamp
print("data read in")
for x in range (0,timesize-4):
    if TimevalETAS[x]==TimevalETAS[x+1]: 
        if TimevalETAS[x]-TimevalETAS[x-1]<0.15:
            if TimevalETAS[x+2]-TimevalETAS[x+1]<0.15:
                duplicatesarray=np.append(duplicatesarray,x)
            else:
                if pointsecond[x+1]==9:
                    pointsecond[x+1]=0
                    Pointsecond[x+1]=pointsecond[x+1]
                    if second[x+1]==59:
                        if minute[x+1]==59:
                            hour[x+1]=hour[x+1]+1
                            Hour[x+1]=hour[x+1]
                            minute[x+1]=0
                            Minute[x+1]=minute[x+1]
                        else: 
                            second[x+1]=0
                            Second[x+1]=second[x+1]
                            minute[x+1]=minute[x+1]+1
                            Minute[x+1]=minute[x+1]
                    else: 
                        second[x+1]=second[x+1]+1
                        Second[x+1]=second[x+1]
                else:
                    pointsecond[x+1]=pointsecond[x+1]+1
                    Pointsecond[x+1]=pointsecond[x+1]
                if hour[x+1]<10:
                    Hour[x+1]="{:02}".format(hour[x+1])
                else:
                    Hour[x+1]=hour[x+1]
                if minute[x+1]<10:
                    Minute[x+1]="{:02}".format(minute[x+1])
                else:
                    Minute[x+1]=minute[x+1]
                if second[x+1]<10:
                    Second[x+1]="{:02}".format(second[x+1])
                else:
                    Second[x+1]=second[x+1]
                sliced_variable1=Timestamp[x+1]
                Timestamp1[x+1]=sliced_variable1[0:11]+str(Hour[x+1])+":" +str(Minute[x+1])+":"+str(Second[x+1])+":"+str(Pointsecond[x+1])+" "
                TimevalETAS[x+1]=(hour[x+1]*3600+minute[x+1]*60+second[x+1]+pointsecond[x+1]/10)
   
        else:
            if pointsecond[x]==0:
                pointsecond[x]=9
                Pointsecond[x]=pointsecond[x]
                if second[x]==0:
                    if minute[x]==0:
                        hour[x]=hour[x]-1
                        Hour[x]=hour[x]
                        minute[x]=59
                        Minute[x]=minute[x]
                        second[x]=59
                        Second[x]=second[x]
                    else:
                        minute[x]=minute[x]-1
                        Minute[x]=minute[x]
                        second[x]=59
                        Second[x]=second[x]
                else:
                    second[x]=second[x]-1
                    Second[x]=second[x]
            else:
                pointsecond[x]=pointsecond[x]-1
                Pointsecond[x]=pointsecond[x]
            if hour[x]<10:
                Hour[x]="{:02}".format(hour[x])
            else:
                Hour[x]=hour[x]
            if minute[x]<10:
                Minute[x]="{:02}".format(minute[x])
            else:
                Minute[x]=minute[x]
            if second[x]<10:
                Second[x]="{:02}".format(second[x])
            else:
                Second[x]=second[x]
            sliced_variable2=Timestamp[x]
            Timestamp1[x]=sliced_variable2[0:11]+str(Hour[x])+":"+str(Minute[x])+":"+str(Second[x])+":"+str(Pointsecond[x])+" "
            TimevalETAS[x]=(hour[x]*3600+minute[x]*60+second[x]+pointsecond[x]/10)
print("duplicates removed")
Timestamp1=np.delete(Timestamp1,duplicatesarray,axis=0)  
Pointsecond=np.delete(Pointsecond,duplicatesarray,axis=0)
TimevalETAS=np.delete(TimevalETAS,duplicatesarray,axis=0)
Dlo=np.delete(Dlo,duplicatesarray,axis=0)
Dli=np.delete(Dli,duplicatesarray,axis=0)            

             
DLfilename=tfnames[1]
DLcols=['Date','Time', 'Lambda1', 'Lambda2', 'inlet_temp', 'bed_temp']
DLdata=pandas.read_csv(DLfilename, names=DLcols,delim_whitespace=(True),skiprows=2, usecols=[0,1,4,5,9,10]) 
MFCfilename=tfnames[2]
MFCcols=['MDate','MTime','CO','Air','N2','Airbal','Set_CO','Set_Air','Set_N2','Set_Airbal']
MFCdata=pandas.read_csv(MFCfilename, names=MFCcols,delim_whitespace=(True),skiprows=2, usecols=[0,1,2,4,6,7,13,15,17,18])
DLtime=np.array(DLdata.Time.tolist())
DLdate=np.array(DLdata.Date.tolist())
DLtimestamp=np.empty(np.size(DLtime), dtype=object)
for x in range (0,np.size(DLtime)):        
    currentdate=DLdate[x]
    CurrentdateA=currentdate[0:6]
    CurrentdateB=currentdate[6:8]
    DLtimestamp[x]= CurrentdateA + "20" + CurrentdateB + " " + DLtime[x]
DLlambda1=np.array(DLdata.Lambda1.tolist())
DLlambda2=np.array(DLdata.Lambda2.tolist())   
DLinlet=np.array(DLdata.inlet_temp.tolist())
DLbed=np.array(DLdata.bed_temp.tolist())
timesize1=np.size(Timestamp1)
DLtimestamp=DLtimestamp.tolist()
Timestamptrunc=list(range(timesize1))
DLsize=np.size(DLtimestamp)
day=np.empty(timesize1, dtype=object)
month=np.empty(timesize1,dtype=object)
year=np.empty(timesize1,dtype=object)
Etas_day=np.empty(timesize1, dtype=object)
Etas_month=np.empty(timesize1, dtype=object)
Etas_year=np.empty(timesize1, dtype=object)
MFCtime=np.array(MFCdata.MTime.tolist())
MFCdate=np.array(MFCdata.MDate.tolist())
MFCtimestamp=np.empty(np.size(MFCtime), dtype=object)
for x in range (0,np.size(MFCtime)-1):
    currentdate=MFCdate[x]
    CurrentdateA=currentdate[0:6]
    CurrentdateB=currentdate[6:8]
    MFCtimestamp[x]= CurrentdateA + "20" + CurrentdateB + " " + MFCtime[x]
MFCtimestamp=MFCtimestamp.tolist()
MFC_n2=np.array(MFCdata.N2.tolist())
MFC_Airbal=np.array(MFCdata.Airbal.tolist())
MFC_co=np.array(MFCdata.CO.tolist())
MFC_Air=np.array(MFCdata.Air.tolist())
MFC_SetN2=np.array(MFCdata.Set_N2.tolist())
MFC_SetAirbal=np.array(MFCdata.Set_Airbal.tolist())
MFC_SetCo=np.array(MFCdata.Set_CO.tolist())
MFC_SetAir=np.array(MFCdata.Set_Air.tolist())
Set_Flow=(MFC_SetN2+MFC_SetAirbal)*101325/288/296/60 
offset=-13
DLindex=0
MFCindex=0
Lambda1_output=np.zeros(timesize1)
Lambda2_output=np.zeros(timesize1)
inlettemp_output=np.zeros(timesize1)
bedtemp_output=np.zeros(timesize1)
N2_output=np.empty(timesize1)
Airbal_output=np.empty(timesize1)
CO_output=np.empty(timesize1)
Air_output=np.empty(timesize1)
Flow_output=np.empty(timesize1)

for x in range(0,timesize1):
    year[x]=Timestamp1[x][0:4]
    month[x]=Timestamp1[x][5:7]
    day[x]=Timestamp1[x][8:10]
    Timestamp1[x]=day[x]+"/"+month[x]+"/"+year[x]+" "+Timestamp1[x][11:21]
    Timestampsize=len(Timestamp1[x]) 
    Timestamptrunc[x]=Timestamp1[x][0:(Timestampsize-2)]   
    if (Timestamptrunc[x] in DLtimestamp):
        DLindex=DLtimestamp.index(Timestamptrunc[x])+offset
        if DLindex<np.size(DLtimestamp) and DLindex>0:
            Lambda1_output[x]=DLlambda1[DLindex]
            Lambda2_output[x]=DLlambda2[DLindex]
            inlettemp_output[x]=DLinlet[DLindex]
            bedtemp_output[x]=DLbed[DLindex]
    if (Timestamptrunc[x] in MFCtimestamp):
        MFCindex=MFCtimestamp.index(Timestamptrunc[x])+offset
        if MFCindex<np.size(MFCtimestamp) and MFCindex>0:
            if MFC_Air[MFCindex]==0 and MFC_SetAir[MFCindex]!=0:                
                Air_output[x]=MFC_SetAir[MFCindex]
            else:
                Air_output[x]=MFC_Air[MFCindex]
                
            if MFC_n2[MFCindex]==0 and MFC_SetN2[MFCindex]!=0:                
                N2_output[x]=MFC_SetN2[MFCindex]
            else:               
                N2_output[x]=MFC_n2[MFCindex]
                
            if MFC_co[MFCindex]==0 and MFC_SetCo[MFCindex]!=0:                
                CO_output[x]=MFC_SetCo[MFCindex]
            else:
                CO_output[x]=MFC_co[MFCindex]
                
            if MFC_Airbal[MFCindex]==0 and MFC_SetAirbal[MFCindex]!=0:
                Airbal_output[x]=MFC_SetAirbal[MFCindex]
            else:
                Airbal_output[x]=MFC_Airbal[MFCindex]
                
Flow_output=(N2_output+Airbal_output)*101325/288/296/60
         
Etas_selection_filename="C:/PythonAnalyser/Lambda_select.csv"
select_cols=['Selections']

selection_data=pandas.read_csv(Etas_selection_filename,names=select_cols)         
selected_lambdas=np.array(selection_data.Selections.tolist())
Dlisize=np.size(Dli)               
ETAS2_Dli=np.empty(Dlisize)
ETAS2_Dlo=np.empty(Dlisize)
Etas_offset=-150                    


if selected_lambdas[0]>2 or selected_lambdas[1]>2:
    ETAS2filename=tfnames[3]
    ETAScols=['Timestamp','DLI','DLO']
    Etas2data=pandas.read_csv(ETAS2filename,names=ETAScols,skiprows=2)
    ETAS_Timestamp=np.array(Etas2data.Timestamp.tolist())
    ETAS_Timestamp=ETAS_Timestamp.tolist()
    etas_size=np.size(ETAS_Timestamp)
    Etas_year=np.empty(etas_size,dtype=object)
    Etas_month=np.empty(etas_size,dtype=object)
    Etas_day=np.empty(etas_size,dtype=object)
    Etas_hour=np.empty(etas_size,dtype=object)
    Etas_minute=np.empty(etas_size,dtype=object)
    Etas_second=np.empty(etas_size,dtype=object)
    Etas_pointsecond=np.empty(etas_size,dtype=object)
    Etas_Timeval=np.empty(etas_size,dtype=object)
    Pointsecond[x]=pointsecond[x]    
    TimevalETAS[x]=(hour[x]*3600+minute[x]*60+second[x]+pointsecond[x]/10)
    ETAS_Dli=np.array(Etas2data.DLI.tolist())
    ETAS_Dlo=np.array(Etas2data.DLO.tolist()) 
    for x in range(0,etas_size):
        sliced_variable=ETAS_Timestamp[x]
        Etas_year[x]=ETAS_Timestamp[x][0:4]
        Etas_month[x]=ETAS_Timestamp[x][5:7]
        Etas_day[x]=ETAS_Timestamp[x][8:10]
        ETAS_Timestamp[x]=Etas_day[x]+"/"+Etas_month[x]+"/"+Etas_year[x]+" "+ETAS_Timestamp[x][11:21]
        Etas_hour[x]=sliced_variable[11:13]
        Etas_minute[x]=sliced_variable[14:16]
        Etas_second[x]=sliced_variable[17:19]
        Etas_pointsecond[x]=sliced_variable[20:21]
        Etas_Timeval[x]=(int(Etas_hour[x])*3600+int(Etas_minute[x])*60+int(Etas_second[x])+int(Etas_pointsecond[x])/10)
    prior_index=0
    for x in range(0,timesize1):
        ETAS_index=0
        if (Timestamp1[x] in ETAS_Timestamp):
            ETAS_index=ETAS_Timestamp.index(Timestamp1[x])+Etas_offset
            if ETAS_index<np.size(ETAS_Timestamp) and ETAS_index>0:
                ETAS2_Dli[x]=ETAS_Dli[ETAS_index]
                ETAS2_Dlo[x]=ETAS_Dlo[ETAS_index]
        else:
            for y in range(0,etas_size):
                if ((TimevalETAS[x]-Etas_Timeval[y])**2)**0.5<0.25:
                    if Etas_Timeval[y]>TimevalETAS[x]:
                        if ETAS_index==0:
                            ETAS_index=y+Etas_offset
                        else:
                            break
            if ETAS_index==0:
                print("error")
                ETAS_index=prior_index
            if ETAS_index<np.size(ETAS_Timestamp) and ETAS_index>0:
                ETAS2_Dli[x]=ETAS_Dli[ETAS_index]
                ETAS2_Dlo[x]=ETAS_Dlo[ETAS_index]
        prior_index=ETAS_index

    if selected_lambdas[0]==1:
        Dli=Dli
    if selected_lambdas[0]==2:
        Dli=Dlo
    if selected_lambdas[0]==3:
        Dli=ETAS2_Dli
    if selected_lambdas[0]==4:
        Dli=ETAS2_Dlo
    if selected_lambdas[1]==1:
        Dlo=Dli
    if selected_lambdas[1]==2:
        Dlo=Dlo
    if selected_lambdas[1]==3:
        Dlo=ETAS2_Dli
    if selected_lambdas[1]==4:
        Dlo=ETAS2_Dlo                
           
ETAS_df= pandas.DataFrame({"ETAS timeval":Timestamptrunc,"Pointsec":Pointsecond ,"DLI":Dli,"DLO":Dlo,"Lambda1":Lambda1_output,"Lambda2":Lambda2_output,"Inlet temp":inlettemp_output,"Bed temp":bedtemp_output,"Aribal":Air_output,"CO":CO_output,"Flow":Flow_output})
ETAS_df.to_csv('C:/PythonAnalyser/OSC data.csv', index=False, header=False)   

DLIoxidisationpoint=0
DLItp=0
DLOoxidisationpoint=0
DLOtp=0
Oxidisation_variable=np.zeros(timesize1)
Oxidisation_variable2=np.zeros(timesize1)
DLIturningpoint=np.array([],dtype=int)
DLOturningpoint=np.array([])
Reduction_variable=np.zeros(timesize1)
Reduction_variable2=np.zeros(timesize1)
DLIreductionpoint=0
DLOreductionpoint=0
DLIturningpoint2=np.array([],dtype=int)
DLOturningpoint2=np.array([])
DLItp2=0
DLOtp2=0
oxidisationpoint=0
offset_inlet_lambda1=np.array([])
offset_outlet_lambda1=np.array([])
offset_inlet_lambda2=np.array([])
offset_outlet_lambda2=np.array([])
PreOxidisation_array=np.empty(10)
PreReduction_array=np.empty(10)
PreOxidisation_array2=np.empty(10)
PreReduction_array2=np.empty(10)
for x in range (0,timesize1,10):
    if Dli[x] >= 1.01:
        Oxidisation_variable[x]=1
    if Dli[x] < 1.01:        
        Oxidisation_variable[x]=0
    if Oxidisation_variable[x]==1 and Oxidisation_variable[x-10]==0:
        DLIoxidisationpoint=x
        for y in range(DLIoxidisationpoint,-1,-1):
            PreOxidisation_array=Dli[y-10:y]
            if y<10:
                break
            if Dli[y]-np.min(PreOxidisation_array)<0.0005:
                DLIturningpoint=np.append(DLIturningpoint,y)
                offset_inlet_lambda1=np.append(offset_inlet_lambda1,(1-statistics.mean(Dli[y-50:y])))
                DLItp=DLIturningpoint[0]
                break
for w in range(DLItp,timesize1,10):
    if Dlo[w] >= 1.01:
        Oxidisation_variable2[w]=1
    if Dlo[w] < 1.01:
        Oxidisation_variable2[w]=0
    if Oxidisation_variable2[w]==1 and Oxidisation_variable2[w-10]==0:
        DLOoxidisationpoint=w        
        for z in range (DLOoxidisationpoint,-1,-1):
            PreOxidisation_array2=Dlo[z-10:z]
            if z<10:
                break
            if Dlo[z]-np.min(PreOxidisation_array2)<0.0005:
                DLOturningpoint=np.append(DLOturningpoint,z)
                offset_outlet_lambda1=np.append(offset_outlet_lambda1,(1-statistics.mean(Dlo[z-50:z])))
                DLOtp=z
                break

for x in range (0,timesize1,10):
    if Dli[x] <= 0.99:
        Reduction_variable[x]=1
    if Dli[x] > 0.99:
        Reduction_variable[x]=0
    if Reduction_variable[x]==1 and Reduction_variable[x-10]==0:
        DLIreductionpoint=x  
        for y in range(DLIreductionpoint,-1,-1):
            PreReduction_array=Dli[y-10:y]
            if y<10:
                break
            if np.max(PreReduction_array)-Dli[y]<0.0005:
                DLIturningpoint2=np.append(DLIturningpoint2,y)
                offset_inlet_lambda2=np.append(offset_inlet_lambda2,(1-statistics.mean(Dli[y-50:y])))
                DLItp2=DLIturningpoint2[0]
                break
for w in range (DLItp2,timesize1,10):
    if Dlo[w]<=0.99:
        Reduction_variable2[w]=1
    if Dlo[w] > 0.99:
        Reduction_variable2[w]=0
    if Reduction_variable2[w]==1 and Reduction_variable2[w-10]==0:
        DLOreductionpoint=w
        for z in range (DLOreductionpoint,-1,-1):
            PreReduction_array2=Dlo[z-10:z]
            if z<10:
                break
            if np.max(PreReduction_array2)-Dlo[z]<0.0005:
                DLOturningpoint2=np.append(DLOturningpoint2,z)
                offset_outlet_lambda2=np.append(offset_outlet_lambda2,(1-statistics.mean(Dlo[z-50:z])))
                DLOtp2=z
                break

Dlisize=np.size(DLIturningpoint)
Dlisize2=np.size(DLIturningpoint2)
Dlosize=np.size(DLOturningpoint)
Dlosize2=np.size(DLOturningpoint2)
TPs_sizearray=np.array([Dlisize,Dlisize2,Dlosize,Dlosize2])
TPs_min=np.min(TPs_sizearray)
TPs_max=np.max(TPs_sizearray)

while TPs_min<TPs_max:
    if Dlisize==TPs_min:
        DLIturningpoint=np.append(DLIturningpoint,0)
        Dlisize=np.size(DLIturningpoint)
        TPs_sizearray=np.array([Dlisize,Dlisize2,Dlosize,Dlosize2])
        TPs_min=np.min(TPs_sizearray)
        TPs_max=np.max(TPs_sizearray)
        if DLIturningpoint[-3]==0:
            print ("DLIturningpoint error!")
            break
    elif Dlisize2==TPs_min:
        DLIturningpoint2=np.append(DLIturningpoint2,0)
        Dlisize2=np.size(DLIturningpoint2)
        TPs_sizearray=np.array([Dlisize,Dlisize2,Dlosize,Dlosize2])
        TPs_min=np.min(TPs_sizearray)
        TPs_max=np.max(TPs_sizearray)
        if DLIturningpoint2[-3]==0:
            print("DLIturningpoint2 error!")
            break
    elif Dlosize==TPs_min:
        DLOturningpoint=np.append(DLOturningpoint,0)
        Dlosize=np.size(DLOturningpoint)
        TPs_sizearray=np.array([Dlisize,Dlisize2,Dlosize,Dlosize2])
        TPs_min=np.min(TPs_sizearray)
        TPs_max=np.max(TPs_sizearray)
        if DLOturningpoint[-3]==0:
            print("DLOturningpoint error!")
            break
    elif Dlosize2==TPs_min:
        DLOturningpoint2=np.append(DLOturningpoint2,0)
        Dlosize2=np.size(DLOturningpoint2)
        TPs_sizearray=np.array([Dlisize,Dlisize2,Dlosize,Dlosize2])
        TPs_min=np.min(TPs_sizearray)
        TPs_max=np.max(TPs_sizearray)
        if DLOturningpoint2[-3]==0:
            print("DLOturningpoint2 error!")
            break  
        
offset_outlet_size=np.size(offset_outlet_lambda1)
offset_outlet_size2=np.size(offset_outlet_lambda2)
offset_inlet_size=np.size(offset_inlet_lambda1)
offset_inlet_size2=np.size(offset_inlet_lambda2) 
offset_sizearray=np.array([offset_inlet_size,offset_inlet_size2,offset_outlet_size,offset_outlet_size2])
offset_max=np.max(offset_sizearray)
offset_min=np.min(offset_sizearray)

while offset_min<offset_max:
    if offset_inlet_size==offset_min:
        offset_inlet_lambda1=np.append(offset_inlet_lambda1,0)
        offset_inlet_size=np.size(offset_inlet_lambda1)
        offset_sizearray=np.array([offset_inlet_size,offset_inlet_size2,offset_outlet_size,offset_outlet_size2])
        offset_max=np.max(offset_sizearray)
        offset_min=np.min(offset_sizearray)
        if offset_inlet_lambda1[-3]==0:
            print('offset_inlet_lambda1 error')
            break
    elif offset_inlet_size2==offset_min:
        offset_inlet_lambda2=np.append(offset_inlet_lambda2,0)
        offset_inlet_size2=np.size(offset_inlet_lambda2)
        offset_sizearray=np.array([offset_inlet_size,offset_inlet_size2,offset_outlet_size,offset_outlet_size2])
        offset_max=np.max(offset_sizearray)
        offset_min=np.min(offset_sizearray)
        if offset_inlet_lambda2[-3]==0:
            print("offset_inlet_lambda2 error")
            break
    elif offset_outlet_size==offset_min:
        offset_outlet_lambda1=np.append(offset_outlet_lambda1,0)
        offset_outlet_size=np.size(offset_outlet_lambda1)
        offset_sizearray=np.array([offset_inlet_size,offset_inlet_size2,offset_outlet_size,offset_outlet_size2])
        offset_max=np.max(offset_sizearray)
        offset_min=np.min(offset_sizearray)
        if offset_outlet_lambda1[-3]==0:
            print("offset_outlet_lambda1 error")
            break
    elif offset_outlet_size2==offset_min:
        offset_outlet_lambda2=np.append(offset_outlet_lambda2,0)
        offset_outlet_size2=np.size(offset_outlet_lambda2)
        offset_sizearray=np.array([offset_inlet_size,offset_inlet_size2,offset_outlet_size,offset_outlet_size2])
        offset_max=np.max(offset_sizearray)
        offset_min=np.min(offset_sizearray)
        if offset_outlet_lambda2[-3]==0:
            print("offset_outlet_lambda2 error")
            break


turningpoints_df= pandas.DataFrame({"DLIoxidisationTP":DLIturningpoint, "DLOoxidisationTP":DLOturningpoint,"DLIreductionTP":DLIturningpoint2,"DLOreductionTP":DLOturningpoint2, "Offset inlet lambda1":offset_inlet_lambda1, "Offset outlet lambda1":offset_outlet_lambda1,"Offset inlet lambda2":offset_inlet_lambda2,"Offset outlet lambda2":offset_outlet_lambda2})  
turningpoints_df.to_csv("C:/PythonAnalyser/Oxidisation and Reduction turning points.csv",index=False,header=False)                  
    
  