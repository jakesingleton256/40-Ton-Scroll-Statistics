import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import glob2
from matplotlib.backends.backend_pdf import PdfPages
import os
for name in glob2.glob('*.csv'):
    print(name)
    x,y=[[],[]]
    file1, ext=os.path.splitext(name)
    Filename=str(file1)+'.csv'
    df=pd.read_csv(Filename,skiprows=12)
    columns=df.dtypes.index
    Psuc=np.array(df[columns[0]])
    Tsuc=np.array(df[columns[1]])
    Pdis=np.array(df[columns[2]])
    Tdis=np.array(df[columns[3]])
    DT_sup=np.array(df[columns[28]])
    DT_sub=np.array(df[columns[29]])
    w=np.array(df[columns[6]])
    # print(x,y)
    # xbar=np.nanmean(x)
    # ybar=sum(y)/len(y)
    # print(xbar,ybar)
    # xmin=min(x)
    # xmax=max(x)
    # ymin=min(y)
    # ymax=max(y)
    
    # errorx=xmax-xmin
    # plt.bar(1, xbar)
    # plt.bar(3, ybar)
    # plt.errorbar(1,xbar,yerr=([xbar-xmin],[xmax-xbar]),fmt='k',capsize=6)
    # plt.errorbar(3,ybar,yerr=([ybar-ymin],[ymax-ybar]),fmt='k',capsize=6)
    # plt.xlim(0,4)
    # plt.ylim(0,80)
    # plt.text(.5,70,'n=10')
    # plt.text(2.5,30,'n=15')
    # plt.xticks([1,3],['Before 2010','After 2010'])

    #calculate averages
    Ps_avg=np.mean(Psuc)
    Pd_avg=np.mean(Pdis)
    Ts_avg=np.mean(Tsuc)
    N=len(Psuc)
    sumPs=0
    sumTs=0
    sumPd=0

    #calculate standard deviation
    for i in range(N):
        sumPs+=((Psuc[i]-Ps_avg)**2)/(N-1)
        sumTs+=((Tsuc[i]-Ts_avg)**2)/(N-1)
        sumPd+=((Pdis[i]-Pd_avg)**2)/(N-1)
    s_x_Psuc=np.sqrt(sumPs)
    s_x_Tsuc=np.sqrt(sumTs)
    s_x_Pdis=np.sqrt(sumPd)
    print("s_x_Psuc=",s_x_Psuc)
    print("s_x_Tsuc=",s_x_Tsuc)
    print("s_x_Pdis=",s_x_Pdis)

    #Calculate random uncertainty
    s_Psuc=s_x_Psuc/np.sqrt(N)
    s_Tsuc=s_x_Tsuc/np.sqrt(N)
    s_Pdis=s_x_Pdis/np.sqrt(N)
    
    print("s_Psuc=",s_Psuc)
    print("s_Tsuc=",s_Tsuc)
    print("s_Pdis=",s_Pdis)

    #Systematic uncertainty
    sigma_sys_P=0.375
    sigma_sys_T=0.2

    #total uncertainty
    u_Psuc=np.sqrt((s_Psuc**2)+(sigma_sys_P**2))
    u_Tsuc=np.sqrt((s_Tsuc**2)+(sigma_sys_T**2))
    u_Pdis=np.sqrt((s_Pdis**2)+(sigma_sys_P**2))
    
    print('Psuc = ',Ps_avg,'\u00B1',u_Psuc)
    print('Tsuc = ',Ts_avg,'\u00B1',u_Tsuc)
    print('Pdis = ',Pd_avg,'\u00B1',u_Pdis)

    data={'Psuc':[u_Psuc,], 'Tsuc':[u_Tsuc,],'Pdis':[u_Pdis,]}
    d=pd.DataFrame(data,columns=['Psuc','Tsuc','Pdis'])
    d.to_csv('total uncertainty.csv')

    for num in range(N,10):
        Ps1=df.iloc[0:N,0]
        print(Ps1)
        

