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
    
    t=np.arange(1,1000)
    P_s=np.ones(999)*54.79
    P_d=np.ones(999)*138.9
    T_s=np.ones(999)*65
    DT_s=np.ones(999)*20
    
    with PdfPages('45E-120C-20S.pdf') as pdf:
        plt.figure(1)
        plt.plot(t,Psuc)
        plt.plot(t,P_s)
        plt.plot(t,P_s+0.5,'r')
        plt.plot(t,P_s-0.5,'r')
        plt.xlabel('Number of Samples')
        plt.ylabel('Suction Pressure (psi)')
        plt.title('Suction Pressure')
        plt.text(0,54.35,'Psuc =  %s \u00B1  %s' %(round(Ps_avg,2),round(u_Psuc,4)))
        pdf.savefig()
        plt.close()

        plt.figure(2)
        plt.plot(t,Pdis)
        plt.plot(t,P_d)
        plt.plot(t,P_d+0.5,'r')
        plt.plot(t,P_d-0.5,'r')
        plt.xlabel('Number of Samples')
        plt.ylabel('Discharge Pressure (psi)')
        plt.title('Discharge Pressure')
        plt.text(0,138.4,'Pdis =  %s \u00B1  %s' %(round(Pd_avg,2),round(u_Pdis,4)))
        pdf.savefig()
        plt.close()

        plt.figure(3)
        plt.plot(t,Tsuc)
        plt.plot(t,T_s)
        plt.plot(t,T_s+0.2,'r')
        plt.plot(t,T_s-0.2,'r')
        plt.xlabel('Number of Samples')
        plt.ylabel('Suction Temperature (°F)')
        plt.title('Suction Temperature')
        plt.text(0,64.5,'Tsuc =  %s \u00B1  %s' %(round(Ts_avg,2),round(u_Tsuc,4)))
        pdf.savefig()
        plt.close()

        plt.figure(4)
        plt.plot(t,Tsuc)
        plt.plot(t,T_s)
        plt.plot(t,T_s+1.8,'r')
        plt.plot(t,T_s-1.8,'r')
        plt.xlabel('Number of Samples')
        plt.ylabel('Suction Temperature (°F)')
        plt.title('Suction Temperature with ASHRAE Standard')
        pdf.savefig()
        plt.close()

        plt.figure(5)
        plt.plot(t,Tdis)
        plt.title('Discharge Temperature')
        pdf.savefig()
        plt.close()

        plt.figure(6)
        plt.plot(t,DT_sup)
        plt.plot(t,DT_s)
        plt.plot(t,DT_s+0.2,'r')
        plt.plot(t,DT_s-0.2,'r')
        plt.title('Superheat')
        pdf.savefig()
        plt.close()

        plt.figure(7)
        plt.plot(t,DT_sub)
        plt.title('Subcooling')
        pdf.savefig()

        plt.figure(8)
        plt.plot(t,w)
        plt.title('Compressor Speed')
        pdf.savefig()
        plt.close()


