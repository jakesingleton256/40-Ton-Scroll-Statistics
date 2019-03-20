import pandas as pd
from pandas import ExcelWriter
import numpy as np
import matplotlib.pyplot as plt
import glob2
import os
import xlrd
from os.path import join
from glob2 import glob
import datetime as dt
from matplotlib.backends.backend_pdf import PdfPages
x,y,w,z=[[],[],[],[]]
y=[]
Pdis_uncert=[]
Psuc_uncert=[]
Tsuc_uncert=[]
T_cond=[]
BP=[]
WP=[]
Twi=[]
writer = ExcelWriter('total uncertainty.xlsx')
for name in glob2.glob('**/*'):
    if name.endswith((".csv")):
        #print(name)
        ##if "comp" in name:
        file1, ext=os.path.splitext(name)
        Filename=str(file1)+'.csv'
        df=pd.read_csv(Filename, skiprows = [0,1,2,3,4,5,6,7,8,9,11])
        #columns=df.dtypes.index
        #x.append(df.index)
        #y.append(np.array(df[columns[3]]))
        df1 = pd.DataFrame(df, columns=[u'Psuc', u'Tsuc', u'Pdis', u'Twi', u'Twe', u'Bypass Position', u'Water Pump Speed'])              

        # Create variables for columns
        Psuc = df1[u'Psuc']
        Tsuc = df1[u'Tsuc']
        Pdis = df1[u'Pdis']
        Twi.append(df1[u'Twi'].mean())
        Twe = df1[u'Twe']
        BP.append(df1[u'Bypass Position'].mean())
        WP.append(df1[u'Water Pump Speed'].mean())

        if '70' in name:
            T_cond.append(70)
        elif '80' in name:
            T_cond.append(80)
        elif '90' in name:
            T_cond.append(90)
        elif '100' in name:
            T_cond.append(100)
        elif '110' in name:
            T_cond.append(110)
        elif '120' in name:
            T_cond.append(120)
        elif '130' in name:
            T_cond.append(130)

        # Calculate averages for key variables
        Ps_avg = df1[u'Psuc'].mean()
        Ts_avg = df1[u'Tsuc'].mean()
        Pd_avg = df1[u'Pdis'].mean()
        N = len(df1[u'Psuc'])
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
##        print("s_x_Psuc=",s_x_Psuc)
##        print("s_x_Tsuc=",s_x_Tsuc)
##        print("s_x_Pdis=",s_x_Pdis)

        # Calculate random uncertainty
        s_Psuc=s_x_Psuc/np.sqrt(N)
        s_Tsuc=s_x_Tsuc/np.sqrt(N)
        s_Pdis=s_x_Pdis/np.sqrt(N)
        Psuc_uncert.append(s_Psuc)
        Tsuc_uncert.append(s_Tsuc)
        Pdis_uncert.append(s_Pdis)
##        print("s_Psuc=",s_Psuc)
##        print("s_Tsuc=",s_Tsuc)
##        print("s_Pdis=",s_Pdis)

        # Systematic uncertainty
        sigma_sys_P=0.375
        sigma_sys_T=0.2

        # total uncertainty
        u_Psuc=np.sqrt((s_Psuc**2)+(sigma_sys_P**2))
        u_Tsuc=np.sqrt((s_Tsuc**2)+(sigma_sys_T**2))
        u_Pdis=np.sqrt((s_Pdis**2)+(sigma_sys_P**2))
        
        print(Filename[:-4], 'total uncertainty')
        print('Psuc = ',Ps_avg,'\u00B1',u_Psuc)
        print('Tsuc = ',Ts_avg,'\u00B1',u_Tsuc)
        print('Pdis = ',Pd_avg,'\u00B1',u_Pdis)

        data = {'Random':[s_Psuc, s_Tsuc, s_Pdis], 'Total':[u_Psuc, u_Tsuc, u_Pdis]}
        d = pd.DataFrame(data, index = ['Psuc', 'Tsuc', 'Pdis'], columns=['Random', 'Total'])
        d.to_excel(writer, '%s' % os.path.basename(Filename[:-4]))

writer.save()

with PdfPages('Condensing Temp v Pdis.pdf') as pdf:
    plt.figure(1)
    plt.scatter(T_cond, Pdis_uncert)
    z = np.polyfit(T_cond, Pdis_uncert, 1)
    p = np.poly1d(z)
    plt.plot(T_cond, p(T_cond), "r-")
    plt.xlabel('Condensing Temperature [F]')
    plt.ylabel(' Discharge Pressure Random Uncertainty (psi)')
    plt.title('Condensing Temp v Pdis Uncertainty')
    #plt.text(0, 54.35, 'Psuc =  %s \u00B1  %s' % (round(Ps_avg, 2), round(u_Psuc, 4)))
    pdf.savefig()
    plt.close()

    plt.figure(2)
    plt.scatter(T_cond, Psuc_uncert)
    z = np.polyfit(T_cond, Psuc_uncert, 1)
    p = np.poly1d(z)
    plt.plot(T_cond, p(T_cond), "r-")
    plt.xlabel('Condensing Temperature [F]')
    plt.ylabel(' Suction Pressure Random Uncertainty (psi)')
    plt.title('Condensing Temp v Psuc Uncertainty')
    # plt.text(0, 54.35, 'Psuc =  %s \u00B1  %s' % (round(Ps_avg, 2), round(u_Psuc, 4)))
    pdf.savefig()
    plt.close()

    plt.figure(3)
    plt.scatter(T_cond, Tsuc_uncert)
    z = np.polyfit(T_cond, Tsuc_uncert, 1)
    p = np.poly1d(z)
    plt.plot(T_cond, p(T_cond), "r-")
    plt.xlabel('Condensing Temperature [F]')
    plt.ylabel(' Suction Temperature Random Uncertainty (psi)')
    plt.title('Condensing Temp v T_suc Uncertainty')
    # plt.text(0, 54.35, 'Psuc =  %s \u00B1  %s' % (round(Ps_avg, 2), round(u_Psuc, 4)))
    pdf.savefig()
    plt.close()

with PdfPages('Bypass Valve Position.pdf') as pdf:
    plt.figure(1)
    plt.scatter(BP, Pdis_uncert)
    z = np.polyfit(BP, Pdis_uncert, 1)
    p = np.poly1d(z)
    plt.plot(BP, p(BP), "r-")
    plt.xlabel('Bypass Valve Position (%)')
    plt.ylabel(' Discharge Pressure Random Uncertainty (psi)')
    plt.title('Bypass Valve Position v Pdis Uncertainty')
    #plt.text(0, 54.35, 'Psuc =  %s \u00B1  %s' % (round(Ps_avg, 2), round(u_Psuc, 4)))
    pdf.savefig()
    plt.close()

    plt.figure(2)
    plt.scatter(BP, Psuc_uncert)
    z = np.polyfit(BP, Psuc_uncert, 1)
    p = np.poly1d(z)
    plt.plot(BP, p(BP), "r-")
    plt.xlabel('Bypass Valve Position (%)')
    plt.ylabel(' Suction Pressure Random Uncertainty (psi)')
    plt.title('Bypass Valve Position v Psuc Uncertainty')
    # plt.text(0, 54.35, 'Psuc =  %s \u00B1  %s' % (round(Ps_avg, 2), round(u_Psuc, 4)))
    pdf.savefig()
    plt.close()

    plt.figure(3)
    plt.scatter(BP, Tsuc_uncert)
    z = np.polyfit(BP, Tsuc_uncert, 1)
    p = np.poly1d(z)
    plt.plot(BP, p(BP), "r-")
    plt.xlabel('Bypass Valve Position (%)')
    plt.ylabel(' Suction Temperature Random Uncertainty (psi)')
    plt.title('Bypass Valve Position v T_suc Uncertainty')
    # plt.text(0, 54.35, 'Psuc =  %s \u00B1  %s' % (round(Ps_avg, 2), round(u_Psuc, 4)))
    pdf.savefig()
    plt.close()

with PdfPages('Water Pump Speed.pdf') as pdf:
    plt.figure(1)
    plt.scatter(WP, Pdis_uncert)
    z = np.polyfit(WP, Pdis_uncert, 1)
    p = np.poly1d(z)
    plt.plot(WP, p(WP), "r-")
    plt.xlabel('Water Pump Speed (Hz)')
    plt.ylabel(' Discharge Pressure Random Uncertainty (psi)')
    plt.title('Water Pump Speed v Pdis Uncertainty')
    # plt.text(0, 54.35, 'Psuc =  %s \u00B1  %s' % (round(Ps_avg, 2), round(u_Psuc, 4)))
    pdf.savefig()
    plt.close()

    plt.figure(2)
    plt.scatter(WP, Psuc_uncert)
    z = np.polyfit(WP, Psuc_uncert, 1)
    p = np.poly1d(z)
    plt.plot(WP, p(WP), "r-")
    plt.xlabel('Water Pump Speed (Hz)')
    plt.ylabel(' Suction Pressure Random Uncertainty (psi)')
    plt.title('Water Pump Speed v Psuc Uncertainty')
    # plt.text(0, 54.35, 'Psuc =  %s \u00B1  %s' % (round(Ps_avg, 2), round(u_Psuc, 4)))
    pdf.savefig()
    plt.close()

    plt.figure(3)
    plt.scatter(WP, Tsuc_uncert)
    z = np.polyfit(WP, Tsuc_uncert, 1)
    p = np.poly1d(z)
    plt.plot(WP, p(WP), "r-")
    plt.xlabel('Water Pump Speed (Hz)')
    plt.ylabel(' Suction Temperature Random Uncertainty (psi)')
    plt.title('Water Pump Speed v T_suc Uncertainty')
    # plt.text(0, 54.35, 'Psuc =  %s \u00B1  %s' % (round(Ps_avg, 2), round(u_Psuc, 4)))
    pdf.savefig()
    plt.close()

with PdfPages('Inlet Water Temperature.pdf') as pdf:
    plt.figure(1)
    plt.scatter(Twi, Pdis_uncert)
    z = np.polyfit(Twi, Pdis_uncert, 1)
    p = np.poly1d(z)
    plt.plot(Twi, p(Twi), "r-")
    plt.xlabel('Inlet Water Temperature (F)')
    plt.ylabel(' Discharge Pressure Random Uncertainty (psi)')
    plt.title('Inlet Water Temperature v Pdis Uncertainty')
    #plt.text(0, 54.35, 'Psuc =  %s \u00B1  %s' % (round(Ps_avg, 2), round(u_Psuc, 4)))
    pdf.savefig()
    plt.close()

    plt.figure(2)
    plt.scatter(Twi, Psuc_uncert)
    z = np.polyfit(Twi, Psuc_uncert, 1)
    p = np.poly1d(z)
    plt.plot(Twi, p(Twi), "r-")
    plt.xlabel('Inlet Water Temperature (F)')
    plt.ylabel(' Suction Pressure Random Uncertainty (psi)')
    plt.title('Inlet Water Temperature v Psuc Uncertainty')
    # plt.text(0, 54.35, 'Psuc =  %s \u00B1  %s' % (round(Ps_avg, 2), round(u_Psuc, 4)))
    pdf.savefig()
    plt.close()

    plt.figure(3)
    plt.scatter(Twi, Tsuc_uncert)
    z = np.polyfit(Twi, Tsuc_uncert, 1)
    p = np.poly1d(z)
    plt.plot(Twi, p(Twi), "r-")
    plt.xlabel('Inlet Water Temperature (F)')
    plt.ylabel(' Suction Temperature Random Uncertainty (psi)')
    plt.title('Inlet Water Temperature v T_suc Uncertainty')
    # plt.text(0, 54.35, 'Psuc =  %s \u00B1  %s' % (round(Ps_avg, 2), round(u_Psuc, 4)))
    pdf.savefig()
    plt.close()
####            print(len(x[0]))
##        elif "Aplus" in name:
##            file2, ext2=os.path.splitext(name)
##            Filename1=str(file2)+'.xlsx'
##            df1=pd.read_excel(Filename1, skiprows=[0], index_col=0)
##            columns=df1.dtypes.index
##            w.append(df1.index)
##            z.append(np.array(df1[columns[0]]))
##        #columns=df.dtypes.index
##    #p=df.iloc[:,3]
##    #print(p)
##    #x=df.index
##        if (x!=[] and w!=[] and z!=[]):
##            plt.plot(x[0],y[0],'-', linewidth=2.0, color='r', label='Power-Hobo')
##            plt.plot(w[0],z[0],'--', linewidth=1.0, color='k', label='Power-Aplus')
##            plt.tick_params(axis='both', which='major', labelsize=6)
##            plt.tick_params(axis='both', which='minor', labelsize=6)
##            x,y,w,z=[[],[],[],[]]
##            plt.legend(bbox_to_anchor=(0, 1), loc=2, borderaxespad=0.)
##            plt.ylabel('Power [W]',fontsize=13)
##            plt.xlabel('Time [Hrs]',fontsize=13)
##            plt.title(Filename[:-4])
##            plt.savefig(str(Filename[:-4])+'.pdf')
##            plt.clf()
