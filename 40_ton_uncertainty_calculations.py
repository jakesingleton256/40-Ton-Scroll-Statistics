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
import statistics as st
from matplotlib.backends.backend_pdf import PdfPages
x,y,w,z=[[],[],[],[]]
y=[]
Pdis_uncert=[]
Psuc_uncert=[]
Tsuc_uncert=[]
Pdis_tot=[]
Psuc_tot=[]
Tsuc_tot=[]
T_cond=[]
BP=[]
WP=[]
Twi=[]
DT_sub=[]
T_evap=[]
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
        df1 = pd.DataFrame(df, columns=[u'Psuc', u'Tsuc', u'Pdis', u'Twi', u'Twe', u'Bypass Position', u'Water Pump Speed', u'DT_sub'])              

        # Create variables for columns
        Psuc = df1[u'Psuc']
        Tsuc = df1[u'Tsuc']
        Pdis = df1[u'Pdis']
        Twi.append(df1[u'Twi'].mean())
        Twe = df1[u'Twe']
        BP.append(df1[u'Bypass Position'].mean())
        WP.append(df1[u'Water Pump Speed'].mean())
        DT_sub.append(df1[u'DT_sub'].mean())

        if '70C' in name:
            T_cond.append(70)
        elif '80C' in name:
            T_cond.append(80)
        elif '90C' in name:
            T_cond.append(90)
        elif '100C' in name:
            T_cond.append(100)
        elif '110C' in name:
            T_cond.append(110)
        elif '120C' in name:
            T_cond.append(120)
        elif '130C' in name:
            T_cond.append(130)

        if '45E' in name:
            T_evap.append(45)
        elif '50E' in name:
            T_evap.append(50)
        elif '40E' in name:
            T_evap.append(40)
        elif '30E' in name:
            T_evap.append(30)
        elif '20E' in name:
            T_evap.append(20)
        elif '10E' in name:
            T_evap.append(10)

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

        # Calculate random uncertainty according to ASHRAE 23.1
        Psuc_tot.append((u_Psuc/Ps_avg)*100)
        Tsuc_tot.append(u_Tsuc)
        Pdis_tot.append((u_Pdis/Pd_avg)*100)
        
        print(Filename[:-4], 'total uncertainty')
        print('Psuc = ',Ps_avg,'\u00B1',u_Psuc)
        print('Tsuc = ',Ts_avg,'\u00B1',u_Tsuc)
        print('Pdis = ',Pd_avg,'\u00B1',u_Pdis)

        data = {'Random':[s_Psuc, s_Tsuc, s_Pdis], 'Total':[u_Psuc, u_Tsuc, u_Pdis]}
        d = pd.DataFrame(data, index = ['Psuc', 'Tsuc', 'Pdis'], columns=['Random', 'Total'])
        d.to_excel(writer, '%s' % os.path.basename(Filename[:-4]))

writer.save()
print("")
print('Average total uncertainty across all measurements')
print('Suction Pressure--', 'Average:', np.mean(Psuc_tot), 'Standard Deviation:', st.stdev(Psuc_tot))
print('Suction Temperature--', 'Average:', np.mean(Tsuc_tot), 'Standard Deviation:', st.stdev(Tsuc_tot))
print('Discharge Pressure--', 'Average:', np.mean(Pdis_tot), 'Standard Deviation:', st.stdev(Pdis_tot))

print("")
print('Average random uncertainty across all measurements')
print('Suction Pressure--', 'Average:', np.mean(Psuc_uncert), 'Standard Deviation:', st.stdev(Psuc_uncert))
print('Suction Temperature--', 'Average:', np.mean(Tsuc_uncert), 'Standard Deviation:', st.stdev(Tsuc_uncert))
print('Discharge Pressure--', 'Average:', np.mean(Pdis_uncert), 'Standard Deviation:', st.stdev(Pdis_uncert))

with PdfPages('Condensing Temp v Pdis.pdf') as pdf:
    plt.figure(1)
    plt.scatter(T_cond, Pdis_uncert)
    z = np.polyfit(T_cond, Pdis_uncert, 1)
    p = np.poly1d(z)
    plt.plot(T_cond, p(T_cond), "r-")
    plt.xlabel('Condensing Temperature [F]')
    plt.ylabel(' Discharge Pressure Random Uncertainty (psi)')
    #plt.title('Condensing Temp v Pdis Uncertainty')
    #plt.text(0, 54.35, 'Psuc =  %s \u00B1  %s' % (round(Ps_avg, 2), round(u_Psuc, 4)))
    pdf.savefig()
    plt.show()
    plt.close()

    plt.figure(2)
    plt.scatter(T_cond, Psuc_uncert)
    z = np.polyfit(T_cond, Psuc_uncert, 1)
    p = np.poly1d(z)
    plt.plot(T_cond, p(T_cond), "r-")
    plt.xlabel('Condensing Temperature [F]')
    plt.ylabel(' Suction Pressure Random Uncertainty (psi)')
    #plt.title('Condensing Temp v Psuc Uncertainty')
    # plt.text(0, 54.35, 'Psuc =  %s \u00B1  %s' % (round(Ps_avg, 2), round(u_Psuc, 4)))
    pdf.savefig()
    plt.close()

    plt.figure(3)
    plt.scatter(T_cond, Tsuc_uncert)
    z = np.polyfit(T_cond, Tsuc_uncert, 1)
    p = np.poly1d(z)
    plt.plot(T_cond, p(T_cond), "r-")
    plt.xlabel('Condensing Temperature [F]')
    plt.ylabel(' Suction Temperature Random Uncertainty (F)')
    #plt.title('Condensing Temp v T_suc Uncertainty')
    # plt.text(0, 54.35, 'Psuc =  %s \u00B1  %s' % (round(Ps_avg, 2), round(u_Psuc, 4)))
    pdf.savefig()
    plt.close()
    
with PdfPages('Evaporating Temp v Pdis.pdf') as pdf:
    plt.figure(1)
    plt.scatter(T_evap, Pdis_uncert)
    z = np.polyfit(T_evap, Pdis_uncert, 1)
    p = np.poly1d(z)
    plt.plot(T_evap, p(T_evap), "r-")
    plt.xlabel('Evaporating Temperature [F]')
    plt.ylabel(' Discharge Pressure Random Uncertainty (psi)')
    #plt.title('Condensing Temp v Pdis Uncertainty')
    #plt.text(0, 54.35, 'Psuc =  %s \u00B1  %s' % (round(Ps_avg, 2), round(u_Psuc, 4)))
    pdf.savefig()
    plt.show()
    plt.close()

    plt.figure(2)
    plt.scatter(T_evap, Psuc_uncert)
    z = np.polyfit(T_evap, Psuc_uncert, 1)
    p = np.poly1d(z)
    plt.plot(T_evap, p(T_evap), "r-")
    plt.xlabel('Evaporating Temperature [F]')
    plt.ylabel(' Suction Pressure Random Uncertainty (psi)')
    #plt.title('Condensing Temp v Psuc Uncertainty')
    # plt.text(0, 54.35, 'Psuc =  %s \u00B1  %s' % (round(Ps_avg, 2), round(u_Psuc, 4)))
    #plt.show()
    pdf.savefig()
    plt.close()

    plt.figure(3)
    plt.scatter(T_evap, Tsuc_uncert)
    z = np.polyfit(T_evap, Tsuc_uncert, 1)
    p = np.poly1d(z)
    plt.plot(T_evap, p(T_evap), "r-")
    plt.xlabel('Evaporating Temperature [F]')
    plt.ylabel(' Suction Temperature Random Uncertainty (F)')
    #plt.title('Condensing Temp v T_suc Uncertainty')
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
    #plt.title('Bypass Valve Position v Pdis Uncertainty')
    #plt.text(0, 54.35, 'Psuc =  %s \u00B1  %s' % (round(Ps_avg, 2), round(u_Psuc, 4)))
    pdf.savefig()
    plt.show()
    plt.close()

    plt.figure(2)
    plt.scatter(BP, Psuc_uncert)
    z = np.polyfit(BP, Psuc_uncert, 1)
    p = np.poly1d(z)
    plt.plot(BP, p(BP), "r-")
    plt.xlabel('Bypass Valve Position (%)')
    plt.ylabel(' Suction Pressure Random Uncertainty (psi)')
    #plt.title('Bypass Valve Position v Psuc Uncertainty')
    # plt.text(0, 54.35, 'Psuc =  %s \u00B1  %s' % (round(Ps_avg, 2), round(u_Psuc, 4)))
    pdf.savefig()
    plt.close()

    plt.figure(3)
    plt.scatter(BP, Tsuc_uncert)
    z = np.polyfit(BP, Tsuc_uncert, 1)
    p = np.poly1d(z)
    plt.plot(BP, p(BP), "r-")
    plt.xlabel('Bypass Valve Position (%)')
    plt.ylabel(' Suction Temperature Random Uncertainty (F)')
    #plt.title('Bypass Valve Position v T_suc Uncertainty')
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
    #plt.title('Water Pump Speed v Pdis Uncertainty')
    # plt.text(0, 54.35, 'Psuc =  %s \u00B1  %s' % (round(Ps_avg, 2), round(u_Psuc, 4)))
    pdf.savefig()
    plt.show()
    plt.close()

    plt.figure(2)
    plt.scatter(WP, Psuc_uncert)
    z = np.polyfit(WP, Psuc_uncert, 1)
    p = np.poly1d(z)
    plt.plot(WP, p(WP), "r-")
    plt.xlabel('Water Pump Speed (Hz)')
    plt.ylabel(' Suction Pressure Random Uncertainty (psi)')
    #plt.title('Water Pump Speed v Psuc Uncertainty')
    # plt.text(0, 54.35, 'Psuc =  %s \u00B1  %s' % (round(Ps_avg, 2), round(u_Psuc, 4)))
    pdf.savefig()
    plt.close()

    plt.figure(3)
    plt.scatter(WP, Tsuc_uncert)
    z = np.polyfit(WP, Tsuc_uncert, 1)
    p = np.poly1d(z)
    plt.plot(WP, p(WP), "r-")
    plt.xlabel('Water Pump Speed (Hz)')
    plt.ylabel(' Suction Temperature Random Uncertainty (F)')
    #plt.title('Water Pump Speed v T_suc Uncertainty')
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
    #plt.title('Inlet Water Temperature v Pdis Uncertainty')
    #plt.text(0, 54.35, 'Psuc =  %s \u00B1  %s' % (round(Ps_avg, 2), round(u_Psuc, 4)))
    pdf.savefig()
    plt.show()
    plt.close()

    plt.figure(2)
    plt.scatter(Twi, Psuc_uncert)
    z = np.polyfit(Twi, Psuc_uncert, 1)
    p = np.poly1d(z)
    plt.plot(Twi, p(Twi), "r-")
    plt.xlabel('Inlet Water Temperature (F)')
    plt.ylabel(' Suction Pressure Random Uncertainty (psi)')
    #plt.title('Inlet Water Temperature v Psuc Uncertainty')
    # plt.text(0, 54.35, 'Psuc =  %s \u00B1  %s' % (round(Ps_avg, 2), round(u_Psuc, 4)))
    pdf.savefig()
    plt.close()

    plt.figure(3)
    plt.scatter(Twi, Tsuc_uncert)
    z = np.polyfit(Twi, Tsuc_uncert, 1)
    p = np.poly1d(z)
    plt.plot(Twi, p(Twi), "r-")
    plt.xlabel('Inlet Water Temperature (F)')
    plt.ylabel(' Suction Temperature Random Uncertainty (F)')
    #plt.title('Inlet Water Temperature v T_suc Uncertainty')
    # plt.text(0, 54.35, 'Psuc =  %s \u00B1  %s' % (round(Ps_avg, 2), round(u_Psuc, 4)))
    pdf.savefig()
    plt.close()

with PdfPages('Subcooling.pdf') as pdf:
    plt.figure(1)
    plt.scatter(DT_sub, Pdis_uncert)
    z = np.polyfit(DT_sub, Pdis_uncert, 1)
    p = np.poly1d(z)
    plt.plot(DT_sub, p(DT_sub), "r-")
    plt.xlabel('Subcooling (F)')
    plt.ylabel(' Discharge Pressure Random Uncertainty (psi)')
    #plt.title('Subcooling v Pdis Uncertainty')
    #plt.text(0, 54.35, 'Psuc =  %s \u00B1  %s' % (round(Ps_avg, 2), round(u_Psuc, 4)))
    pdf.savefig()
    plt.show()
    plt.close()

    plt.figure(2)
    plt.scatter(DT_sub, Psuc_uncert)
    z = np.polyfit(DT_sub, Psuc_uncert, 1)
    p = np.poly1d(z)
    plt.plot(DT_sub, p(DT_sub), "r-")
    plt.xlabel('Subcooling (F)')
    plt.ylabel(' Suction Pressure Random Uncertainty (psi)')
    #plt.title('Subcooling v Psuc Uncertainty')
    # plt.text(0, 54.35, 'Psuc =  %s \u00B1  %s' % (round(Ps_avg, 2), round(u_Psuc, 4)))
    pdf.savefig()
    plt.close()

    plt.figure(3)
    plt.scatter(DT_sub, Tsuc_uncert)
    z = np.polyfit(DT_sub, Tsuc_uncert, 1)
    p = np.poly1d(z)
    plt.plot(DT_sub, p(DT_sub), "r-")
    plt.xlabel('Subcooling (F)')
    plt.ylabel(' Suction Temperature Random Uncertainty (F)')
    #plt.title('Subcooling v T_suc Uncertainty')
    # plt.text(0, 54.35, 'Psuc =  %s \u00B1  %s' % (round(Ps_avg, 2), round(u_Psuc, 4)))
    pdf.savefig()
    plt.show()
    plt.close()


import statsmodels.api as sm
data = {'Discharge Pressure Uncertainty': Pdis_uncert, 'Suction Pressure Uncertainty': Psuc_uncert,
        'Suction Temperature Uncertainty': Tsuc_uncert, 'Condensing Temperature': T_cond,
        'Water Pump Speed': WP, 'Bypass Valve Position': BP, 'Condenser Water Inlet Temperature': Twi, 'Evaporating Temperature': T_evap}
df2 = pd.DataFrame(data, columns=['Discharge Pressure Uncertainty', 'Suction Pressure Uncertainty',
                               'Suction Temperature Uncertainty', 'Condensing Temperature',
                               'Water Pump Speed', 'Bypass Valve Position', 'Condenser Water Inlet Temperature', 'Evaporating Temperature'])
x = df2['Condensing Temperature']
y = df2['Discharge Pressure Uncertainty']
model = sm.OLS(x, y).fit()
predictions = model.predict(x)
print(model.summary())

x1 = df2['Water Pump Speed']
y1 = df2['Discharge Pressure Uncertainty']
model1 = sm.OLS(x1, y1).fit()
predictions1 = model1.predict(x1)
print(model1.summary())

x2 = df2['Bypass Valve Position']
y2 = df2['Discharge Pressure Uncertainty']
model2 = sm.OLS(x2, y2).fit()
predictions2 = model2.predict(x2)
print(model2.summary())

x3 = df2['Condenser Water Inlet Temperature']
y3 = df2['Discharge Pressure Uncertainty']
model3 = sm.OLS(x3, y3).fit()
predictions3 = model3.predict(x3)
print(model3.summary())

x4 = df2['Evaporating Temperature']
y4 = df2['Discharge Pressure Uncertainty']
model4 = sm.OLS(x4, y4).fit()
predictions4 = model4.predict(x4)
print(model4.summary())

x5 = df2['Evaporating Temperature']
y5 = df2['Suction Pressure Uncertainty']
model5 = sm.OLS(x5, y5).fit()
predictions5 = model5.predict(x5)
print(model5.summary())

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
