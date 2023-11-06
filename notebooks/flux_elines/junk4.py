    if (tab_now['F_6563_cen'] > (n_sig*tab_now['e_F_6563_cen'])):
        if ((tab_now['F_4861_cen'] > (tab_now['e_F_4861_cen'])) & (tab_now['F_6583_cen'] >0)&(tab_now['F_5007_cen'] >0)):
            if (np.abs(tab_now['EW_Ha_cen'])<3):
                ic_cen_now=3 # Post-AGB
            else:
                cut_Kew=0.61/(tab_now['N2_cen']-0.47)+1.19
                if ((tab_now['N2_cen']<0.3) & (tab_now['O3_cen']<cut_Kew)):
                    if (np.abs(tab_now['EW_Ha_cen'])>6):
                        ic_cen_now=2 # SF
                    else:
                        ic_cen_now=1 # Unkwon
                else:
                    if  (np.abs(tab_now['EW_Ha_cen'])<6):
                        ic_cen_now=4
                    else:
                        ic_cen_now=5
        else:
            ic_cen_now=1
    else:
#        prcen('NG')
        ic_cen_now=0
    ion_class_cen.append(ic_cen_now)
    
