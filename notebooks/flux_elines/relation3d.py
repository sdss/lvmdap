def my_scatter_3d(ax,x_par,y_par,z_par,c_par,x_min,x_max,y_min,y_max,z_min,z_max,c_min,c_max,x_label,y_label,z_label,order=1,xf_min=0.0,yf_min=0.0,den_par_min=0.95,x_par_cut=-1e12,y_par_cut=-1e12,z_par_cut=-1e12,title="3Dplot",error=0.1,MC=10):
    mask = (x_par>x_min) & (x_par<x_max) & (y_par>y_min) & (y_par<y_max) & (z_par>z_min) & (z_par<z_max) & (c_par>c_min) & (c_par<c_max) & (c_par>0.78) 
#    print (x_par_cut,y_par_cut,z_par_cut)
    mask_cut = mask & (x_par>x_par_cut) & (y_par>y_par_cut) & (z_par>z_par_cut)
    mean = np.array([0.0,0.0,0.0])
    cov = np.array([[1.0,-0.5,0.8], [-0.5,1.1,0.0], [0.8,0.0,1.0]])
    x = x_par[mask]
    y = y_par[mask]
    z = z_par[mask]
    n_obj=len(x)
    data = np.c_[x,y,z]
    print(data.ndim)
    mn = np.min(data, axis=0)
    mx = np.max(data, axis=0)
    (a,b,c,d) = best_fitting_plane(data, equation=True)
#    print('PCA = ',a,b,c,d)
    #    color = c_par[mask]
#    color = 'darksalmon'
#    color = 'gold'
    color = 'goldenrod'

    xyz = np.vstack([x,y,z])
    kde = stats.gaussian_kde(xyz)
    density = np.array(kde(xyz))
    
    
    alpha=100/n_obj
    if (alpha>1.): 
        alpha=1.
    if (alpha<0.05):
        alpha=0.05
        
    #
    # MC 
    #
    a_C=np.zeros((MC,3))
    a_PCA_a=np.zeros(MC)
    a_PCA_b=np.zeros(MC)
    a_PCA_c=np.zeros(MC)
    a_PCA_d=np.zeros(MC)
    for iMC in range(MC):
        e_x=2*error-error*(x_par-x_min)/(x_max-x_min)
        e_y=2*error-error*(y_par-y_min)/(y_max-y_min)    
        e_z=2*error-error*(z_par-z_min)/(z_max-z_min)
        x_now=x_par+np.random.normal(loc=0.0,scale=e_x,size=len(e_x))
        y_now=y_par+np.random.normal(loc=0.0,scale=e_y,size=len(e_y))
        z_now=z_par+np.random.normal(loc=0.0,scale=e_z,size=len(e_z))
        x_cut = x_now[mask_cut]
        y_cut = y_now[mask_cut]
        z_cut = z_now[mask_cut]
        data_cut = np.c_[x_cut,y_cut,z_cut]
        X,Y = np.meshgrid(np.linspace(mn[0], mx[0], 10), np.linspace(mn[1], mx[1], 10))
        XX = X.flatten()
        YY = Y.flatten()            
        if order == 1:
            A = np.c_[data_cut[:,0], data_cut[:,1], np.ones(data_cut.shape[0])]
            C,_,_,_ = scipy.linalg.lstsq(A, data_cut[:,2])    # coefficients
#        Z = C[0]*X + C[1]*Y + C[2]
        elif order == 2:
            A = np.c_[np.ones(data_cut.shape[0]), data_cut[:,:2], np.prod(data_cut[:,:2], axis=1), data_cut[:,:2]**2]
            C,_,_,_ = scipy.linalg.lstsq(A, data_cut[:,2])
#        Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2], C).reshape(X.shape)
        a_C[iMC,:]=C 
        data_now = np.c_[x_now[mask],y_now[mask],z_now[mask]]
        (a,b,c,d) = best_fitting_plane(data_now, equation=True)
#        print('PCA (',iMC,') = ',a,b,c,d)
        a_PCA_a[iMC]=a
        a_PCA_b[iMC]=b
        a_PCA_c[iMC]=c
        a_PCA_d[iMC]=d
    #        print('Coeffs=',C)

    C=np.mean(a_C,axis=0)
    e_C=np.std(a_C,axis=0)
    
    a=np.mean(a_PCA_a)
    e_a=np.std(a_PCA_a)
    b=np.mean(a_PCA_b)
    e_b=np.std(a_PCA_b)
    c=np.mean(a_PCA_c)
    e_c=np.std(a_PCA_c)
    d=np.mean(a_PCA_d)
    e_d=np.std(a_PCA_d)
    

    if order == 1:
        A = np.c_[data_cut[:,0], data_cut[:,1], np.ones(data_cut.shape[0])]
#        C,_,_,_ = scipy.linalg.lstsq(A, data_cut[:,2])    # coefficients
        Z = C[0]*X + C[1]*Y + C[2]
    elif order == 2:
        A = np.c_[np.ones(data_cut.shape[0]), data_cut[:,:2], np.prod(data_cut[:,:2], axis=1), data_cut[:,:2]**2]
#        C,_,_,_ = scipy.linalg.lstsq(A, data_cut[:,2])
        Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2], C).reshape(X.shape)



#
# Fitting
#
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.1, color='black')
#    ax.plot_surface(X, Y, z_min+0.0*Z, rstride=1, cstride=1, alpha=0.1, color='black')
#    ax.plot_surface(x_min+0.0*X, Y, Z, rstride=1, cstride=1, alpha=0.1, color='black')
#    ax.plot_surface(X, y_min+0.0*Y, Z, rstride=1, cstride=1, alpha=0.1, color='black')
    
#
# PCA
#
    Z_new = (a*X+b*Y+d)/(-c)
    X_new = (b*Y+c*Z_new+d)/(-a)
    Y_new = (a*X+c*Z_new+d)/(-b)
    ax.plot_surface(X, Y, Z_new, rstride=1, cstride=1, alpha=0.1, color='blue')
#    ax.plot_surface(X_new, Y, z_min+0.0*Z, rstride=1, cstride=1, alpha=0.1, color='blue')
#    ax.plot_surface(x_min+0.0*X, Y_new, Z_new, rstride=1, cstride=1, alpha=0.1, color='blue')
#    ax.plot_surface(X, y_min+0.0*Y, Z_new, rstride=1, cstride=1, alpha=0.1, color='blue')
    
    #    ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1, alpha=0.15, color='black',linewidth=2)
#    ax.plot_trisurf(X, Y, Z, rstride=1, cstride=1, alpha=0.15)
    cm = 'Wistia'
#    ax.scatter(data[:,0], data[:,1], data[:,2], c=color, vmin=c_min,vmax=c_max, \
#               rasterized=True, cmap = cm, s=50, alpha=alpha)
    c_min=np.nanmin(density)
    c_max=np.nanmax(density)
#    print ("Density min,max = ",c_min,c_max,density)
#    ax.scatter(data[:,0], data[:,1], data[:,2], c=color, vmin=c_min,vmax=c_max, \
#               rasterized=True, cmap = cm, s=50, alpha=alpha)
    c_min=0
    c_max=1
    ax.scatter(data[:,0], data[:,1], data[:,2], c=density, vmin=c_min,vmax=c_max, \
               rasterized=True, cmap = cm, s=50, alpha=alpha*0.5)

    cx = x_min+np.zeros_like(data[:,0])* ax.get_xlim3d()[0]
    cy = y_min+np.zeros_like(data[:,1])* ax.get_ylim3d()[1]
    cz = z_min+np.zeros_like(data[:,2])* ax.get_zlim3d()[0]
    ax.scatter(data[:,0], data[:,1], cz, c=density, vmin=c_min,vmax=c_max,  \
               rasterized=True, cmap = cm, s=10, alpha=alpha,zorder=1)
    ax.scatter(data[:,0], cy, data[:,2], c=density, vmin=c_min,vmax=c_max, \
               rasterized=True, cmap = cm, s=10, alpha=alpha,zorder=1)
    ax.scatter(cx, data[:,1], data[:,2], c=density, vmin=c_min,vmax=c_max, \
               rasterized=True, cmap = cm, s=10, alpha=alpha,zorder=1)

#
# Contour x,y
#
#    c_color="black"
    c_color="black"
    nbins=40

#    x_range = np.linspace(x_min+0.1*(x_max-x_min),x_max-0.1*(x_max-x_min),10)
#    y_range = np.linspace(y_min+0.1*(y_max-y_min),y_max-0.1*(y_max-y_min),10)
#    z_range = np.linspace(z_min+0.1*(z_max-z_min),z_max-0.1*(z_max-z_min),10)
    
    x_range = np.linspace(np.min(x),np.max(x),10)
    y_range = np.linspace(np.min(y),np.max(y),10)
    z_range = np.linspace(np.min(z),np.max(z),10)
    x_mean=np.mean(x)
    y_mean=np.mean(y)
    z_mean=np.mean(z)

# X,Y
    counts, xbins, ybins = np.histogram2d(x, y, bins=nbins, normed=True,range=[[x_min,x_max],[y_min,y_max]])
    counts=ndimage.gaussian_filter(counts, sigma=1, order=0)
    counts /= counts.max()
    sum_total=counts.sum()
    vals=[]
    levels=[]
    for idx,cuts in enumerate(np.arange(0.00,1.0,0.01)):
        mask_now= counts>cuts
        levels.append(cuts)
        vals.append(counts[mask_now].sum()/sum_total)
    vals_cont=np.array([0.95,0.80,0.40])
    levels_cont=np.interp(vals_cont,np.array(levels),np.array(vals))
    counts_rot=np.rot90(counts,3)
    xbins=xbins+0.5*(x_max-x_min)/nbins
    ybins=ybins+0.5*(y_max-y_min)/nbins
    flip_counts_rot=np.fliplr(counts_rot)
    p_cont=ax.contour(xbins[0:nbins],ybins[0:nbins],flip_counts_rot,levels_cont,zdir='z',colors=c_color, offset=z_min)

    x_plot = x_range
    y_plot = y_mean+0.*x_plot
    z_plot =  C[0]*x_plot + C[1]*y_plot + C[2]

    y_plot = y_min+0.*x_plot
    ax.plot(x_plot,y_plot,z_plot,':',color=c_color,linewidth=3,zorder=10)
#    print(x_plot,y_plot,z_plot)

# X,Z
    counts, xbins, ybins = np.histogram2d(x, z, bins=nbins, normed=True,range=[[x_min,x_max],[z_min,z_max]])
#    print(xbins,ybins)
    counts=ndimage.gaussian_filter(counts, sigma=1, order=0)
    counts /= counts.max()
    sum_total=counts.sum()
    vals=[]
    levels=[]
    for idx,cuts in enumerate(np.arange(0.00,1.0,0.01)):
        mask_now= counts>cuts
        levels.append(cuts)
        vals.append(counts[mask_now].sum()/sum_total)
    vals_cont=np.array([0.95,0.80,0.40])
    levels_cont=np.interp(vals_cont,np.array(levels),np.array(vals))
    counts_rot=np.rot90(counts,3)
    xbins=xbins+0.5*(x_max-x_min)/nbins
    ybins=ybins+0.5*(z_max-z_min)/nbins
    flip_counts_rot=np.fliplr(counts_rot)
    Xc, Yc = np.meshgrid(xbins[0:nbins], ybins[0:nbins])
    p_cont=ax.contour(Xc,flip_counts_rot,Yc,levels_cont,zdir='y',colors=c_color,offset=y_min)
# p_cont=ax.contour(xbins[0:nbins],ybins[0:nbins],flip_counts_rot,levels_cont,zdir='x',colors=c_color,offset=y_max)

    x_plot = x_mean+0.*x_range
    y_plot = y_range
    z_plot =  C[0]*x_plot + C[1]*y_plot + C[2]
    x_plot = x_min+0.*x_range
    ax.plot(x_plot,y_plot,z_plot,':',color=c_color,linewidth=3,zorder=10)
#    print(x_plot,y_plot,z_plot)
    
    # Y,Z
    counts, xbins, ybins = np.histogram2d(y, z, bins=nbins, normed=True,range=[[y_min,y_max],[z_min,z_max]])
    counts=ndimage.gaussian_filter(counts, sigma=1, order=0)
    counts /= counts.max()
    sum_total=counts.sum()
    vals=[]
    levels=[]
    for idx,cuts in enumerate(np.arange(0.00,1.0,0.01)):
        mask_now= counts>cuts
        levels.append(cuts)
        vals.append(counts[mask_now].sum()/sum_total)
    vals_cont=np.array([0.95,0.80,0.40])
    levels_cont=np.interp(vals_cont,np.array(levels),np.array(vals))
    counts_rot=np.rot90(counts,3)
    xbins=xbins+0.5*(y_max-y_min)/nbins
    ybins=ybins+0.5*(z_max-z_min)/nbins
    flip_counts_rot=np.fliplr(counts_rot)
#    Xc, Yc = (u.reshape(nbins, nbins) for u in (xbins[0:nbins], ybins[0:nbins]))
#    print(xbins)
    Xc, Yc = np.meshgrid(xbins[0:nbins], ybins[0:nbins])
    p_cont=ax.contour(flip_counts_rot,Xc,Yc,levels_cont,zdir='x',colors=c_color,offset=x_min)

    x_plot = x_range
    z_plot = z_mean+0.*x_plot
    y_plot = (z_plot-C[0]*x_plot-C[2])/(C[1])
#    print(x_plot,)
    z_plot = z_min+0.*x_plot
    mask_plot = (y_plot>np.min(y)) & (y_plot<np.max(y)) & (x_plot>np.min(x)) & (x_plot<np.max(x)) 
    C0=round(C[0],2)
    C1=round(C[1],2)
    PC2=round(10**C[2],2)
    C2=round(C[2],2)
    SFR="SFR"
    gas="gas"
#    label=f'$\Sigma_{\rm SFR}=$$\Sigma_*^{C0}$'
#    label=f'$\Sigma_{{SFR}} = 10^{ {C2} } \Sigma_*^{ {C0} } \Sigma_{{gas}}^{ {C1} }$'
    C0=round(C[0],2)
    C1=round(C[1],2)
    PC2=round(10**C[2],2)
    C2=round(C[2],2)
    C2r=round((-1)*C[2],2)
#    label=f'$\Sigma_{{SFR}} = 10^{ {C2} }\Sigma_*^{ {C0} } \Sigma_{{gas}}^{ {C1} }$'
#    label=f'$\Sigma_{{SFR}} = 10^{ {C2} }\Sigma_*^{ {C0} } \Sigma_{{gas}}^{ {C1} }$'

    label=f'$\Sigma_*^{ {C0} } \Sigma_{{mol}}^{ {C1} } \Sigma_{{SFR}}^{ {-1} } = 10^{ {C2r}}$'
    ax.plot(x_plot[mask_plot],y_plot[mask_plot],z_plot[mask_plot],':',color=c_color,\
            linewidth=3,label=label,zorder=10)

    
    pA=round(a,2)
    pB=round(b,2)
    pC=round(c,2)
    pD=round((-1)*d,2)
    # X,Y
    y_plot = y_range
    z_plot = z_mean+0.*y_plot
    x_plot=(b*y_plot+c*z_plot+d)/(-a)    
    z_plot = z_min+0.*x_plot
    mask_plot = (y_plot>np.min(y)) & (y_plot<np.max(y)) & (x_plot>np.min(x)) & (x_plot<np.max(x)) 
    label=f'$\Sigma_*^{ {pA} } \Sigma_{{mol}}^{ {pB} } \Sigma_{{SFR}}^{ {pC} } = 10^{ {pD}}$'
    ax.plot(x_plot[mask_plot],y_plot[mask_plot],z_plot[mask_plot],'--',color='blue',\
            linewidth=3,label=label,zorder=10)   
    # X,Z
    x_plot = x_range
    y_plot = y_mean+0.*x_plot
    z_plot=(b*y_plot+a*x_plot+d)/(-c)    
    y_plot = y_min+0.*x_plot
    ax.plot(x_plot,y_plot,z_plot,'--',color='blue',linewidth=3,zorder=10)    
    # Y,Z

    y_plot = y_range
    x_plot = x_mean+0.*y_plot
    z_plot=(b*y_plot+a*x_plot+d)/(-c)    
    x_plot = x_min+0.*x_plot
    ax.plot(x_plot,y_plot,z_plot,'--',color='blue',linewidth=3,zorder=10)    
    
#    ax.plot(x_plot,y_plot,z_plot,'--',color='grey',linewidth=3)
#    print(x_plot,y_plot,z_plot)
    
#    cset = ax.contour(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)

#
# PCA plot
#
#    Z_new = (a*X+b*Y+d)/(-c)
#    X_new = (b*Y+c*Z_new+d)/(-a)
#    Y_new = (a*X+c*Z_new+d)/(-b)
    
          
    ax.set_xlabel(x_label,fontsize=23, labelpad=10)
    ax.set_ylabel(y_label,fontsize=23, labelpad=10)
    ax.set_zlabel(z_label,fontsize=23, labelpad=10)
    ax.set_xlim([x_min,x_max])
    ax.set_ylim([y_max,y_min])       
    ax.set_zlim([z_min,z_max])     
#    print(x_min,x_max)
#    print(y_min,y_max)
#    print(z_min,z_max)
#    ax.axis('auto')
#    ax.axis('tight')
    handles, labels = ax.get_legend_handles_labels()
#    ax.legend(handles[::-1], labels[::-1],loc=(0.05,0.8),frameon=True,handlelength=1.5)#loc="lower right", bbox_to_anchor=(0.6,0.5))    

    z_mod=C[0]*x_cut+C[1]*y_cut+C[2]
    rc=np.corrcoef(z_mod, z_cut)
    delta_z=z_cut-z_mod
    s_z = biweight_midvariance(z_cut)
    s_dz = biweight_midvariance(delta_z)
    
    print(title, ' & ',round(C[0],3),' +- ',round(e_C[0],3),' & ',\
    round(C[1],3),' +- ',round(e_C[1],3),' & ',\
    round(C[2],3),' +- ',round(e_C[2],3),' & ',\
    round(rc[0,1],3),' & ',round(s_z,3),' & ',round(s_dz,3),' \\\\')

    delta_mod=(a*x_cut+b*y_cut+c*z_cut+d)
    z_mod=(a*x_cut+b*y_cut+d)/(-c)
    rc=np.corrcoef(z_mod, z_cut)
    delta_z=z_cut-z_mod
    s_z = biweight_midvariance(z_cut)
    s_dz = biweight_midvariance(delta_z)

    print(title, ' PCA: & ',round(a,3),' +- ',round(e_a,3),' & ',\
    round(b,3),' +- ',round(e_b,3),' & ',\
    round(c,3),' +- ',round(e_c,3),' & ',\
    round(d,3),' +- ',round(e_d,3),' & ',\
    round(rc[0,1],3),' & ',round(s_z,3),' & ',round(s_dz,3),' \\\\')
    ax.text2D(0.8, 0.90, title, transform=ax.transAxes, fontsize=24, verticalalignment='top')

    
    
#      round(np.sqrt(np.diag(ea1))[0],3),round(pa1[1],3),round(np.sqrt(np.diag(ea1))[1],3),round(rc[0,1],3),round(s_y_par,3),round(s_dy_par,3),n_obj,n_sf)   
#    print ('3D Coeffs (',title,') = ',C,', error=',e_C,'; rc=',rc,'; s_z=',s_z,'; s_dz=',s_dz)

#
# 3D plot
#
#print ("CALIFA")
#%matplotlib notebook
%matplotlib inline
fig = plt.figure(figsize=(10,10))
#fig, ax = plt.subplots(1,figsize=(10,10))
#ax = fig.gca(projection='3d')

print("Sample &  C0 & C1 & C2 & rc & std_in & std_out \\\\")

SMass_st_min=-1.5
SMass_st_max=4.5
SMass_gas_min=-3
SMass_gas_max=3.25
SSFR_st_min=-13.5 # -13.5
SSFR_st_max=-5.5

#
# Labels
#
EW_label=r'log$|$EW$_{\rm H\alpha}|$ ${\rm \AA}$'
S_Mst_label=r'$\Sigma_*$ log(M$_\odot$/pc$^2$)'
S_Mgas_label=r'$\Sigma_{\rm gas}$ log(M$_\odot$/pc$^2$)'
S_SFR_label=r'$\Sigma_{\rm SFR}$ log(M$_\odot$/pc$^2$/yr)'
dS_Mst_label=r'$\Delta\Sigma_*$ log(M$_\odot$/pc$^2$)'
dS_Mgas_label=r'$\Delta\Sigma_{\rm gas}$ log(M$_\odot$/pc$^2$)'
dS_SFR_label=r'$\Delta\Sigma_{\rm SFR}$ log(M$_\odot$/pc$^2$/yr)'


ax= fig.add_subplot(111, projection= '3d')
my_scatter_3d(ax,log_Sigma_Mst_CAL,log_Sigma_Mgas_CAL,log_Sigma_SFR_CAL,log_EW_Ha_CAL,SMass_st_min,SMass_st_max,SMass_gas_min,SMass_gas_max,SSFR_st_min,SSFR_st_max,0.78-2,0.78+3,S_Mst_label,S_Mgas_label,S_SFR_label,title="CALIFA")


def get_den_point(x_plt,y_plt, x_min,x_max,y_min,y_max,nbins=30):
    print(x_plt)
    mask = (x_plt>x_min) & (x_plt<x_max) & (y_plt>y_min) & (y_plt<y_max)  
    n_sf=len(x_plt[mask])
    counts, xbins, ybins = np.histogram2d(x_plt[mask], y_plt[mask], bins=nbins,
        normed=True,
        range=[[np.nanmin(x_plt[mask]),np.nanmax(x_plt[mask])],[np.nanmin(y_plt[mask]),np.nanmax(y_plt[mask])]])
    counts=ndimage.gaussian_filter(counts, sigma=1, order=0)
    counts /= counts.max()
    sum_total=counts.sum()
    vals_new=[]
    levels_new=[]
    for idx,cuts in enumerate(np.arange(0.00,1.0,0.01)):
        mask_now= counts>cuts
        levels_new.append(cuts)
        vals_new.append(counts[mask_now].sum()/sum_total)
    den_par=np.ones(len(x_plt))
    for i in range(len(x_plt)):
        print('DEN = ',i,x_plot[i],y_plot[i])
        print('DEN = ',np.isfinite(x_plot[i]),np.isfinite(y_plot[i]))
        if ((x_plt[i]>x_min) & (x_plt[i]<x_max) & (y_plt[i]>y_min) & (y_plt[i]<y_max)):  
            if ((np.isfinite(x_plt[i])) and (np.isfinite(y_plt[i]))):
                i_x=np.argmin(np.abs(xbins-x_plt[i]))
                i_y=np.argmin(np.abs(ybins-y_plt[i]))
                print (i_x,i_y)
                if ((i_x>0) and (i_x<nbins) and (i_y>0) and (i_y<nbins)):
                    den_par[i]=np.interp(counts[i_x,i_y],np.array(levels_new),np.array(vals_new))
    return den_par
                
                
def add_relation_3d(ax,x_par,y_par,z_par,c_par,a_xy,b_xy,a_xz,b_xz,a_yz,b_yz,x_min,x_max,y_min,y_max,z_min,z_max,label='test'):
    x_range = np.linspace(x_min+0.15*(x_max-x_min),x_max-0.15*(x_max-x_min),10)
    y_range = np.linspace(y_min+0.15*(y_max-y_min),y_max-0.15*(y_max-y_min),10)
    z_range = np.linspace(z_min+0.15*(z_max-z_min),z_max-0.15*(z_max-z_min),10)
    #
    # xy
    #
    mask = (x_par>x_min) & (x_par<x_max) & (y_par>y_min) & (y_par<y_max) & \
    (z_par>z_min) & (z_par<z_max) & (c_par>0.78-2) & (c_par<0.78+3) & (c_par>0.78) 
    x = x_par[mask]
    y = y_par[mask]
    z = z_par[mask]
    
    x_plot = x_range
    y_plot = a_xy + b_xy * x_range
    z_plot = z_min + 0*x_range
    ax.plot(x_plot,y_plot,z_plot,'-',color='darkred',linewidth=3)#,label=label)     
    y_mod = a_xy + b_xy * x
#    den_par = get_den_point(x,y,x_min,x_max,y_min,y_max)
    
    delta_y = y - y_mod
    rc=np.corrcoef(y_mod, y)    
    s_dz = biweight_midvariance(delta_y)    
    print('rMS: r_cor=',rc[0,1], '; std_out = ',s_dz)      
    
    #
    # xz
    #
    x_plot = x_range
    z_plot = a_xz + b_xz * x_range
    y_plot = y_min + 0*x_range
    ax.plot(x_plot,y_plot,z_plot,'-',color='darkred',linewidth=3)     

    z_mod = a_xz + b_xz * x
    delta_z = z - z_mod
    rc=np.corrcoef(z_mod, z)    
    s_dz = biweight_midvariance(delta_z)
    print('rSFMS: r_cor=',rc[0,1], '; std_out = ',s_dz)    
    
    #
    # yz
    #
    y_plot = y_range
    z_plot = a_yz + b_yz * y_range
    x_plot = x_min + 0*y_range
    ax.plot(x_plot,y_plot,z_plot,'-',color='darkred',linewidth=3)   
    z_mod = a_yz + b_yz * y
    delta_z = z - z_mod
    rc=np.corrcoef(z_mod, z)    
    s_dz = biweight_midvariance(delta_z)
    print('rSK: r_cor=',rc[0,1], '; std_out = ',s_dz)    
    
    #
    # xyz
    #
    z_plot = z_range
    x_plot = (z_plot - a_xz)/b_xz
    y_plot = (z_plot - a_yz)/b_yz
#    ax.plot(x_plot,y_plot,z_plot,'-',color='darkred',linewidth=3)     

    #
    # 3D
    #
    x_plot = x_range
    y_plot = a_xy + b_xy * x_range
    z_plot1 = a_xz + b_xz * x_plot
    z_plot2 = a_yz + b_yz * y_plot
    z_plot = 0.5*(z_plot1+z_plot2)
    ax.plot(x_plot,y_plot,z_plot,'-.',color='darkred',linewidth=3,label=label)         

    #
    # Projection of the 3D line
    #
    y_plot = y_range
    z_plot = z_min + 0*x_range
    ax.plot(x_plot,y_plot,z_plot,'-.',color='darkred',linewidth=3)     

#    0.78-2,0.78+3

    
#    x_plot = x
#    y_plot = a_xy + b_xy * x_plot
    z_plot1 = a_xz + b_xz * x
    z_plot2 = a_yz + b_yz * y
    z_plot = 0.5*(z_plot1+z_plot2)
    zp = 0.5*(a_xz + a_yz)
    a = 0.5*b_xz
    b = 0.5*b_yz
    delta_z = z - z_plot
    rc=np.corrcoef(z_plot, z)
    
#    x_mean = 0.5*()
#    rcNOW=np.corrcoef(z_plot, z)
    
#    delta_z=z_cut-z_mod
    s_z = biweight_midvariance(z)
    s_dz = biweight_midvariance(delta_z)
    print('3D_LINE a_*=',a,'b_gas=',b,' ZP=', zp,' r_cor=',rc[0,1], ' std_in = ',s_z,'; std_out = ',s_dz)
    
    
#### rSFMS
#Mean Coeff.:  -10.111 0.307 0.949 0.203 0.844 0.596 0.32 941 533
#Mean Coeff.:  -0.055 0.257 0.008 0.186 0.009 0.329 0.329 941 533
#### rSK
#Mean Coeff.:  -8.974 0.124 0.897 0.162 0.775 0.605 0.382 903 533
#Mean Coeff.:  -0.034 0.136 -0.005 0.174 -0.014 0.382 0.382 903 533
#### rMgM*
#Mean Coeff.:  -0.919 0.207 0.818 0.15 0.727 0.586 0.404 941 533
#Mean Coeff.:  -0.018 0.204 -0.004 0.152 -0.04 0.386 0.386 941 533

label=f'rSFMS, rSK, rMGMS'
#
# To add a 3d line corresponding to a previous relation
#
add_relation_3d(ax,log_Sigma_Mst_CAL,log_Sigma_Mgas_CAL,log_Sigma_SFR_CAL,log_EW_Ha_CAL,\
                pa1_MGMS_CAL[0],pa1_MGMS_CAL[1],pa1_SFMS_CAL[0],pa1_SFMS_CAL[1],pa1_SK_CAL[0],\
                pa1_SK_CAL[1],SMass_st_min,SMass_st_max,SMass_gas_min,SMass_gas_max,SSFR_st_min,\
                SSFR_st_max,label=label)    
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels,loc=(0.05,0.75),frameon=True,handlelength=1.5)


def rotate(angle):
    ax.view_init(azim=angle)

rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0,362,2),interval=100)
rot_animation.save('rot_CALIFA.gif', dpi=80, writer='imagemagick')

#print(log_Sigma_Mst_CAL)
#print('3D Coeffs = ',C)
#fig,ax = plt.figure(figsize=(10,10))

#%matplotlib inline
#fig.savefig("rp_CAL_3D.png", transparent=False, facecolor='white', edgecolor='white')#.pdf")
#plt.show()
