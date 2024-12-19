#!/usr/bin/env python3

import sys, os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1



import time
from astropy.io.fits.column import _parse_tdim
import numpy as np
import argparse
from copy import deepcopy as copy
from pprint import pprint

# pyFIT3D dependencies
from pyFIT3D.common.io import clean_preview_results_files, print_time, read_spectra

# 18.11.2023
# So far we were ysing the auto_ssp_tools from pyFIT3D
# We will attempt to modify them
#from pyFIT3D.common.auto_ssp_tools import auto_ssp_elines_single_main
from lvmdap.modelling.auto_rsp_tools import auto_rsp_elines_single_main

from pyFIT3D.common.auto_ssp_tools import load_rss, dump_rss_output
from pyFIT3D.common.io import clean_preview_results_files, print_time, read_spectra

from pyFIT3D.common.gas_tools import detect_create_ConfigEmissionModel
from pyFIT3D.common.io import create_ConfigAutoSSP_from_lists
from pyFIT3D.common.io import create_emission_lines_file_from_list
from pyFIT3D.common.io import create_emission_lines_mask_file_from_list
#from pyFIT3D.common.tools import read_coeffs_CS

from lvmdap.modelling.synthesis import StellarSynthesis
from lvmdap.modelling.auto_rsp_tools import ConfigAutoSSP

from lvmdap.modelling.auto_rsp_tools import model_rsp_elines_single_main


from lvmdap.dap_tools import load_LVM_rss, read_PT, rsp_print_header, plot_spec, read_rsp
from lvmdap.dap_tools import plot_spectra, read_coeffs_RSP, read_elines_RSP, read_tab_EL
from lvmdap.dap_tools import find_redshift_spec
from lvmdap.flux_elines_tools import flux_elines_RSS_EW

from scipy.ndimage import gaussian_filter1d,median_filter

from astropy.table import Table
from astropy.table import join as tab_join
from astropy.table import vstack as vstack_table
from astropy.io import fits, ascii

import yaml
import re
from collections import Counter

from lvmdap.dap_tools import list_columns,read_DAP_file,map_plot_DAP
from lvmdap.dap_tools import fit_legendre_polynomial
from lvmdap.dap_tools import find_redshift_spec, replace_nan_inf_with_adjacent_avg

#
# Just for tests
#
# import matplotlib.pyplot as plt


CWD = os.path.abspath(".")
EXT_CHOICES = ["CCM", "CAL"]
EXT_CURVE = EXT_CHOICES[0]
EXT_RV = 3.1
N_MC = 20


def eline(w,W,F,D,V,dap_fwhm=2.354):
  c = 299792.00
  w0 = W*(1+(V/c))
  sigma=D/dap_fwhm
  e1=np.exp(-0.5*((w-w0)/sigma)**2)
  return F*e1/(sigma*((2*3.1416)**0.5))     


def _no_traceback(type, value, traceback):
  print(value)



#######################################################
# RSP version of the auto_ssp_elines_rnd from pyFIT3D
#######################################################
def auto_rsp_elines_rnd(
    wl__w, f__w, ef__w, ssp_file, spaxel_id, config_file=None, plot=None,
    ssp_nl_fit_file=None, sigma_inst=None, mask_list=None,
    min=None, max=None, w_min=None, w_max=None,
    nl_w_min=None, nl_w_max=None, elines_mask_file=None, fit_gas=True, refine_gas=True,
    input_redshift=None, delta_redshift=None, min_redshift=None, max_redshift=None,
    input_sigma=None, delta_sigma=None, min_sigma=None, max_sigma=None, sigma_gas=None,
    input_AV=None, delta_AV=None, min_AV=None, max_AV=None, ratio=True, y_ratio=None,
    fit_sigma_rnd=True, out_path=None, SPS_master=None, SN_CUT=2):

  #
  # If there is no RSP for the Non-Linear (nl) fitting, they it is 
  # used the one for the Linear Fitting (that it is slower)
  #
    ssp_nl_fit_file = ssp_file if ssp_nl_fit_file is None else ssp_nl_fit_file
    if delta_redshift == 0:
      cc_redshift_boundaries = None
    else:
      cc_redshift_boundaries = [min_redshift, max_redshift]

  #
  # If the emission lines are fitted, but there is no config file, then
  # the program creates a set of configuraton files for the detected emission lines
  # NOTE: I think this is overdoing having the flux_elines script
  # But needs to be explored!
  #
    if fit_gas and config_file is None:
        print("##############################");
        print("# START: Autodectecting emission lines...");
        if sigma_gas is None: sigma_gas = 3.0
        if out_path is None: out_path = "."
        wl_mask = (w_min<=wl__w)&(wl__w<=w_max)
        config_filenames, wl_chunks, _, wave_peaks_tot_rf = detect_create_ConfigEmissionModel(
            wl__w[wl_mask], f__w[wl_mask],
            redshift=input_redshift,
            sigma_guess=sigma_gas,
            chunks=4,
            polynomial_order=1,
            polynomial_coeff_guess=[0.000, 0.001],
            polynomial_coeff_boundaries=[[-1e13, 1e13], [-1e13, 1e13]],
            flux_boundaries_fact=[0.001, 1000],
            sigma_boundaries_fact=[0.1, 1.5],
            v0_boundaries_add=[-1000, 1000],
            peak_find_nsearch=1,
            peak_find_threshold=0.2,
            peak_find_dmin=1,
            crossmatch_list_filename=elines_mask_file,
            crossmatch_absdmax_AA=5,
            crossmatch_redshift_search_boundaries=cc_redshift_boundaries,
            sort_by_flux=True,
            output_path=out_path,
            label=spaxel_id,
            verbose=0,
            plot=0,
        )

        create_emission_lines_mask_file_from_list(wave_peaks_tot_rf, eline_half_range=3*sigma_gas, output_path=out_path, label=spaxel_id)
        create_emission_lines_file_from_list(wave_peaks_tot_rf, output_path=out_path, label=spaxel_id)
        create_ConfigAutoSSP_from_lists(wl_chunks, config_filenames, output_path=out_path, label=spaxel_id)

        config_file = os.path.join(out_path, f"{spaxel_id}.autodetect.auto_ssp_several.config")
        if not refine_gas: elines_mask_file = os.path.join(out_path, f"{spaxel_id}.autodetect.emission_lines.txt")
        print("# END: Autodectecting emission lines...");     
    else:
      print("# Using predefined configuration file for the emission lines");
    #
    # The spectrum is fitted for the 1st time in here
    #
    print("##############################");
    print(f"# START: fitting the continuum+emission lines, fit_gas:{fit_gas} ...");
    cf, SPS = auto_rsp_elines_single_main(
        wl__w, f__w, ef__w, ssp_file,
        config_file=config_file,
        ssp_nl_fit_file=ssp_nl_fit_file, sigma_inst=sigma_inst, out_file="NOT_USED",
        mask_list=mask_list, elines_mask_file=elines_mask_file, fit_gas=fit_gas,
        min=min, max=max, w_min=w_min, w_max=w_max, nl_w_min=nl_w_min, nl_w_max=nl_w_max,
        input_redshift=input_redshift, delta_redshift=delta_redshift,
        min_redshift=min_redshift, max_redshift=max_redshift,
        input_sigma=input_sigma, delta_sigma=delta_sigma, min_sigma=min_sigma, max_sigma=max_sigma,
        input_AV=input_AV, delta_AV=delta_AV, min_AV=min_AV, max_AV=max_AV,
        plot=plot, single_ssp=False, ratio=ratio, y_ratio=y_ratio, fit_sigma_rnd=fit_sigma_rnd,
        sps_class=StellarSynthesis, SPS_master=SPS_master , SN_CUT=  SN_CUT 
    )
    print(f"# END: fitting the continuum+emission lines, fit_gas:{fit_gas} ...");
    print("##############################");
    #
    # There is refinement in the fitting
    #
    print(f"# refine_gas: {refine_gas}");
    if refine_gas:
        print(f"# START: refining gas fitting, refine_gas:{refine_gas} ...");
        if sigma_gas is None: sigma_gas = 3.0
        if out_path is None: out_path = "."
        wl_mask = (w_min<=wl__w)&(wl__w<=w_max)
        gas_wl, gas_fl = SPS.spectra["orig_wave"][wl_mask], (SPS.output_spectra_list[0] - SPS.output_spectra_list[1])[wl_mask]
        config_filenames, wl_chunks, _, wave_peaks_tot_rf = detect_create_ConfigEmissionModel(
            gas_wl, gas_fl,
            redshift=input_redshift,
            sigma_guess=sigma_gas,
            chunks=4,
            polynomial_order=1,
            polynomial_coeff_guess=[0.000, 0.001],
            polynomial_coeff_boundaries=[[-1e13, 1e13], [-1e13, 1e13]],
            flux_boundaries_fact=[0.001, 1000],
            sigma_boundaries_fact=[0.1, 1.5],
            v0_boundaries_add=[-1000, 1000],
            peak_find_nsearch=1,
            peak_find_threshold=0.2,
            peak_find_dmin=1,
            crossmatch_list_filename=elines_mask_file,
            crossmatch_absdmax_AA=5,
            crossmatch_redshift_search_boundaries=cc_redshift_boundaries,
            sort_by_flux=True,
            output_path=out_path,
            label=spaxel_id,
            verbose=0,
            plot=0,
        )

        create_emission_lines_mask_file_from_list(wave_peaks_tot_rf, eline_half_range=3*sigma_gas, output_path=out_path, label=spaxel_id)
        create_emission_lines_file_from_list(wave_peaks_tot_rf, output_path=out_path, label=spaxel_id)
        create_ConfigAutoSSP_from_lists(wl_chunks, config_filenames, output_path=out_path, label=spaxel_id)

        config_file = os.path.join(out_path, f"{spaxel_id}.autodetect.auto_ssp_several.config")
        elines_mask_file = os.path.join(out_path, f"{spaxel_id}.autodetect.emission_lines.txt")

        cf, SPS = auto_rsp_elines_single_main(
            wl__w, f__w, ef__w, ssp_file,
            config_file=config_file,
            ssp_nl_fit_file=ssp_nl_fit_file, sigma_inst=sigma_inst, out_file="NOT_USED",
            mask_list=mask_list, elines_mask_file=elines_mask_file, fit_gas=fit_gas,
            min=min, max=max, w_min=w_min, w_max=w_max, nl_w_min=nl_w_min, nl_w_max=nl_w_max,
            input_redshift=input_redshift, delta_redshift=delta_redshift,
            min_redshift=min_redshift, max_redshift=max_redshift,
            input_sigma=input_sigma, delta_sigma=delta_sigma, min_sigma=min_sigma, max_sigma=max_sigma,
            input_AV=input_AV, delta_AV=delta_AV, min_AV=min_AV, max_AV=max_AV,
            plot=plot, single_ssp=False, ratio=ratio, y_ratio=y_ratio, fit_sigma_rnd=fit_sigma_rnd,
            sps_class=StellarSynthesis
        )
        print(f"# END: refining gas fitting, refine_gas:{refine_gas} ...");
        print("########################################");
    print("# END RSP fitting...");
    print("########################################");
    return cf, SPS



####################################################
# MAIN script. Create a simulation using a yaml file
####################################################

def _main(cmd_args=sys.argv[1:]):
  PLATESCALE = 112.36748321030637

  #    print(f'n_MC = {__n_Monte_Carlo__}')
  #    quit()

  parser = argparse.ArgumentParser(
    description="lvm-dap-gen-out-mod LVM_FILE DAP_table_in label CONFIG.YAML"
  )

    
  parser.add_argument(
    "lvm_file", metavar="lvm_file",
    help="input LVM spectrum that was fitted"
  )
  
  parser.add_argument(
    "DAP_table_in",
    help="DAP file generated by the LVM-DAP fitting"
  )

  parser.add_argument(
    "label",
    help="Label for the output file, e.g.,  LABEL.output.fits.gz"
  )
  
  parser.add_argument(
    "config_yaml",
    help="config_yaml with the fitting parameters"
  )


  
  
  f_scale=1e16
  
  
  parser.add_argument(
    "-d", "--debug",
    help="debugging mode. Defaults to false.",
    action="store_true"
  )


  parser.add_argument(
    "-output_path",
    help="Directory to store the results",
    default='./'
  )

  parser.add_argument(
    "--plot", type=np.int,
    help="whether to plot (1) or not (0, default) the fitting procedure. If 2, a plot of the result is store in a file without display on screen",
    default=0
  )
  parser.add_argument(
    "--flux-scale", metavar=("min","max"), type=np.float, nargs=2,
    help="scale of the flux in the input spectrum",
    default=[-1,1]
  )
  
  
  args = parser.parse_args(cmd_args)
  print(cmd_args)
  print(args)
  if not args.debug:
    sys.excepthook = _no_traceback
  else:
    pprint("COMMAND LINE ARGUMENTS")
    pprint(f"{args}\n")

    
  dap_fwhm=1.0

  #
  # Read the YAML file
  #
  print(args.config_yaml)
  with open(args.config_yaml, 'r') as yaml_file:
    dap_config_args = yaml.safe_load(yaml_file)
    #
    # We add the full list of arguments
    #
  dict_param={}
  for k, v in dap_config_args.items():
    if(isinstance(v, str)):
      v=v.replace("..",dap_config_args['lvmdap_dir'])
      dict_param[k]=v
    parser.add_argument(
      '--' + k, default=v
    )


  args = parser.parse_args(cmd_args)
  config_file=args.config_file
  print(f'config_file = {config_file}')


  try:
    smooth_size = args.smooth_size
  except:
    smooth_size = 21

  try:
    n_leg = args.n_leg
  except:
    n_leg = 11

  
  lvmCFrame=args.lvm_file
  dap_file=args.DAP_table_in
  out_file_fit = os.path.join(args.output_path, f"{args.label}.output.fits.gz")

  
  print("##############################################")
  print("# Reading input files...")
  hdu_org=fits.open(lvmCFrame)
  tab_DAP=read_DAP_file(dap_file,verbose=False)
  tab_DAP.sort('fiberid')
  print(f"# N.Spec. input: {len(tab_DAP)}")                                
  print("# Done...")
  print("##############################################")
  
  if ((args.flux_scale[0]==-1) and (args.flux_scale[1]==1)):
    args.flux_scale[0]=args.flux_scale_org[0]
    args.flux_scale[1]=args.flux_scale_org[1]
    

  
  tab_PT_org = read_PT(lvmCFrame,'none',ny_range=None)


  hdu_org['FLUX'].data=hdu_org['FLUX'].data[tab_PT_org['mask']]

  hdu_org['FLUX'].data = replace_nan_inf_with_adjacent_avg( hdu_org['FLUX'].data)

  
  
  wave = hdu_org['WAVE'].data
  crpix1 = 1
  crval1 = wave[0]
  cdelt1 = wave[1]-wave[0]
  (ny,nx)=hdu_org['FLUX'].data.shape
  model_spectra = np.zeros((9,ny,nx))
  model_spectra[0,:,:]=hdu_org['FLUX'].data
  hdr_out = fits.Header()
  hdr_out['CRPIX1'] = crpix1
  hdr_out['CRVAL1'] = crval1
  hdr_out['CDELT1'] = cdelt1

  # Add the NAME fields
  hdr_out['NAME0'] = 'org_spec'
  hdr_out['NAME1'] = 'model_spec'
  hdr_out['NAME2'] = 'mod_joint_spec'
  hdr_out['NAME3'] = 'gas_spec'
  hdr_out['NAME4'] = 'res_joint_spec'
  hdr_out['NAME5'] = 'no_gas_spec'
  hdr_out['NAME6'] = 'np_mod_spec'
  hdr_out['NAME7'] = 'pe_mod_spec'
  hdr_out['NAME8'] = 'pk_mod_spec'

  
  pe_e=[]
  w_e=[]
  for td_col in tab_DAP.columns:
    if ((td_col.find("_pe_")==-1) & (td_col.find("_pek_")==-1) &  (td_col.find("e_flux_")>-1)):
      l_name = td_col.replace('e_flux_','')
      w_name = float(l_name.split("_")[-1])
      pe_e.append(l_name)
      w_e.append(w_name)
  print(f'# Number of emission lines in the model: {len(pe_e)}')
  #
  # Create emission lines!
  #
  print("##############################################")
  print('# Creating NP the emission line models\n')
  spec2D_elines=0.0*hdu_org['FLUX'].data
  for i,spec2D_now in enumerate(spec2D_elines):
    for j,(pe_now,w_now) in enumerate(zip(pe_e,w_e)):        
      F=np.abs(tab_DAP[f'flux_{pe_now}'][i])
      D=tab_DAP[f'disp_{pe_now}'][i]
      V=tab_DAP[f'vel_{pe_now}'][i]
      spec_eline = eline(wave,w_now,F,D,V,dap_fwhm=dap_fwhm)
      spec2D_now += spec_eline
      
  print("# Done...")
  print("##############################################")
  model_spectra[6,:,:]=np.copy(spec2D_elines/f_scale)


  pe_e=[]
  w_e=[]
  for td_col in tab_DAP.columns:
    if ((td_col.find("_pe_")>-1) & (td_col.find("_pek_")==-1) &  (td_col.find("e_flux_")>-1)):
      l_name = td_col.replace('e_flux_','')
      w_name = float(l_name.split("_")[-1])
      pe_e.append(l_name)
      w_e.append(w_name)
  print(f'# Number of emission lines in the model: {len(pe_e)}')
  #
  # Create emission lines!
  #
  print("##############################################")
  print('# Creating the emission line models\n')
  spec2D_elines=0.0*hdu_org['FLUX'].data
  for i,spec2D_now in enumerate(spec2D_elines):
    for j,(pe_now,w_now) in enumerate(zip(pe_e,w_e)):        
      F=np.abs(tab_DAP[f'flux_{pe_now}'][i])
      D=tab_DAP[f'disp_{pe_now}'][i]
      V=tab_DAP[f'vel_{pe_now}'][i]
      spec_eline = eline(wave,w_now,F,D,V,dap_fwhm=dap_fwhm)
      spec2D_now += spec_eline
      
  print("# Done...")
  print("##############################################")
  model_spectra[7,:,:]=np.copy(spec2D_elines/f_scale)

  
  seed = print_time(print_seed=False, get_time_only=True)

  #
  a_redshift=np.array(tab_DAP['redshift_st'])
  a_sigma=np.array(tab_DAP['disp_st'])
  a_AV=np.array(tab_DAP['Av_st'])

  n_coeffs=0
  for cols in tab_DAP.columns:
        if (cols.find('min_W_rsp_')>-1):
          n_coeffs=n_coeffs+1
  print(f'# N.RSP: {n_coeffs}')
  a_coeffs_input = np.zeros((ny,n_coeffs))
  for i in np.arange(n_coeffs):
    w_id = f'W_rsp_{i}'
    a_coeffs_input[:,i]=np.array(tab_DAP[w_id])
  
  
  for i,model_st in enumerate(model_spectra[1,:,:]):
   # print(f'### {i}/{ny}')
   # print(f'### (AV,redshift,sigma) : {a_AV[i]},{a_redshift[i]},{a_sigma[i]}')
    coeffs_input = a_coeffs_input[i,:]
    org_spec = hdu_org['FLUX'].data[i,:]
    res_spec = 0.1*np.abs(org_spec)
    wl__w, f__w, ef__w = wave, org_spec, res_spec
    out_file = os.path.join(args.output_path, f"junk_{args.label}.fits")

    
    if (i==0):
      SPS_master=None

    _, SPS = model_rsp_elines_single_main(
      wl__w, f__w, ef__w, args.rsp_file,  config_file, out_file, ssp_nl_fit_file=args.rsp_file,
      w_min=args.w_range[0], w_max=args.w_range[1], nl_w_min=args.w_range_nl[0],
      nl_w_max=args.w_range_nl[1], mask_list=args.mask_file,
      min=args.flux_scale[0], max=args.flux_scale[1], elines_mask_file=args.emission_lines_file,
      fit_gas=not args.ignore_gas, 
      input_redshift=a_redshift[i], delta_redshift=0, 
      min_redshift=a_redshift[i], max_redshift=a_redshift[i],
      input_sigma=a_sigma[i], delta_sigma=0, min_sigma=a_sigma[i], max_sigma=a_sigma[i],
      input_AV=a_AV[i], delta_AV=0, min_AV=a_sigma[i], max_AV=a_sigma[i],
      sigma_inst=args.sigma_inst, plot=args.plot, sps_class=StellarSynthesis,
      coeffs_input=coeffs_input, SPS_master=SPS_master
    )
    SPS_master=SPS
    #print(i)
    #print(SPS.spectra['model_min'].shape)
    #print(SPS.spectra['model_min'])
    model_st = np.array(SPS.spectra['model_min'])
    model_spectra[1,i,:] = model_st
#    print(model_st)
    print(f'model_st stats: {np.median(model_st)} {np.min(model_st)} {np.max(model_st)}')
    #print(model_st.shape)
    #model_spectra[1,i,:]=SPS.spectra['model_min']
    print(f'### **** {i}/{ny} *****')

#  print(model_spectra.shape)
  gas_spectra = model_spectra[0,:,:] - model_spectra[1,:,:]
  l_smooth_spectra = gas_spectra*0.0
  for idx,gas in enumerate(gas_spectra): 
    smooth = median_filter(gas, size=smooth_size, mode='reflect')
    l_smooth = fit_legendre_polynomial(wl__w, smooth,n_leg)
    # print(f'{idx} {np.median(smooth)} {np.median(l_smooth)}')
    l_smooth_spectra[idx,:] = l_smooth
  model_spectra[2,:,:] = model_spectra[1,:,:] + model_spectra[6,:,:] + l_smooth_spectra
  model_spectra[3,:,:] = model_spectra[0,:,:] - (model_spectra[1,:,:] + l_smooth_spectra)
  model_spectra[4,:,:] = model_spectra[0,:,:] - model_spectra[2,:,:]
  model_spectra[5,:,:] = model_spectra[0,:,:] - model_spectra[6,:,:]

  dump_rss_output(out_file_fit=out_file_fit, wavelength=wl__w, model_spectra=model_spectra)
#  hdu_org.writeto(out_file_fit,overwrite=True)
  #sys.stdout = original_stdout
  print("#############################")
  print('# ALL DONE !!! ')
  print("#############################")
  
