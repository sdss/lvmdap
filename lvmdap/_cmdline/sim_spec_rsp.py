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
from lvmdap.pyFIT3D.common.io import clean_preview_results_files, print_time, read_spectra

# 18.11.2023
# So far we were ysing the auto_ssp_tools from lvmdap.pyFIT3D
# We will attempt to modify them
#from lvmdap.pyFIT3D.common.auto_ssp_tools import auto_ssp_elines_single_main
from lvmdap.modelling.auto_rsp_tools import auto_rsp_elines_single_main

from lvmdap.pyFIT3D.common.auto_ssp_tools import load_rss, dump_rss_output
from lvmdap.pyFIT3D.common.io import clean_preview_results_files, print_time, read_spectra

from lvmdap.pyFIT3D.common.gas_tools import detect_create_ConfigEmissionModel
from lvmdap.pyFIT3D.common.io import create_ConfigAutoSSP_from_lists
from lvmdap.pyFIT3D.common.io import create_emission_lines_file_from_list
from lvmdap.pyFIT3D.common.io import create_emission_lines_mask_file_from_list
#from lvmdap.pyFIT3D.common.tools import read_coeffs_CS

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
# RSP version of the auto_ssp_elines_rnd from lvmdap.pyFIT3D
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
    description="lvm-dap-yaml LVM_FILE OUTPUT_LABEL CONFIG.YAML"
  )

    
  parser.add_argument(
    "lvm_file", metavar="lvm_file",
    help="input LVM spectrum to use for the simulation"
  )
  
  parser.add_argument(
    "label",
    help="string to label the current run"
  )
  
  parser.add_argument(
    "config_yaml",
    help="config_yaml with the fitting parameters"
  )

#  ref_file=args.DAP_output_in
#  dap_file=args.DAP_table_in
  
  parser.add_argument(
    "-n_sim",  metavar="n_sim", type=np.int64,
    help="Number of simulated spectra. Default =10",
    default=10
  )
  
  parser.add_argument(
    "-n_st",  metavar="n_st", type=np.int64,
    help="Number of stars included in the model. Default =10",
    default=10
  )
  
  parser.add_argument(
    "-f_st", metavar = 'f_scale_st', type=np.float64,
    help="Scaling factor applied to the stellar population spectra (~S/N level). Default = 1.0",
    default=10.0
  )
  
  parser.add_argument(
    "-f_el", metavar = 'f_scale_el', type=np.float64,
    help="Scaling factor applied to the emission lines with respect to the reference. Default = 1.0",
    default=1.0
  )

  parser.add_argument(
    "-dap_fwhm", metavar = 'dap_fwhm', type=np.float64,
    help="Scaling factor applied to input DAP non-parametric dispersion for the emission lines to transform to the sigma of a Gaussian. Default = 2.354",
    default=2.354
  )

  f_scale=1e16
  #label=f'sim_{n_sim}_{n_st}_{f_scale_st}_{f_scale_el}'

  #
  # Args to include in the YAML file
  #
  # lvmCFrame=f'data/lvmCFrame-{fileID}.fits'
  # ref_file=f"output_ofelia_new/dap-rsp30-sn20-{fileID}.output.fits.gz"
  # dap_file=f"output_ofelia_new/dap-rsp30-sn20-{fileID}.dap.fits.gz"
  
  
  
  parser.add_argument(
    "-d", "--debug",
    help="debugging mode. Defaults to false.",
    action="store_true"
  )
  
  
  
  args = parser.parse_args(cmd_args)
  print(cmd_args)
  print(args)
  if not args.debug:
    sys.excepthook = _no_traceback
  else:
    pprint("COMMAND LINE ARGUMENTS")
    pprint(f"{args}\n")

    
  n_sim=args.n_sim
  n_st=args.n_st
  f_scale_st=args.f_st
  f_scale_el=args.f_el
  dap_fwhm=args.dap_fwhm

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

  #
  # We transform it to a set of arguments
  #

  parser.add_argument(
    "--flux-scale", metavar=("min","max"), type=np.float64, nargs=2,
    help="scale of the flux in the input spectrum",
    default=[-1, +1]
  )
  
  parser.add_argument(
    "--plot", type=np.int64,
    help="whether to plot (1) or not (0, default) the fitting procedure. If 2, a plot of the result is store in a file without display on screen",
    default=0
  )
  
  args = parser.parse_args(cmd_args)

  args.label=f'sim_{args.label}_{n_sim}_{n_st}_{f_scale_st}_{f_scale_el}'
    


  lvmCFrame=args.lvm_file
  ref_file=args.DAP_output_in
  dap_file=args.DAP_table_in
  
  print("##############################################")
  print("# Reading reference files...")
  hdu_org=fits.open(lvmCFrame)
  hdu=fits.open(ref_file)
  tab_DAP=read_DAP_file(dap_file,verbose=False)
  tab_DAP.sort('fiberid')
  print(f"# N.Spec. input: {len(tab_DAP)}")                                
  print("# Done...")
  print("##############################################")
  
  
  
  tab_PT_org = read_PT(lvmCFrame,'none',ny_range=None)
  #hdu_org[0].header['dap_ver']=1.240208
  hdu_org['FLUX'].data=hdu_org['FLUX'].data[tab_PT_org['mask']]
  hdu_org['ERROR'].data=hdu_org['ERROR'].data[tab_PT_org['mask']]
  hdu_org['MASK'].data=hdu_org['MASK'].data[tab_PT_org['mask']]
  hdu_org['FWHM'].data=hdu_org['FWHM'].data[tab_PT_org['mask']]
  hdu_org['SKY'].data=hdu_org['SKY'].data[tab_PT_org['mask']]
  hdu_org['SKY_ERROR'].data=hdu_org['SKY_ERROR'].data[tab_PT_org['mask']]
  hdu_org['SLITMAP'].data=hdu_org['SLITMAP'].data[tab_PT_org['mask']]
  #hdu_org.info()
  
  pe_e=[]
  w_e=[]
  for td_col in tab_DAP.columns:
    if ((td_col.find("_pe_")==-1) & (td_col.find("e_flux_")>-1)):
      l_name = td_col.replace('e_flux_','')
      w_name = float(l_name.split("_")[-1])
      pe_e.append(l_name)
      w_e.append(w_name)
  print(f'# Number of emission lines in the model: {len(pe_e)}')



  hdu_org['FLUX'].data=hdu_org['FLUX'].data[0:n_sim,:]
  hdu_org['ERROR'].data=hdu_org['ERROR'].data[0:n_sim,:]
  hdu_org['MASK'].data=hdu_org['MASK'].data[0:n_sim,:]
  hdu_org['FWHM'].data=hdu_org['FWHM'].data[0:n_sim,:]
  hdu_org['SKY'].data=hdu_org['SKY'].data[0:n_sim,:]
  hdu_org['SKY_ERROR'].data=hdu_org['SKY_ERROR'].data[0:n_sim,:]
  hdu_org['SLITMAP'].data=hdu_org['SLITMAP'].data[0:n_sim]
  #hdu_org.info()
  
  #print(hdu[0].data.shape)
  #print(hdu[0].header["NAXIS1"])
  #print(hdu[0].header["NAXIS2"])
  #print(hdu[0].header["NAXIS3"])
  #print(hdu[0].header["CRVAL1"])
  

  wave = hdu[0].header["CRVAL1"]+hdu[0].header["CDELT1"]*(np.arange(hdu[0].header["NAXIS1"]))#-hdu[0].header["CRPIX1"])



  #
  # Create emission lines!
  #
  spec2D_elines=0.0*hdu_org['FLUX'].data
  for i,spec2D_now in enumerate(spec2D_elines):
    for j,(pe_now,w_now) in enumerate(zip(pe_e,w_e)):        
      F=np.abs(tab_DAP[f'flux_{pe_now}'][i])
      D=tab_DAP[f'disp_{pe_now}'][i]
      V=tab_DAP[f'vel_{pe_now}'][i]
      spec_eline = eline(wave,w_now,F,D,V,dap_fwhm=dap_fwhm)
      spec2D_now += spec_eline
      
  i_spec=0
  org_spec = hdu[0].data[0,i_spec,:]
  mod_spec = hdu[0].data[1,i_spec,:]
  res_spec = hdu[0].data[4,i_spec,:]
  

  # OUTPUT NAMES ---------------------------------------------------------------------------------
  out_file_elines = os.path.join(args.output_path, f"elines_{args.label}")
  out_file_single = os.path.join(args.output_path, f"single_{args.label}")
  out_file_coeffs = os.path.join(args.output_path, f"coeffs_{args.label}")
  out_file_fit = os.path.join(args.output_path, f"sim_{args.label}.fits")
  out_file_ps = os.path.join(args.output_path, args.label)
  out_file_stdout = os.path.join(args.output_path, f"log_{args.label}.log")
  


  if args.clear_outputs:
    clean_preview_results_files(out_file_ps, out_file_elines, out_file_single, out_file_coeffs, out_file_fit)

  seed = print_time(print_seed=False, get_time_only=True)

  #print(out_file_ps)
  #
  # Range of NL parameters
  #
  a_redshift=np.random.normal(0.5*(args.redshift[3]+args.redshift[2]),0.33*(args.redshift[3]-args.redshift[2]), n_sim)
  a_sigma=np.random.normal(0.5*(args.sigma[3]+args.sigma[2]),0.33*(args.sigma[3]-args.sigma[2]), n_sim)
  a_AV=np.random.normal(0.5*(args.AV[3]+args.AV[2]),0.33*(args.AV[3]-args.AV[2]), n_sim)
  
#  sys.stdout = open(out_file_stdout, 'w')
  for i_sim in np.arange(n_sim):
    random_map=np.random.normal(0,1.0,size=(hdu[0].data.shape[1],hdu[0].data.shape[2]))
    err_map=random_map*hdu[0].data[4,:,:]
    random_index=hdu[0].data.shape[1]*np.random.rand(hdu[0].data.shape[2])
    random_index=random_index.astype(int)
    err_spec=0.0*res_spec
    err_spec_t=0.0*res_spec
    for i in np.arange(hdu[0].data.shape[2]):
      err_spec[i]=err_map[random_index[i],i]
      #
      # Change this
      #
      err_spec_t[i]=hdu[0].data[4,:,:][random_index[i],i]
    err_spec_t=np.nan_to_num(err_spec_t,nan=np.nanmedian(err_spec_t))
    wl__w, f__w, ef__w = wave, org_spec, res_spec
#    out_file = args.output_path+'/sim_junk.fits'
    out_file = os.path.join(args.output_path, f"sim_junk_{args.label}.fits")
    config_file=args.config_file
    rsp_hdu=fits.open(args.rsp_file)
    n_coeffs=rsp_hdu[0].header['NAXIS2']
    coeffs_random=np.random.rand(n_coeffs)
    coeffs_select=np.zeros(n_coeffs)
    choice=np.random.choice(rsp_hdu[0].header['NAXIS2'], n_st)
    coeffs_select[choice]=coeffs_random[choice]
    coeffs_input=coeffs_select/np.sum(coeffs_select)
    model_spectra = []
    
    if (i_sim==0):
      SPS_master=None
    print(f'### (AV,redshift,sigma) : {a_AV[i_sim]},{a_redshift[i_sim]},{a_sigma[i_sim]}')    
    _, SPS = model_rsp_elines_single_main(
      wl__w, f__w, ef__w, args.rsp_file,  config_file, out_file, ssp_nl_fit_file=args.rsp_file,
      w_min=args.w_range[0], w_max=args.w_range[1], nl_w_min=args.w_range_nl[0],
      nl_w_max=args.w_range_nl[1], mask_list=args.mask_file,
      min=args.flux_scale[0], max=args.flux_scale[1], elines_mask_file=args.emission_lines_file,
      fit_gas=not args.ignore_gas, 
      input_redshift=a_redshift[i_sim], delta_redshift=args.redshift[1], 
      min_redshift=args.redshift[2], max_redshift=args.redshift[3],
      input_sigma=a_sigma[i_sim], delta_sigma=args.sigma[1], min_sigma=args.sigma[2], max_sigma=args.sigma[3],
      input_AV=a_AV[i_sim], delta_AV=args.AV[1], min_AV=args.AV[2], max_AV=args.AV[3],
      sigma_inst=args.sigma_inst, plot=args.plot, sps_class=StellarSynthesis,
      coeffs_input=coeffs_input, SPS_master=SPS_master
    )
    SPS_master=SPS
    i=i_sim
    SPS.output_gas_emission(filename=out_file_elines, spec_id=i)
    SPS.output_coeffs_MC(filename=out_file_coeffs, write_header=i==0)
    SPS.output(filename=out_file_ps, write_header=i==0, block_plot=False)
    hdu_org['FLUX'].data[i_sim,:]=SPS.spectra['model_min']+err_spec_t/f_scale_st
    print(f'### **** {i_sim}/{n_sim} *****')

  #
  # Add elines
  #
  hdu_org['FLUX'].data=(f_scale_st*hdu_org['FLUX'].data+f_scale_el*spec2D_elines)/f_scale
  mean_error = np.nanmean(hdu_org['ERROR'].data)
  #hdu_org['ERROR'].data=0.001*np.abs(np.nan_to_num(hdu_org['FLUX']))
  hdu_org['ERROR'].data=np.nan_to_num(hdu_org['ERROR'].data,nan=mean_error)#/f_scale
  mean_flux = np.nanmean(hdu_org['FLUX'].data)
  hdu_org['ERROR'].data += (0.7*np.abs(np.nan_to_num(hdu_org['FLUX'].data,nan=mean_flux))+0.3*mean_flux)
  mean_error = np.nanmean(hdu_org['ERROR'].data)
  hdu_org['ERROR'].data *= mean_flux/mean_error/f_scale_st
  
  hdu_org.writeto(out_file_fit,overwrite=True)
  #sys.stdout = original_stdout
  print("#############################")
  print('# Simulation done')
  print("#############################")
  
  
  
  
  tab_PT = tab_PT_org[tab_PT_org['mask']]
  tab_PT = tab_PT[0:n_sim]
  tab_DAP = tab_DAP[0:n_sim]
  tab_RSP=read_rsp(file_ssp=out_file_ps)
  tab_RSP.add_column(tab_PT['id'].value,name='id',index=0)
  tab_COEFFS=read_coeffs_RSP(coeffs_file=out_file_coeffs)
  id_coeffs=[]    
  for id_fib in tab_COEFFS['id_fib']:
    id_coeffs.append(tab_PT['id'].value[id_fib])
  id_coeffs=np.array(id_coeffs)
  tab_COEFFS.add_column(id_coeffs,name='id',index=0)

  tab_PE=read_elines_RSP(elines_file=out_file_elines)
  id_elines=[]    
  for id_fib in tab_PE['id_fib']:
    id_elines.append(tab_PT['id'].value[id_fib])
  id_elines=np.array(id_elines)
  tab_PE.add_column(id_elines,name='id',index=0)
 
  #
  # Rename some entries!
  #
  tab_RSP.rename_column('Av','Av_st')
  tab_RSP.rename_column('e_Av','e_Av_st')
  tab_RSP.rename_column('z','z_st')
  tab_RSP.rename_column('e_z','e_z_st')
  tab_RSP.rename_column('disp','disp_st')
  tab_RSP.rename_column('e_disp','e_disp_st')
  tab_RSP.rename_column('flux','flux_st')
  tab_RSP.rename_column('redshift','redshift_st')
  tab_RSP.rename_column('med_flux','med_flux_st')
  tab_RSP.rename_column('e_med_flux','e_med_flux_st')
  tab_RSP.rename_column('sys_vel','vel_st')
  #
  # Parametric elines
  #
  tab_PE.rename_column('flux','flux_pe')
  tab_PE.rename_column('e_flux','e_flux_pe')
  tab_PE.rename_column('disp','disp_pe')
  tab_PE.rename_column('e_disp','e_disp_pe')
  tab_PE.rename_column('vel','vel_pe')
  tab_PE.rename_column('e_vel','e_vel_pe')
  
  #
  # id	id_fib	rsp	TEFF	LOGG	META	ALPHAM	COEFF	Min.Coeff	log(M/L)	AV	N.Coeff	Err.Coeff
  #
  tab_COEFFS.rename_column('rsp','id_rsp')
  tab_COEFFS.rename_column('TEFF','Teff_rsp')
  tab_COEFFS.rename_column('LOGG','Log_g_rsp')
  tab_COEFFS.rename_column('META','Fe_rsp')
  tab_COEFFS.rename_column('ALPHAM','alpha_rsp')
  tab_COEFFS.rename_column('COEFF','W_rsp')
  tab_COEFFS.rename_column('Min.Coeff','min_W_rsp')
  tab_COEFFS.rename_column('log(M/L)','log_ML_rsp')
  tab_COEFFS.rename_column('AV','Av_rsp')
  tab_COEFFS.rename_column('N.Coeff','n_W_rsp')
  tab_COEFFS.rename_column('Err.Coeff','e_W_rsp')
  
    
  #
  # order parametric emission line table
  #
  a_wl = np.unique(tab_PE['wl'])
  print(a_wl)
  I=0
  for wl_now in a_wl:
    if (wl_now>0.0):
      tab_PE_now=tab_PE[tab_PE['wl']==wl_now]
      tab_PE_tmp=tab_PE_now['id','flux_pe','e_flux_pe','disp_pe','e_disp_pe','vel_pe','e_vel_pe']
      for cols in tab_PE_tmp.colnames:        
        if (cols != 'id'):
          tab_PE_tmp.rename_column(cols,f'{cols}_{wl_now}')
        if (I==0):
          tab_PE_ord=tab_PE_tmp
        else:
          tab_PE_ord=tab_join(tab_PE_ord,tab_PE_tmp,keys=['id'],join_type='left')
          I=I+1

            
  #
  # Order COEFFS table
  #
  a_rsp=np.unique(tab_COEFFS['id_rsp'])
  for I,rsp_now in enumerate(a_rsp):
    tab_C_now=tab_COEFFS[tab_COEFFS['id_rsp']==rsp_now]
    tab_C_tmp=tab_C_now['id','Teff_rsp', 'Log_g_rsp', 'Fe_rsp',                        'alpha_rsp', 'W_rsp', 'min_W_rsp',                        'log_ML_rsp', 'Av_rsp', 'n_W_rsp', 'e_W_rsp']
    for cols in tab_C_tmp.colnames:        
        if (cols != 'id'):
            tab_C_tmp.rename_column(cols,f'{cols}_{rsp_now}')
    if (I==0):
        tab_C_ord=tab_C_tmp
    else:
        tab_C_ord=tab_join(tab_C_ord,tab_C_tmp,keys=['id'],join_type='left')
            


  out_file_dap = os.path.join(args.output_path, f"{args.label}.dap.ecsv")
  tab_DAP_sim=tab_DAP
  for c_C_ord in tab_C_ord.columns:
    tab_DAP_sim[c_C_ord]=tab_C_ord[c_C_ord]
  for c_RSP in tab_RSP.columns:
    tab_DAP_sim[c_RSP]=tab_RSP[c_RSP]
  tab_DAP_sim.write(out_file_dap,overwrite=True)


  print("####################################")
  print("# All done and storaged  ####################")
  print("####################################")

  
