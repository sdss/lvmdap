#!/usr/bin/env python3
"""LVM-DAP command-line entry points.

Core functions and their shared dependencies live in lvmdap.dap_core.
Importing the core performs the one-time BLAS threading initialization
before any of its NumPy/FIT3D imports.
"""

from lvmdap.dap_core import *  # noqa: F401,F403 - preserve original CLI namespace

from lvmdap import memory_diagnostics as _mem
from lvmdap.cube_io import write_dap_parameter_cubes

def _main(cmd_args=sys.argv[1:]):
    print("########################################")
    print("### Do nothing")


#########################################################################
# MAIN script. Uses the entries from commands lines and the YAML file
#########################################################################
def _dap_yaml(cmd_args=sys.argv[1:]):



    print(f'##############################################################')
    print(f'# LVM-DAP Started ...')
    print(f'##############################################################')
    PLATESCALE = 112.36748321030637
    empirical_errors = True
    #####################################################################
    # LVM DAP version and date
    #####################################################################
    dap_ver = '2.0.0'
    dap_date = '2026-10-09'


    #####################################################################
    # START: Initialize DAP configuration
    #####################################################################
    _mem.report('after_imports')
    args = parse_dap_arguments(cmd_args)
    args = initialize_dap_parameters(args)
    hdu_info = create_dap_info_hdu(args)

    ny_range = args.ny_range
    nx_range = args.nx_range
    out_plot_format = args.out_plot_format
    only_integrated = args.only_integrated
    sky_hack = args.sky_hack
    mask_to_val = args.mask_to_val

    auto_redshift = args.auto_redshift
    auto_z_min = args.auto_z_min
    auto_z_max = args.auto_z_max
    auto_z_del = args.auto_z_d
    dump_model = args.dump_model

    SN_CUT = args.SN_CUT
    SN_CUT_INT = args.SN_CUT_INT
    auto_z_bg = args.auto_z_bg
    auto_z_sc = args.auto_z_sc

    kin_fixed = args.kin_fixed
    N_MC = args.N_MC
    n_loops = args.n_loops
    w_range_FE = args.w_range_FE
    smooth_size = args.smooth_size
    n_leg = args.n_leg
    flux_scaling = float(getattr(args, "flux_scaling", 1e16))
    print(f'# Flux scaling factor: {flux_scaling}')
    #####################################################################
    # END: Initialize DAP configuration
    #####################################################################
    print(f'# Input parameters initialized ...')


    #####################################################################
    # START: Reading the input data to be analyzed
    #####################################################################
    print(f'# Reading input data ...')
    (wl__w, rss_flux_org, rss_eflux_org,
     hdr_flux_org, hdr_0, LSF_rss_org,
     tab_PT_org, LSF_mean, input_mask_org, bad_pixels_org) = load_input_data(
         args,
         ny_range=ny_range,
         nx_range=nx_range,
         sky_hack=sky_hack,
         mask_to_val=mask_to_val,
         flux_scale=flux_scaling
     )
    _mem.report('after_load_input', arrays={'rss_flux_org': rss_flux_org, 'rss_eflux_org': rss_eflux_org, 'LSF_rss_org': LSF_rss_org, 'input_mask_org': input_mask_org, 'bad_pixels_org': bad_pixels_org})
    ###############################################################################
    # This is the default tab_el_strong. We dynamically select later a better list
    ###############################################################################
    tab_el_strong = read_tab_EL(args.emission_lines_file)
    ######################################################################
    # We mask the non-science spectra and broken fibers ones
    ######################################################################
    rss_flux = rss_flux_org[tab_PT_org['mask']]
    rss_eflux = rss_eflux_org[tab_PT_org['mask']]
    tab_PT = tab_PT_org[tab_PT_org['mask']]
    LSF_rss = LSF_rss_org[tab_PT_org['mask']]
    input_mask = input_mask_org[tab_PT_org['mask']]
    bad_pixels = bad_pixels_org[tab_PT_org['mask']]
    hdr_flux=copy(hdr_flux_org)
    hdr_flux['NAXIS2']=len(tab_PT)
    print(f'# Number of spectra to analyze : {len(tab_PT)}')
    _mem.report('after_fiber_selection', arrays={'rss_flux_org': rss_flux_org, 'rss_flux': rss_flux, 'rss_eflux_org': rss_eflux_org, 'rss_eflux': rss_eflux, 'LSF_rss_org': LSF_rss_org, 'LSF_rss': LSF_rss, 'input_mask': input_mask, 'bad_pixels': bad_pixels})
    # Boolean fiber selection made independent copies of these arrays.
    # The selected rss_flux/rss_eflux/LSF_rss/masks are retained unchanged.
    del rss_flux_org, rss_eflux_org, LSF_rss_org, input_mask_org, bad_pixels_org
    _mem.report('after_original_rss_release', arrays={'rss_flux': rss_flux, 'rss_eflux': rss_eflux, 'LSF_rss': LSF_rss})

    #####################################################################################
    # Reading the emission lines file and identify the instrumental LSF and blended lines
    #####################################################################################
    tab_el=read_tab_EL(args.emission_lines_file_long)
    lsf = []
    blended = []
    for wl_el in tab_el['wl']:
      i_wl_el = find_closest_indices(wl__w, wl_el)
      lsf_now = LSF_mean[i_wl_el[0]]
      lsf.append(lsf_now)
      i_wl_near = find_closest_indices_different(tab_el['wl'], wl_el)
      dist_abs = np.abs(wl_el-tab_el['wl'][i_wl_near[0]])
      if (dist_abs<4*args.sigma_gas):
        blended.append(1)
      else:
        blended.append(0)
    blended = np.array(blended)    
    tab_el['blended'] = blended
    lsf = np.array(lsf)
    tab_el['lsf'] = lsf


    ########################################################
    # Clusters of emission lines for future fitting
    ########################################################

    el_clusters, el_cluster_id = cluster_emission_lines(
        tab_el,
        max_cluster_span=float(getattr(args, "max_cluster_span", 300.0)),
        max_adjacent_distance=float(getattr(args, "max_adjacent_distance", 100.0)),
    )
    pm_strong_clusters = select_pm_strong_clusters(
        tab_el, tab_el_strong,
        max_cluster_span=float(getattr(args, "max_cluster_span", 300.0)),
        max_adjacent_distance=float(getattr(args, "max_adjacent_distance", 100.0)),
    )
    print(f'# PM strong lines: {sum(len(c) for c in pm_strong_clusters)} '
          f'/ {len(tab_el)} long-list lines')


    #####################################################################
    # END: Reading the input data to be analyzed
    #####################################################################


    #####################################################################
    # START: Generate integrated spectra
    #####################################################################

    integrated = create_integrated_spectra(
        wl__w,
        rss_flux,
        rss_eflux,
        LSF_rss,
        auto_z_min=args.auto_z_min,
        auto_z_max=args.auto_z_max,
        SN_threshold=args.SN_CUT,
        we_range=(6500, 6600),
        cw_range=(6400, 6500),
        bad_pixels=bad_pixels
    )

    int_flux = integrated['int_flux']
    int_eflux = integrated['int_eflux']
    int_flux_we = integrated['int_flux_we']
    int_flux_cw = integrated['int_flux_cw']
    int_flux_w_ce = integrated['int_flux_w_ce']
    rat_flux_cw = integrated['rat_flux_cw']
    int_lsf = integrated['int_lsf']
    SN_map = integrated['SN_map']
    high_SN_mask = integrated['high_SN_mask']
    #m_flux = int_flux_cw
    #e_flux = int_eflux
    m_flux, e_flux, s_flux, m_flux_SN, m_flux_rss = \
      prepare_integrated_spectrum(int_flux_cw, int_eflux, smooth_box=51)
    m_flux_rss = m_flux[np.newaxis, :].copy()

    if ((args.flux_scale[0]==-1) and (args.flux_scale[1]==1)):
      args.flux_scale[0]=-0.1*np.abs(np.median(m_flux_rss))
      args.flux_scale[1]=3*np.abs(np.median(m_flux_rss))+10*np.std(m_flux_rss)

    #####################################################################
    # END: Generate integrated spectra
    #####################################################################
    _mem.report('after_integrated', arrays={'rss_flux': rss_flux, 'rss_eflux': rss_eflux, 'LSF_rss': LSF_rss, 'bad_pixels': bad_pixels, 'int_flux': int_flux, 'int_eflux': int_eflux})

    # Initialize kinematics
    vel__yx = np.zeros(1)
    sigma__yx = 0.8


    #####################################################################
    # START: Automatic search of the guess redshift
    #####################################################################

    (args,
     out_decompose_flux_rss,
     find_z_out_w,
     z_guess) = estimate_integrated_redshift(
         wl__w,
         int_flux_cw,
         e_flux,
         args,
         m_flux_SN=m_flux_SN
     )


    #####################################################################
    # END: Automatic redefine the guess redshift
    #####################################################################


    ###########################################################################
    #
    # START: Removing the emission lines and fitting just the stellar continuum 
    #
    ###########################################################################



    #########################################################
    # We fit the mean/integrated spectrum with RSPs 
    #
    # (1) We define and remove the output files.
    #     We try to minimize the reading/writing of files to speed up the process in version 2.0.0
    #
    try:
        os.makedirs(args.output_path)
        print(f'# dir {args.output_path} created')
    except:
        print(f'# dir {args.output_path} alrady exists')
    out_file_PT = os.path.join(args.output_path, f"m_{args.label}.PT.ecsv")
    out_file_fe = os.path.join(args.output_path, f"m_{args.label}.fe.ecsv")
    out_file_elines = os.path.join(args.output_path, f"m_{args.label}.elines.txt")
    out_file_single = os.path.join(args.output_path, f"m_{args.label}.single.txt")
    out_file_coeffs = os.path.join(args.output_path, f"m_{args.label}.coeffs.txt")
    out_file_fit = os.path.join(args.output_path, f"m_{args.label}.output.fits")
    out_file_ps = os.path.join(args.output_path, f"m_{args.label}.rsp.txt")

    if args.clear_outputs:
        clean_preview_results_files(out_file_ps, out_file_elines, out_file_single, out_file_coeffs, out_file_fit)
        clean_preview_results_files(out_file_fe, out_file_elines, out_file_single, out_file_coeffs, out_file_fit)
    # ----------------------------------------------------------------------------------------------

    #
    # (2) We set the random seed to ensure reproducibility of the fitting results
    #
    seed = print_time(print_seed=False, get_time_only=True)
    np.random.seed(seed)
    #
    # (3) We fit the integrated spectrum with RSPs
    #

    print(f'### START RSP fitting the integrated spectrum ###')
    lsf_mean = np.mean(LSF_rss, axis=0)
 
    n_narrow = int(getattr(args, "n_narrow", 3))
    n_medium = int(getattr(args, "n_medium", 5))
    n_broad = int(getattr(args, "n_broad", 21)) 
    n_iter = int(getattr(args, "n_iter", 5))

    i_fib = 0 

    tab_m_rsp, tab_m_coeffs, \
    tab_m_elines, tab_m_PM, \
    tab_m_fe, hdu_m_sps, SPS = fit_individual_rss_spectrum_or_adopt_result(
        wl=wl__w,
        flux=int_flux_cw,
        eflux=int_eflux,
        lsf=lsf_mean,
        tab_PT=tab_PT,
        i_fib=i_fib,
        tab_el=tab_el,
        el_clusters=el_clusters,
        hdr_flux=hdr_flux,
        args_copy=args,
        z_guess=args.redshift[0],
        plot=args.plot,
        adopt_st_z = True,
        SN_cut_st_z = args.SN_CUT_INT, 
        SPS_master=None,
        verbose=args.verbose,
        fit_PM=True,
        fit_NP=True,
        bad_pixel_mask=(
            np.all(bad_pixels, axis=0)
            | ~np.isfinite(int_flux_cw)
            | ~np.isfinite(int_eflux)
            | (int_eflux <= 0)
        ),
        empirical_errors = empirical_errors,
        pm_strong_clusters=pm_strong_clusters,
    )
    print(f'### END RSP fitting the integrated spectrum ###')

    label_strongest_line = 'Halpha_6562.85_pm'
    # In 'both' mode, use the original FIT3D-only integrated bootstrap for
    # choosing the RSS strong-line list and the shared PEK reference line.
    # The official PM table and MODEL still contain the chi-square winners.
    tab_m_PM_anchor = getattr(SPS, '_pm_fit3d_bootstrap', tab_m_PM)

    f_strongest = []
    name_strongest = []
    for column in tab_m_PM_anchor.colnames:
        if (column.startswith('flux_') and column.endswith('_pm')) and (column.find('[OI]') == -1):
            f_strongest.append(tab_m_PM_anchor[column][0])
            name_strongest.append(column)
    f_strongest = np.array(f_strongest)
    name_strongest = np.array(name_strongest)
    arg_strongest = np.nanargmax(f_strongest)
    label_strongest_line = name_strongest[arg_strongest]
    label_strongest_line = label_strongest_line.replace('flux_', '').replace('_pm', '')
    print(f'# Strongest PM line label: {label_strongest_line}')

    tab_el_strong_dynamic = select_tab_el_strong(
        tab_el, tab_m_PM_anchor,
        sn_min=float(getattr(args, "pm_strong_sn_min", 10.0)),
        max_lines=float(getattr(args, "pm_strong_max_lines", 10.0)),
        min_separation=float(getattr(args, "pm_strong_min_separation", 5.0)),
    )
    if len(tab_el_strong_dynamic):
        # This is the RSS FIT3D anchor selection.  The integrated fit above
        # deliberately used the original bootstrap clusters: tab_m_PM was not
        # available until that fit finished.  Rebuild for both RSS fitting paths.
        tab_el_strong = tab_el_strong_dynamic
        pm_strong_clusters = select_pm_strong_clusters(
            tab_el, tab_el_strong,
            max_cluster_span=float(getattr(args, "max_cluster_span", 300.0)),
            max_adjacent_distance=float(getattr(args, "max_adjacent_distance", 100.0)),
        )
        print(f'# RSS PM dynamically selected {len(tab_el_strong_dynamic)} / '
              f'{len(tab_el)} strong lines; FIT3D cluster members: '
              f'{sum(len(c) for c in pm_strong_clusters)}')
    else:
        # Preserve the existing predefined FIT3D strong-line clusters when
        # nothing meets the requested S/N and separation selection criteria.
        print(f'# RSS PM: no dynamic strong lines satisfy selection; using '
              f'{sum(len(c) for c in pm_strong_clusters)} predefined '
              f'FIT3D cluster members')
    # ------------------------------------------------------------
    # PEK fit of the integrated spectrum, using its own Halpha PM kinematics.
    # The independent RSS PEK algorithm and all fitted PM/NP parameters remain
    # unchanged; SPS is deliberately not modified (used later as SPS_master).
    # ------------------------------------------------------------
    tab_m_PEK = None
    m_pek_model = None
    if getattr(args, 'fit_PEK', True):
        ha_vel = float(tab_m_PM[f'vel_{label_strongest_line}_pm'][0])
        ha_disp = float(tab_m_PM[f'disp_{label_strongest_line}_pm'][0])
        if not (np.isfinite(ha_vel) and np.isfinite(ha_disp) and ha_disp > 0):
            print('# Integrated PEK: invalid Halpha velocity/dispersion; retaining PM model')
        else:
            bad_m_pek = (
                np.all(bad_pixels, axis=0)
                | ~np.isfinite(int_flux_cw)
                | ~np.isfinite(int_eflux)
                | (int_eflux <= 0)
            )
            tab_m_PEK, m_pek_model = fit_pek_rss(
                gas_spectrum=np.asarray(SPS.spectra['res_ssp'], dtype=float),
                error_spectrum=int_eflux,
                vel_interp=ha_vel,
                disp_interp=ha_disp,
                tab_el=tab_el,
                tab_PM=tab_m_PM,
                tab_fe=tab_m_fe,
                reference_line=label_strongest_line,
                wl=wl__w,
                lsf_spectrum=lsf_mean,  # same wavelength-dependent LSF as integrated PM
                bad_pixel_mask=bad_m_pek,
                velocity_tolerance=float(getattr(args, 'pek_velocity_tolerance', 5.0)),
                dispersion_tolerance=float(getattr(args, 'pek_dispersion_tolerance', 1.0)),
                fit_padding=float(getattr(args, 'pek_fit_padding', 20.0)),
                max_cluster_span=float(getattr(args, 'pek_max_cluster_span', 75.0)),
                max_adjacent_distance=float(getattr(args, 'pek_max_adjacent_distance', 25.0)),
                verbose=args.verbose,
            )
            add_emission_fit_quality(
                tab_m_PEK, wl__w, SPS.spectra['res_ssp'], m_pek_model,
                int_eflux, bad_pixel_mask=bad_m_pek, suffix='_pek')
            # With no successful cluster, the PEK routine returns a zero model.
            # Do not silently erase the original integrated emission model.
            valid_pek_flux = any(
                np.isfinite(tab_m_PEK[column][0])
                for column in tab_m_PEK.colnames
                if column.startswith('flux_') and column.endswith('_pek')
            )
            if not valid_pek_flux:
                print('# Integrated PEK: no fitted lines; retaining PM model')
                tab_m_PEK = None
                m_pek_model = None
            else:
                # Same convention as RSS: 7=PEK gas, 2=updated joint,
                # 4=updated residual, 8=explicit PEK joint spectrum.
                m_planes = np.asarray(hdu_m_sps[0].data, dtype=float).copy()
                if (m_planes.ndim != 2 or m_planes.shape[0] < 8 or
                        m_planes.shape[1] != len(wl__w)):
                    raise ValueError(f'Unexpected integrated MODEL shape: {m_planes.shape}')
                # SPS.output_fits_hdu currently has 8 planes (0..7).
                # Append plane 8 for the PEK joint spectrum; preserve 0..7.
                if m_planes.shape[0] == 8:
                    m_planes = np.concatenate((m_planes, m_planes[2:3, :].copy()), axis=0)
                old_m_gas = m_planes[7, :].copy()
                m_planes[7, :] = m_pek_model
                m_planes[2, :] = m_planes[2, :] - old_m_gas + m_pek_model
                m_planes[4, :] = m_planes[0, :] - m_planes[2, :]
                m_planes[8, :] = m_planes[2, :]
                for plane in (2, 4, 7, 8):
                    m_planes[plane, bad_m_pek] = np.nan
                hdu_m_sps[0].data = m_planes
                hdu_m_sps[0].header['NAME7'] = 'gas_model_PEK'
                hdu_m_sps[0].header['NAME8'] = 'mod_joint_spec_PEK'


    # Select a separate best line for each integrated-spectrum measurement,
    # using the original NP/PM/PEK per-line quality statistics.
    # The kinematics and individual method tables are left unchanged.
    add_gas_sigma_kms(tab_m_fe, wl__w, lsf_mean, suffix='')
    add_gas_sigma_kms(tab_m_PM, wl__w, lsf_mean, suffix='_pm')
    if tab_m_PEK is not None:
        add_gas_sigma_kms(tab_m_PEK, wl__w, lsf_mean, suffix='_pek')
    tab_m_best = select_best_emission_lines(tab_el, tab_m_fe, tab_m_PM, tab_m_PEK)
    m_best_gas = create_best_emission_model(wl__w, tab_m_best)
    bad_m_best = (np.all(bad_pixels, axis=0) |
                  ~np.isfinite(int_flux_cw) | ~np.isfinite(int_eflux) |
                  (int_eflux <= 0))
    # The integrated spectrum is 1D while the BEST model builder returns
    # (n_spectra, n_wave).  Select its only row for this single-spectrum QA;
    # the RSS path below correctly passes the full 2D BEST model.
    add_best_global_quality(
        tab_m_best, wl__w, np.asarray(hdu_m_sps[0].data[0]) -
        np.asarray(hdu_m_sps[0].data[1]), m_best_gas[0],
        int_eflux, bad_m_best)
    hdu_m_sps[0].data = adopt_best_model_planes(hdu_m_sps[0].data, m_best_gas)
    for plane in (2, 4, 5, 8):
        hdu_m_sps[0].data[plane, bad_m_best] = np.nan
    hdu_m_sps[0].header['NAME8'] = 'gas_model_best'


    #####################################################################
    # Test plot
    ##################################################################### 
    if (args.verbose == True):
        import matplotlib
        matplotlib.use('TkAgg')  # Must be set BEFORE importing pyplot
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(wl__w, SPS.spectra['orig_flux_ratio'], color='black', lw=2.0, label='Integrated Spectrum')
        #ax.plot(wl__w, out_decompose_flux_rss['st_flux_cw'], color='yellow', lw=1.5,  alpha = 0.9, label='Decomposed Stellar')
        ax.plot(wl__w, hdu_m_sps[0].data[2], color='tab:blue', lw=1.5,  alpha = 0.9, label='RSP + PEK Model' if tab_m_PEK is not None else 'RSP + Emission Model')
        ax.plot(wl__w, SPS.spectra['model_ssp_min'], color='tab:red', lw=1.5,  alpha = 0.9, label='RSP Model (Smoothed)')
        ax.plot(wl__w, hdu_m_sps[0].data[4], color='tab:orange', lw=1.5,  alpha = 0.9, label='Residual Spectrum')
        ax.plot(wl__w, int_eflux, color='grey', lw=3,  alpha = 0.75, label='Error Spectrum')
        ax.set_xlabel('Wavelength (Angstrom)')
        ax.set_ylabel('Flux')
        ax.set_title('Integrated Spectrum Decomposition')
        ax.legend()
        plt.tight_layout()
        fig.savefig(f'{args.output_path}/output_m.{args.label}.decompose.{out_plot_format}', dpi=300)
        plt.show(block=True)  # Wait until the window is manually closed
        plt.close(fig)

#    exit(0)
    ####################################################################
    # Preliminar version of the output plot
    if (args.do_plots==1):
        y_off = -1.09
        y_off2 = -2.18
        plot_spec_art_hdu(dir='',hdu=hdu_m_sps,\
                      name=args.label,text=args.label,output=f'{args.output_path}/output_m.{args.label}.{out_plot_format}',\
                      c_map='CMRmap',\
                      insets=((0.00, y_off, 0.09, 0.9,3715,3739,-1,35,'[OII]'),\
                              (0.107, y_off, 0.09, 0.9,4088,4118,-0.5,10,r'H$\delta$'),\
                              (0.213, y_off, 0.09, 0.9,4324,4357,-0.5,18,r'H$\gamma$'),\
                              (0.320, y_off, 0.09, 0.9,4851,4872,-1.0,38,r'H$\beta$'),\
                              (0.425, y_off, 0.09, 0.9,4998,5015,-1.0,50,'[OIII]'),\
                              (0.530, y_off, 0.155, 0.9,6543,6590,-2,220,r'H$\alpha$+[NII]'),\
                              (0.700, y_off, 0.09, 0.9,6709,6737,-0.5,20,'[SII]'),\
                              (0.807, y_off, 0.09, 0.9,9062,9076,-1.0,28,'[SIII]'),\
                              (0.910, y_off, 0.09, 0.9,9526,9537,-1.5,82,'[SIII]'),\
                              (0.00, y_off2, 0.22, 0.9,3801,4099,-0.5,3.2,'3800-4100'),\
                              (0.23, y_off2, 0.22, 0.9,4801,5099,-0.2,2.8,'4800-5100'),\
                              (0.46, y_off2, 0.10, 0.9,4363-8.1,4363+7.9,-0.25,0.8,'[OIII]'),\
                              (0.57, y_off2, 0.10, 0.9,4472-8.1,4472+7.9,-0.25,0.8,'HeI'),\
                              (0.68, y_off2, 0.10, 0.9,7320-8,7320+8,-0.2,1.7,'[OII]'),\
                              (0.79, y_off2, 0.10, 0.9,5755-6,5755+6,-0.25,2.5,'[NII]'),\
                              (0.90, y_off2, 0.10, 0.9,6312-6,6312+6,-0.25,2.5,'[SIII]')
                              ),
                      y_min=-0.5,y_max=30,y0_d=0.3,y1_d=2.9,\
                      x_min=3600,x_max=9600,plot_el=True, tab_el=tab_el,plot_res=True,show_scale=False,n_ord=2,gamma=0.5)#,x_min=3600,x_max=9600)



    tab_m_fe_B, tab_m_fe_R, tab_m_fe_I = split_fe_table(tab_m_fe, w_br=FIT_QUALITY_W_BR, w_ri=FIT_QUALITY_W_RI)
    tab_m_PM_B, tab_m_PM_R, tab_m_PM_I = split_fe_table(tab_m_PM, w_br=FIT_QUALITY_W_BR, w_ri=FIT_QUALITY_W_RI)
    hdr_0['SPLITBR'] = (FIT_QUALITY_W_BR, 'B/R table wavelength boundary [A]')
    hdr_0['SPLITRI'] = (FIT_QUALITY_W_RI, 'R/I table wavelength boundary [A]')
    hdr_0['dap_ver']=dap_ver
    hdr_0['dap_date']=dap_date   
    hdu_m_hdr_0 = fits.PrimaryHDU(header=hdr_0) 
    hdu_m_FE_B = fits.BinTableHDU(tab_m_fe_B,name='NP_ELINES_B')
    hdu_m_FE_R = fits.BinTableHDU(tab_m_fe_R,name='NP_ELINES_R')
    hdu_m_FE_I = fits.BinTableHDU(tab_m_fe_I,name='NP_ELINES_I')
    hdu_m_PM_B = fits.BinTableHDU(tab_m_PM_B,name='PM_ELINES_B')
    hdu_m_PM_R = fits.BinTableHDU(tab_m_PM_R,name='PM_ELINES_R')
    hdu_m_PM_I = fits.BinTableHDU(tab_m_PM_I,name='PM_ELINES_I')
    tab_m_best_B, tab_m_best_R, tab_m_best_I = split_fe_table(
        tab_m_best, w_br=FIT_QUALITY_W_BR, w_ri=FIT_QUALITY_W_RI)
    hdu_m_RSP = fits.BinTableHDU(tab_m_rsp,name='RSP')
    hdu_m_COEFFS = fits.BinTableHDU(tab_m_coeffs,name='COEFFS')
    hdu_m_dap =fits.HDUList([hdu_m_hdr_0,hdu_m_RSP,hdu_m_COEFFS,\
                           hdu_m_FE_B,hdu_m_FE_R,hdu_m_FE_I,hdu_m_PM_B,hdu_m_PM_R,hdu_m_PM_I,\
                            hdu_info])
    for band, table in (('B', tab_m_best_B), ('R', tab_m_best_R),
                        ('I', tab_m_best_I)):
        hdu_m_dap.append(fits.BinTableHDU(table, name=f'BEST_ELINES_{band}'))
    if tab_m_PEK is not None:
        tab_m_PEK_B, tab_m_PEK_R, tab_m_PEK_I = split_fe_table(
            tab_m_PEK, w_br=FIT_QUALITY_W_BR, w_ri=FIT_QUALITY_W_RI)
        for band, table in (('B', tab_m_PEK_B), ('R', tab_m_PEK_R), ('I', tab_m_PEK_I)):
            hdu_m_dap.append(fits.BinTableHDU(table, name=f'PEK_ELINES_{band}'))
    out_file_m_dap = os.path.join(args.output_path, f"m_{args.label}.dap.fits.gz")
    hdu_m_dap.writeto(out_file_m_dap,overwrite=True)

    if args.dump_model == True:
        out_file_m_model = os.path.join(args.output_path, f"m_{args.label}.model.fits.gz")
        fits.PrimaryHDU(
            data=hdu_m_sps[0].data,
            header=hdu_m_sps[0].header,
        ).writeto(out_file_m_model, overwrite=True)

    if (only_integrated == True):
      #
      # ***** NEED TO ADD WRITTING RESULTS FOR INTEGRATED SPECTRUM!!!!!
      #
      print("# Only mean spectrum analyzed: END ALL")
      quit()
    

    print("##############################################")
    print("# End fitting the integrated spectrum ########")
    print("##############################################")
    print("\n")


    _mem.report('after_integrated_fit', arrays={'rss_flux': rss_flux, 'rss_eflux': rss_eflux, 'SPS_model': hdu_m_sps[0].data}, sps=SPS)
    print(f'### START analyzing the RSS for the strong stellar component ###')

    #
    #  We find the spectra with strong stellar component to be fitted with RSPs
    #     
    w1 = args.w_range_nl[0]*(1+args.redshift[0])
    w2 = args.w_range_nl[1]*(1+args.redshift[0])
    i1 = np.argmin(np.abs(wl__w-w1))
    i2 = np.argmin(np.abs(wl__w-w2))
    # Ignore repaired pixels when selecting reliable stellar fits.
    flux_c = np.nanmean(np.where(~bad_pixels[:, i1:i2],
                                 rss_flux[:, i1:i2], np.nan), axis=-1)
    flux_e = np.nanmean(np.where(~bad_pixels[:, i1:i2],
                                 rss_eflux[:, i1:i2], np.nan), axis=-1)
    mask_st = (flux_c>args.SN_CUT*flux_e) & (flux_c>0)
    len_mask_st = np.sum(mask_st)
    print(f'# Number of spectra with strong stellar component: {len_mask_st} / {len(rss_flux)}')
    print(f'### END analyzing the RSS for the strong stellar component ###')

    if (len_mask_st==0):
        print("# No spectra with strong stellar component: END ALL")
        tab_rsp_interp = TableAccumulator(batch_size=64)
        tab_coeffs_interp = TableAccumulator(batch_size=64)
        for (i,id_fib) in enumerate(tab_PT['id']):
            tab_m_rsp_now = tab_m_rsp.copy()
            tab_m_coeffs_now = tab_m_coeffs.copy()
            tab_m_rsp_now['id'] = np.repeat(id_fib, len(tab_m_rsp))
            tab_m_coeffs_now['id'] = np.repeat(id_fib, len(tab_m_coeffs))
            tab_rsp_interp.append(tab_m_rsp_now)
            tab_coeffs_interp.append(tab_m_coeffs_now)

        tab_rsp_interp = tab_rsp_interp.finish()
        tab_coeffs_interp = tab_coeffs_interp.finish()
        ############################################################
        # We will assume that all spaxels have as similar stellar component as the integrated spectrum
        ############################################################
    else:
        print("##########################################################################")
        print("# Start fitting the RSS spectra of bright stellar components with RSPs ##")
        print("##########################################################################")
        rss_flux_st = rss_flux[mask_st]
        rss_eflux_st = rss_eflux[mask_st]
        lsf_rss_st = LSF_rss[mask_st]
        tab_PT_st = tab_PT[mask_st]
        z_org = args.redshift[0]
        z_min = args.redshift[1]
        z_max = args.redshift[2]
        tab_rsp_st = TableAccumulator(batch_size=64)
        tab_coeffs_st = TableAccumulator(batch_size=64)
        tab_elines_st = TableAccumulator(batch_size=64)
        tab_PM_st = TableAccumulator(batch_size=64)
        tab_fe_st = TableAccumulator(batch_size=64)
        nspec = rss_flux_st.shape[0]
        for i, (f__w, ef__w) in enumerate(zip(rss_flux_st, rss_eflux_st)):
            if (args.verbose==True):
                print(f"\n### Fitting spaxel {i+1}/{nspec} (ID: {tab_PT_st['id'][i]})")
            else:
                print_progress_bar(i, nspec, label="St. Fit. progress", bar_width=40)
            tab_i_rsp, tab_i_coeffs, \
                tab_i_elines, tab_i_PM, \
                tab_i_fe, hdu_i_sps, SPS = fit_individual_rss_spectrum_or_adopt_result(
                wl=wl__w,
                flux=f__w,
                eflux=ef__w,
                # DRP per-fiber LSF where available; fall back to mean LSF
                # for generic RSS/simulator inputs without an LSF extension.
                lsf=np.where(np.isfinite(lsf_rss_st[i]) & (lsf_rss_st[i] > 0),
                             lsf_rss_st[i], lsf_mean),
                tab_PT=tab_PT_st,
                i_fib=i,
                tab_el=tab_el,
                el_clusters=el_clusters,
                hdr_flux=hdr_flux,
                args_copy=args,
                z_guess=args.redshift[0],
                plot=args.plot,
                adopt_st_z = True,
                SN_cut_st_z = args.SN_CUT, 
                SPS_master=SPS,
                verbose=args.verbose,
                fit_NP=False,
                fit_PM=True,
                bad_pixel_mask=bad_pixels[mask_st][i],
                empirical_errors = empirical_errors,
                pm_strong_clusters=pm_strong_clusters,
            )
            tab_rsp_st.append(tab_i_rsp)
            tab_coeffs_st.append(tab_i_coeffs)
            tab_elines_st.append(tab_i_elines)
            tab_PM_st.append(tab_i_PM)
            tab_fe_st.append(tab_i_fe)

        tab_rsp_st = tab_rsp_st.finish()
        tab_coeffs_st = tab_coeffs_st.finish()
        tab_elines_st = tab_elines_st.finish()
        tab_fe_st = tab_fe_st.finish()
        tab_PM_st = tab_PM_st.finish()
        ####################################################################################
        # Now we propagate the velocity map for the remaining spaxels with low stellar S/N
        ####################################################################################
        
        #############################################################################
        # We create the interpolated RSP and coefficients tables for the remaining spaxels
        #############################################################################        
        tab_rsp_interp, tab_coeffs_interp = interpolate_rsp_coeffs(
            tab_PT_st=tab_PT_st,
            tab_rsp_st=tab_rsp_st,
            tab_coeffs_st=tab_coeffs_st,
            tab_PT=tab_PT,
            method="spline",
            smoothing=0,
            decimals=8,
             min_spline_points=10,
            k=4,
            power=2.0,
            verbose=args.verbose
        )


        tab_rsp_interp, tab_coeffs_interp = preserve_rsp_coeffs_samples(
            tab_PT_st,
            tab_rsp_st,
            tab_coeffs_st,
            tab_PT,
            tab_rsp_interp,
            tab_coeffs_interp
        )

        #####################################################
        # Comparison plots if required
        #####################################################

        if (args.do_plots==1):
            vmin = args.redshift[1]*__c__
            vmax = args.redshift[2]*__c__
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2, figsize=(12.5, 6))
            ax[0].scatter(tab_PT_st['ra'], tab_PT_st['dec'], c = tab_rsp_st['z']*__c__, \
                s= 150, cmap='Spectral', alpha = 0.5 , edgecolor='none')
            ax[1].scatter(tab_PT['ra'], tab_PT['dec'], c = tab_rsp_interp['z']*__c__, \
                s= 150, cmap='Spectral', alpha = 0.5 , edgecolor='none')
            ax[0].scatter(tab_PT['ra'], tab_PT['dec'], color='None', alpha = 0.5 , s = 150, edgecolor='Black')
            ax[1].scatter(tab_PT['ra'], tab_PT['dec'], color='None', alpha = 0.5 , s = 150,edgecolor='Black')
            fig.savefig(f'vel_map_interp_{args.label}.png', dpi=300)
            plt.close(fig)
        #plt.show()

        print("##########################################################################")
        print("# End fitting the RSS spectra with bright stellar components with RSPs ##")
        print("##########################################################################")


    if (args.verbose==True):
        print("##############################################")
        print("# Interpolated RSP and coefficients tables ###")
        print("##############################################")
        print(tab_rsp_interp.colnames)
        print(tab_coeffs_interp.colnames)
        print(tab_rsp_interp[0:5])
        print(tab_coeffs_interp[0:108])


    ##############################################################################
    # START: Fitting the full RSS spectra using the interpolated RSPs and coefficients
    ##############################################################################

    _mem.report('before_full_rss', arrays={'rss_flux': rss_flux, 'rss_eflux': rss_eflux, 'LSF_rss': LSF_rss, 'bad_pixels': bad_pixels, 'tab_rsp_interp': tab_rsp_interp, 'tab_coeffs_interp': tab_coeffs_interp}, sps=SPS)
    print("##############################################")
    print("# Start fitting full RSS spectra with interpolated RSPs and coefficients ##")
    print("##############################################")

    tab_rsp = TableAccumulator(batch_size=64)
    tab_coeffs = TableAccumulator(batch_size=64)
    tab_elines = TableAccumulator(batch_size=64)
    tab_PM = TableAccumulator(batch_size=64)
    tab_fe = TableAccumulator(batch_size=64)
    output_model = []
    gas_spectra_for_pek = []
    nspec = len(rss_flux)
    for i, (f__w, ef__w) in enumerate(zip(rss_flux, rss_eflux)):
        if _mem.enabled() and (i % _mem.every() == 0 or i+1 == nspec):
            _mem.report(f'before_rss_fiber_{i+1}_of_{nspec}',
                lists={'output_model': output_model,
                       'gas_spectra_for_pek': gas_spectra_for_pek,
                       'tab_rsp': tab_rsp, 'tab_coeffs': tab_coeffs,
                       'tab_elines': tab_elines, 'tab_PM': tab_PM, 'tab_fe': tab_fe},
                sps=SPS)
        if (args.verbose==True):
            print(f"\n### EL Fitting spaxel {i+1}/{nspec} (ID: {tab_PT['id'][i]})")
        else:
            print_progress_bar(i, nspec, label="EL fit progress", bar_width=40)

        tab_i_rsp, tab_i_coeffs, \
        tab_i_elines, tab_i_PM, \
        tab_i_fe, hdu_i_sps, SPS = fit_individual_rss_spectrum_or_adopt_result(
            wl=wl__w,
            flux=f__w,
            eflux=ef__w,
            lsf=np.where(np.isfinite(LSF_rss[i]) & (LSF_rss[i] > 0),
                         LSF_rss[i], lsf_mean),
            tab_PT=tab_PT,
            i_fib=i,
            tab_el=tab_el,
            el_clusters=el_clusters,
            hdr_flux=hdr_flux,
            args_copy=args,
            z_guess=args.redshift[0],
            plot=args.plot,
            adopt_st_z = True,
            SN_cut_st_z = args.SN_CUT, 
            SPS_master=SPS,
            verbose=args.verbose,
            fit_NP=True,
            fit_PM=True,
            tab_rsp_adopt=tab_rsp_interp,
            tab_coeffs_adopt=tab_coeffs_interp,
            bad_pixel_mask=bad_pixels[i],
            empirical_errors = empirical_errors,
            pm_strong_clusters=pm_strong_clusters,
        )
        if _mem.enabled() and (i % _mem.every() == 0 or i+1 == nspec):
            _mem.report(f'after_fit_before_retain_{i+1}_of_{nspec}',
                arrays={'last_model_hdu': hdu_i_sps[0].data},
                lists={'output_model': output_model,
                       'gas_spectra_for_pek': gas_spectra_for_pek,
                       'tab_rsp': tab_rsp, 'tab_PM': tab_PM, 'tab_fe': tab_fe},
                sps=SPS)
        tab_rsp.append(tab_i_rsp)
        tab_coeffs.append(tab_i_coeffs)
        tab_elines.append(tab_i_elines)
        tab_PM.append(tab_i_PM)
        tab_fe.append(tab_i_fe)
        # Keep FIT3D/RSP/PEK calculations in their original precision;
        # store each completed MODEL fiber as float32 to halve cube RAM.
        output_model.append(np.asarray(hdu_i_sps[0].data, dtype=np.float32))
        if getattr(args, 'fit_PEK', True):
            # Fit PEK to the observed flux minus the FINAL stellar model.
            # The stored 'res_ssp' is stale after continuum correction.
            gas_spectra_for_pek.append(
                (np.asarray(f__w, float) -
                 np.asarray(SPS.spectra['model_ssp_min'], float)).copy()
            )
        if _mem.enabled() and (i % _mem.every() == 0 or i+1 == nspec):
            _mem.report(f'rss_fiber_{i+1}_of_{nspec}',
                arrays={'last_model_hdu': hdu_i_sps[0].data, 'fitted_flux': f__w,
                        'fitted_eflux': ef__w, 'lsf_row': LSF_rss[i]},
                lists={'output_model': output_model,
                       'gas_spectra_for_pek': gas_spectra_for_pek,
                       'tab_rsp': tab_rsp, 'tab_coeffs': tab_coeffs,
                       'tab_elines': tab_elines, 'tab_PM': tab_PM, 'tab_fe': tab_fe},
                sps=SPS)


    _mem.report('after_rss_fits_before_stack', arrays={'rss_flux': rss_flux}, lists={'output_model': output_model, 'gas_spectra_for_pek': gas_spectra_for_pek, 'tab_rsp': tab_rsp, 'tab_coeffs': tab_coeffs, 'tab_elines': tab_elines, 'tab_PM': tab_PM, 'tab_fe': tab_fe}, sps=SPS)
    tab_rsp = tab_rsp.finish()
    tab_coeffs = tab_coeffs.finish()
    tab_elines = tab_elines.finish()
    tab_fe = tab_fe.finish()
    tab_PM = tab_PM.finish()

    ########################################################################
    # START: Interpolating the velocity map of the emission lines for the spaxels with low S/N
    ########################################################################





    vel_Ha = tab_PM[f'vel_{label_strongest_line}_pm']
    disp_Ha = tab_PM[f'disp_{label_strongest_line}_pm']
    SN_Ha = tab_PM[f'flux_{label_strongest_line}_pm']/tab_PM[f'e_flux_{label_strongest_line}_pm']
    vel_mask_Ha = (np.asarray(SN_Ha > args.SN_CUT, bool)
                   & np.isfinite(vel_Ha) & np.isfinite(disp_Ha)
                   & (np.asarray(disp_Ha) > 0)
                   & np.isfinite(tab_PT['ra']) & np.isfinite(tab_PT['dec']))
    if np.sum(vel_mask_Ha) >= 4:
        # Preserve the spline map introduced in dap(8).py.
        try:
            vel_map_Ha_interp = interp2d_scattered(
                tab_PT['ra'][vel_mask_Ha], tab_PT['dec'][vel_mask_Ha],
                vel_Ha[vel_mask_Ha], tab_PT['ra'], tab_PT['dec'],
                method='spline', smoothing=0)
            disp_map_Ha_interp = interp2d_scattered(
                tab_PT['ra'][vel_mask_Ha], tab_PT['dec'][vel_mask_Ha],
                disp_Ha[vel_mask_Ha], tab_PT['ra'], tab_PT['dec'],
                method='spline', smoothing=0)
        except (ValueError, RuntimeError):
            vel_map_Ha_interp, disp_map_Ha_interp = interpolate_ha_kinematics(
                tab_PT, tab_PM, sn_cut=float(args.SN_CUT),
                line_key=label_strongest_line)
    elif np.any(vel_mask_Ha):
        # The spline requires sufficient anchors; use linear/nearest interpolation.
        vel_map_Ha_interp, disp_map_Ha_interp = interpolate_ha_kinematics(
            tab_PT, tab_PM, sn_cut=float(args.SN_CUT),
                line_key=label_strongest_line)
    else:
        vel_map_Ha_interp = vel_Ha.copy()
        disp_map_Ha_interp = disp_Ha.copy()
    # A spline may extrapolate invalid or nonphysical values: replace only those.
    vel_map_Ha_interp = np.asarray(vel_map_Ha_interp, dtype=float)
    disp_map_Ha_interp = np.asarray(disp_map_Ha_interp, dtype=float)
    if (np.any(~np.isfinite(vel_map_Ha_interp)) or
            np.any(~np.isfinite(disp_map_Ha_interp) | (disp_map_Ha_interp <= 0))):
        v_fallback, s_fallback = interpolate_ha_kinematics(
            tab_PT, tab_PM, sn_cut=float(args.SN_CUT),
                line_key=label_strongest_line)
        bad_v = ~np.isfinite(vel_map_Ha_interp)
        bad_s = ~np.isfinite(disp_map_Ha_interp) | (disp_map_Ha_interp <= 0)
        vel_map_Ha_interp[bad_v] = v_fallback[bad_v]
        disp_map_Ha_interp[bad_s] = s_fallback[bad_s]

    




    _mem.report('after_rss_table_stack', arrays={'tab_rsp': tab_rsp, 'tab_coeffs': tab_coeffs, 'tab_elines': tab_elines, 'tab_PM': tab_PM, 'tab_fe': tab_fe}, lists={'output_model': output_model, 'gas_spectra_for_pek': gas_spectra_for_pek}, sps=SPS)
    output_model = np.asarray(output_model)
    output_model = np.swapaxes(output_model, 1, 0)  # (n_plane, n_fiber, n_wave)
    # Apply the ORIGINAL mask to every plane only AFTER all computations.
    if output_model.ndim != 3 or output_model.shape[1:] != bad_pixels.shape:
        raise ValueError(f'MODEL shape {output_model.shape} is inconsistent '
                         f'with MASK shape {bad_pixels.shape}')
    output_model[:, bad_pixels] = np.nan

#    print(tab_rsp_interp.colnames)
#    print(tab_coeffs_interp.colnames)
#    print(tab_rsp_interp[0:5])
#    print(tab_coeffs_interp[0:108])
#    print(tab_elines.colnames)
#    print(tab_elines[0:20])
#    print(tab_fe.colnames)
#    print(tab_fe[0:20])
#    print(tab_PM.colnames)
#    print(tab_PM[0:20])



    _mem.report('after_model_cube', arrays={'output_model': output_model, 'rss_flux': rss_flux, 'rss_eflux': rss_eflux}, lists={'gas_spectra_for_pek': gas_spectra_for_pek})
    # PEK: constrained Gaussian + constant-background fits, independent of pyFIT3D.
    # Reuse the Halpha velocity/dispersion maps computed immediately above.
    tab_PEK = None
    pek_model = None
    if getattr(args, 'fit_PEK', True):
        tab_PEK, pek_model = fit_pek_rss(
            gas_spectrum=np.asarray(gas_spectra_for_pek),
            error_spectrum=rss_eflux,
            vel_interp=vel_map_Ha_interp,
            disp_interp=disp_map_Ha_interp,
            tab_el=tab_el,
            tab_PM=tab_PM,
            tab_fe=tab_fe,
            reference_line=label_strongest_line,
            wl=wl__w,
            lsf_spectrum=LSF_rss,       # per-fiber Gaussian sigma [Angstrom]
            bad_pixel_mask=bad_pixels,
            velocity_tolerance=float(getattr(args, 'pek_velocity_tolerance', 5.0)),
            dispersion_tolerance=float(getattr(args, 'pek_dispersion_tolerance', 1.0)),
            fit_padding=float(getattr(args, 'pek_fit_padding', 20.0)),
            max_cluster_span=float(getattr(args, 'pek_max_cluster_span', 75.0)),
            max_adjacent_distance=float(getattr(args, 'pek_max_adjacent_distance', 25.0)),
            verbose=args.verbose,
        )
        add_emission_fit_quality(
            tab_PEK, wl__w, np.asarray(gas_spectra_for_pek), pek_model,
            rss_eflux, bad_pixel_mask=bad_pixels, suffix='_pek')
        add_gas_sigma_kms(tab_PEK, wl__w, LSF_rss, suffix='_pek')
        tab_PEK_B, tab_PEK_R, tab_PEK_I = split_fe_table(
            tab_PEK, w_br=FIT_QUALITY_W_BR, w_ri=FIT_QUALITY_W_RI)

    # PEK has finished both fitting and computing per-line quality metrics.
    # No later stage needs these full-resolution gas residual spectra.
    del gas_spectra_for_pek
    _mem.report('after_pek_input_release', arrays={'pek_model': pek_model})

    #################################################################################
    # Writting DAP file
    #################################################################################
    if (args.verbose==True):
        print("##############################################")
        print("# Writing output DAP file           ##########")
        print("##############################################")


    # Restored from the pre-2.0 legacy outputs: LSF-deconvolved velocity
    # dispersions, stored per line in the existing wide NP/PM FITS tables.
    # Unlike the old block below exit(0), this uses each fiber's actual LSF
    # at the observed wavelength and does not fabricate resolved dispersions.
    add_gas_sigma_kms(tab_fe, wl__w, LSF_rss, suffix='')
    add_gas_sigma_kms(tab_PM, wl__w, LSF_rss, suffix='_pm')

    # Each line/fiber adopts the lowest valid local reduced chi-square from
    # NP, PM and (if fitted) PEK, copying all quantities and formal errors
    # from that method. Rebuild the Gaussian composite from selected results.
    tab_best = select_best_emission_lines(tab_el, tab_fe, tab_PM, tab_PEK)
    gas_model_best = create_best_emission_model(wl__w, tab_best)
    add_best_global_quality(
        tab_best, wl__w, np.asarray(rss_flux, float) - output_model[1],
        gas_model_best, rss_eflux, bad_pixels)
    tab_best_B, tab_best_R, tab_best_I = split_fe_table(
        tab_best, w_br=FIT_QUALITY_W_BR, w_ri=FIT_QUALITY_W_RI)
    print(f'# BEST selected lines per method: ' + ', '.join(
        f'{method}={sum(np.count_nonzero(np.asarray(tab_best[c]) == method) for c in tab_best.colnames if c.startswith("best_method_"))}'
        for method in ('NP', 'PM', 'PEK')))

    tab_fe_B, tab_fe_R, tab_fe_I = split_fe_table(tab_fe, w_br=FIT_QUALITY_W_BR, w_ri=FIT_QUALITY_W_RI)
    tab_PM_B, tab_PM_R, tab_PM_I = split_fe_table(tab_PM, w_br=FIT_QUALITY_W_BR, w_ri=FIT_QUALITY_W_RI)
    hdr_0['SPLITBR'] = (FIT_QUALITY_W_BR, 'B/R table wavelength boundary [A]')
    hdr_0['SPLITRI'] = (FIT_QUALITY_W_RI, 'R/I table wavelength boundary [A]')
    hdr_0['dap_ver']=dap_ver
    hdr_0['dap_date']=dap_date   
    hdu_hdr_0 = fits.PrimaryHDU(header=hdr_0) 
    # Averages of all scalar measurements with an associated error column.
    # COEFFS is split by template (rsp); source tables are not modified.
    tab_avg_param_stats = build_average_measurement_statistics_table(
        tab_PT,
        {'PT': tab_PT, 'RSP': tab_rsp, 'COEFFS': tab_coeffs,
         'NP': tab_fe, 'PM': tab_PM, 'PEK': tab_PEK, 'BEST': tab_best},
        get_input_expnum(hdr_0, hdr_flux_org, tab_PT),
    )
    # Fit-quality columns have no paired e_* uncertainties, but should also
    # appear in both average-parameter products as fiber-averaged statistics.
    add_average_xsq_statistics(
        tab_avg_param_stats, {'NP': tab_fe, 'PM': tab_PM, 'PEK': tab_PEK, 'BEST': tab_best})
    hdu_avg_param_stats = fits.BinTableHDU(
        tab_avg_param_stats, name='AVG_PARAM_STATS')
    hdu_avg_param_stats.header['COMMENT'] = (
        'Fiber mean, sample std and combined mean error per measured parameter')
    # Compare the masked mean RSS gas spectrum to NP, PM and (when
    # available) PEK gas models. This QA is stored in every RSS DAP file,
    # even when plots and separate model output are disabled.
    tab_avg_el_fit_qa = build_average_emission_quality_table(
        wl__w, np.asarray(rss_flux, float) - output_model[1],
        rss_eflux, bad_pixels,
        {'np': output_model[6], 'pm': output_model[7],
         **({'pek': pek_model} if pek_model is not None else {}),
         'best': gas_model_best},
        tab_el, tab_m_PM)
    hdu_avg_el_fit_qa = fits.BinTableHDU(
        tab_avg_el_fit_qa, name='AVG_EL_FIT_QA')
    hdu_avg_el_fit_qa.header['COMMENT'] = (
        'Mean RSS gas-spectrum quality by line; NP, PM, PEK compared')
    if getattr(args, 'in_cube', False):
        # Same measured scalar columns as the normal DAP, now arranged as
        # WCS-registered [parameter, y, x] image cubes. The integrated and
        # averaged products remain in their existing formats.
        parameter_tables = {
            'PT_MAP': tab_PT,
            'RSP': tab_rsp,
            'COEFFS': tab_coeffs,
            'NP_ELINES_B': tab_fe_B,
            'NP_ELINES_R': tab_fe_R,
            'NP_ELINES_I': tab_fe_I,
            'PM_ELINES_B': tab_PM_B,
            'PM_ELINES_R': tab_PM_R,
            'PM_ELINES_I': tab_PM_I,
        }
        if tab_PEK is not None:
            parameter_tables.update({
                'PEK_ELINES_B': tab_PEK_B,
                'PEK_ELINES_R': tab_PEK_R,
                'PEK_ELINES_I': tab_PEK_I,
            })
        parameter_tables.update({
            'BEST_ELINES_B': tab_best_B,
            'BEST_ELINES_R': tab_best_R,
            'BEST_ELINES_I': tab_best_I,
        })
        out_file_dap = os.path.join(
            args.output_path, f'{args.label}.cube.dap.fits.gz')
        write_dap_parameter_cubes(
            out_file_dap, args.lvm_file, tab_PT, parameter_tables,
            primary_header=hdr_0, info=hdu_info,
            avg_param_stats=tab_avg_param_stats,
            avg_el_fit_qa=tab_avg_el_fit_qa,
            bad_pixel_mask=bad_pixels,
        )
        print(f'# Wrote WCS-registered parameter cubes: {out_file_dap}')
    else:
        hdu_PT = fits.BinTableHDU(tab_PT,name='PT')
        hdu_FE_B = fits.BinTableHDU(tab_fe_B,name='NP_ELINES_B')
        hdu_FE_R = fits.BinTableHDU(tab_fe_R,name='NP_ELINES_R')
        hdu_FE_I = fits.BinTableHDU(tab_fe_I,name='NP_ELINES_I')
        hdu_PM_B = fits.BinTableHDU(tab_PM_B,name='PM_ELINES_B')
        hdu_PM_R = fits.BinTableHDU(tab_PM_R,name='PM_ELINES_R')
        hdu_PM_I = fits.BinTableHDU(tab_PM_I,name='PM_ELINES_I')
        hdu_RSP = fits.BinTableHDU(tab_rsp,name='RSP')
        hdu_COEFFS = fits.BinTableHDU(tab_coeffs,name='COEFFS')
        hdu_mask = fits.ImageHDU(data=input_mask, name='MASK')
        hdu_mask.header['COMMENT'] = 'Original DRP bitmask; nonzero marks a flagged pixel'
        hdu_badpix = fits.ImageHDU(data=bad_pixels.astype(np.uint8), name='BADPIX')
        hdu_badpix.header['COMMENT'] = 'Effective bad-pixel mask; 1=excluded/filled'
        hdu_dap =fits.HDUList([hdu_hdr_0,hdu_PT,hdu_RSP,hdu_COEFFS,\
                               hdu_FE_B,hdu_FE_R,hdu_FE_I,hdu_PM_B,hdu_PM_R,hdu_PM_I])
        out_file_dap = os.path.join(args.output_path, f"{args.label}.dap.fits.gz")
        if tab_PEK is not None:
            for band, table in (('B', tab_PEK_B), ('R', tab_PEK_R), ('I', tab_PEK_I)):
                hdu_dap.append(fits.BinTableHDU(table, name=f'PEK_ELINES_{band}'))
        for band, table in (('B', tab_best_B), ('R', tab_best_R),
                            ('I', tab_best_I)):
            hdu_dap.append(fits.BinTableHDU(table, name=f'BEST_ELINES_{band}'))
        hdu_dap.append(hdu_avg_el_fit_qa)
        hdu_dap.append(hdu_avg_param_stats)
        hdu_dap.append(hdu_mask)
        hdu_dap.append(hdu_badpix)
        hdu_dap.append(hdu_info)
        hdu_dap.writeto(out_file_dap,overwrite=True)

    # A standalone, compact summary with the same primary header as the
    # full DAP file.  Both the per-fiber X_sq statistics (AVG_PARAM_STATS)
    # and the fit quality of the averaged gas spectrum (AVG_EL_FIT_QA)
    # are retained, including all global NP/PM/PEK X_sq measurements.
    out_file_avg_dap = os.path.join(
        args.output_path, f"avg_{args.label}.dap.fits.gz")
    fits.HDUList([
        fits.PrimaryHDU(header=hdu_hdr_0.header.copy()),
        fits.BinTableHDU(tab_avg_param_stats, name='AVG_PARAM_STATS'),
        fits.BinTableHDU(tab_avg_el_fit_qa, name='AVG_EL_FIT_QA'),
        hdu_info.copy(),
    ]).writeto(out_file_avg_dap, overwrite=True)

    # Restored final spatial maps. Keep independent of --dump_model and avoid
    # executing the legacy RSS analysis that remains below exit(0).
    if args.do_plots == 1 and not getattr(args, 'in_cube', False):
        try:
            tab_DAP = read_DAP_file(out_file_dap, verbose=False)
        except Exception as exc:
            print(f'# Unable to read DAP output for final maps: {exc}')
        else:
            for field in ('flux_Halpha_6562.85', 'med_flux_st'):
                if field not in tab_DAP.colnames:
                    print(f'# Final map skipped: missing {field}')
                    continue
                try:
                    map_plot_DAP(
                        tab_DAP, line=field, vmin=0, vmax=0, title=None,
                        filename=f'{args.label}_{field}', cmap='Spectral',
                        fsize=8, figs_dir=args.output_path,
                        fig_type=out_plot_format, gamma=0.3)
                except Exception as exc:
                    print(f'# Unable to plot {field}: {exc}')

            rgb_keys = (
                'flux_[NII]_6583.45_pek',
                'flux_Halpha_6562.85_pek',
                'flux_[OIII]_5006.84_pek',
            )
            if tab_PEK is None:
                print('# RGB [NII]/Halpha/[OIII] map skipped: PEK not enabled')
            elif not all(key in tab_DAP.colnames for key in rgb_keys):
                missing = [key for key in rgb_keys if key not in tab_DAP.colnames]
                print(f'# RGB PEK map skipped: missing {missing}')
            else:
                try:
                    map_plot_DAP_rgb(
                        tab_DAP,
                        rgb_key=rgb_keys,
                        rgb_scale=(1, 1, 6),
                        vmin=0, vmax=0.0,
                        titles=('[NII]', r'H$\alpha$', '[OIII]'),
                        filename=args.label,
                        fsize=6, figs_dir=args.output_path,
                        gamma=0.35, fig_type='png', fs=15)
                except Exception as exc:
                    print(f'# Unable to plot PEK RGB map: {exc}')

    _mem.report('after_pm_pek_best', arrays={'output_model': output_model, 'pek_model': pek_model, 'gas_model_best': gas_model_best, 'tab_PEK': tab_PEK, 'tab_best': tab_best})
    # Optional first/last RSS diagnostic spectra.  The average-RSS quality
    # comparison above is always stored in the AVG_EL_FIT_QA FITS extension.
    if args.do_plots == 1:
        try:
            files = plot_first_last_rss_diagnostics(
                wl__w, rss_flux, rss_eflux, bad_pixels, output_model,
                pek_model, args.output_path, args.label,
                fig_type=out_plot_format, best_model=gas_model_best)
            print(f'# Saved {len(files)} first/last RSS diagnostic figures')
        except Exception as exc:
            print(f'# Unable to write first/last RSS diagnostic figures: {exc}')

    #######################################################
    # Finalize the existing MODEL planes, and save the averaged spectrum.
    # Only the full RSS MODEL file remains controlled by --dump_model.
    #######################################################
    # Preserve channel 6=NP. Channel 7 holds PEK if available, otherwise PM.
    # The previous ninth channel redundantly duplicated the PEK joint model;
    # it now contains the independent, per-line selected BEST gas model.
    if pek_model is not None:
        output_model[7, :, :] = pek_model
        hdu_i_sps[0].header['NAME7'] = 'gas_model_PEK'
    output_model = adopt_best_model_planes(output_model, gas_model_best)
    hdu_i_sps[0].header['NAME8'] = 'gas_model_best'
    # Both gas-model arrays have been consumed by the final MODEL planes.
    # PEK/BEST quality statistics and optional plots were completed above.
    del pek_model, gas_model_best
    _mem.report('after_final_gas_model_release', arrays={'output_model': output_model})
    # Original DRP masks must apply to every final MODEL channel, including
    # the PEK-replaced and newly constructed BEST channels.
    output_model[:, bad_pixels] = np.nan

    # Mean each channel over valid fibers, without a full-sized temporary cube.
    # The original bad-pixel mask has already been applied to output_model.
    _mem.report('before_average_model', arrays={'output_model': output_model, 'bad_pixels': bad_pixels})
    avg_output_model = average_model_channels(output_model)
    if avg_output_model.shape != (output_model.shape[0], len(wl__w)):
        raise ValueError('Average MODEL has an unexpected spectral shape')
    avg_model_header = hdu_i_sps[0].header.copy()
    avg_model_header['NFIBERS'] = (len(tab_PT), 'Number of averaged RSS fibers')

    # The integrated plot uses the pre-flux-scaled model, so draw the
    # matching average-spectrum figure before converting physical units.
    if args.do_plots == 1:
        y_off = -1.09
        y_off2 = -2.18
        avg_plot_hdu = fits.HDUList([
            fits.PrimaryHDU(data=avg_output_model, header=avg_model_header.copy())])
        try:
            plot_spec_art_hdu(
                dir='', hdu=avg_plot_hdu,
                name=args.label, text=args.label,
                output=f'{args.output_path}/output_avg.{args.label}.{out_plot_format}',
                c_map='CMRmap',
                insets=((0.00, y_off, 0.09, 0.9,3715,3739,-1,35,'[OII]'),
                        (0.107, y_off, 0.09, 0.9,4088,4118,-0.5,10,r'H$\delta$'),
                        (0.213, y_off, 0.09, 0.9,4324,4357,-0.5,18,r'H$\gamma$'),
                        (0.320, y_off, 0.09, 0.9,4851,4872,-1.0,38,r'H$\beta$'),
                        (0.425, y_off, 0.09, 0.9,4998,5015,-1.0,50,'[OIII]'),
                        (0.530, y_off, 0.155, 0.9,6543,6590,-2,220,r'H$\alpha$+[NII]'),
                        (0.700, y_off, 0.09, 0.9,6709,6737,-0.5,20,'[SII]'),
                        (0.807, y_off, 0.09, 0.9,9062,9076,-1.0,28,'[SIII]'),
                        (0.910, y_off, 0.09, 0.9,9526,9537,-1.5,82,'[SIII]'),
                        (0.00, y_off2, 0.22, 0.9,3801,4099,-0.5,3.2,'3800-4100'),
                        (0.23, y_off2, 0.22, 0.9,4801,5099,-0.2,2.8,'4800-5100'),
                        (0.46, y_off2, 0.10, 0.9,4363-8.1,4363+7.9,-0.25,0.8,'[OIII]'),
                        (0.57, y_off2, 0.10, 0.9,4472-8.1,4472+7.9,-0.25,0.8,'HeI'),
                        (0.68, y_off2, 0.10, 0.9,7320-8,7320+8,-0.2,1.7,'[OII]'),
                        (0.79, y_off2, 0.10, 0.9,5755-6,5755+6,-0.25,2.5,'[NII]'),
                        (0.90, y_off2, 0.10, 0.9,6312-6,6312+6,-0.25,2.5,'[SIII]')),
                y_min=-0.5, y_max=30, y0_d=0.3, y1_d=2.9,
                x_min=3600, x_max=9600, plot_el=True, tab_el=tab_el,
                plot_res=True, show_scale=False, n_ord=2, gamma=0.5)
        except Exception as exc:
            print(f'# Unable to plot averaged RSS spectrum: {exc}')

    # Scale both products in place.  Neither operation creates a float64
    # copy of the full MODEL cube.
    output_model /= flux_scaling
    avg_output_model /= flux_scaling
    out_file_avg_model = os.path.join(
        args.output_path, f"avg_{args.label}.model.fits.gz")
    fits.PrimaryHDU(
        header=avg_model_header, data=avg_output_model).writeto(
            out_file_avg_model, overwrite=True)

    if args.dump_model == True:
        if args.verbose == True:
            print("# Writing output model to FITS file")
        out_file_model = os.path.join(args.output_path, f"{args.label}.model.fits.gz")
        model_header = hdu_i_sps[0].header.copy()
        if getattr(args, 'in_cube', False):
            # Avoid an extra multi-GiB allocation for a dense 4-D image.
            # (plane,RSS row,wavelength) is indexed via SPAXEL_INDEX in
            # <label>.cube.dap.fits.gz; no model pixels are rearranged.
            model_header['CUBEMODE'] = True
            model_header['MAPFILE'] = f'{args.label}.cube.dap.fits.gz'
            model_header['COMMENT'] = ('MODEL is plane/RSS-row/wave; '
                                       'SPAXEL_INDEX provides xpix/ypix mapping')
        hdu_model = fits.PrimaryHDU(
            header=model_header, data=output_model)
        hdu_model.writeto(out_file_model, overwrite=True)

    print("#******   ALL DONE ******#")


    
