import numpy as np

def Cardelli_extlaw(l, R_V=3.1):
    """Calculates q(lambda) to the Cardelli, Clayton & Mathis 1989
    reddening law.

    Parameters
    ----------
    l : array like
        Wavelenght vector (Angstroms)

    R_V : float, optional
        Selective extinction parameter (roughly "slope").
        Default is R_V = 3.1.

    Returns
    -------
    array like
        q(lambda) of the reddedning law.
    """
    a = np.zeros(np.shape(l))
    b = np.zeros(np.shape(l))
    F_a = np.zeros(np.shape(l))
    F_b = np.zeros(np.shape(l))
    x = np.zeros(np.shape(l))
    y = np.zeros(np.shape(l))
    q = np.zeros(np.shape(l))
    for i in range(0,len(l)):
        x[i]=10000/l[i]
        y[i]=10000/l[i] - 1.82
    # Far-Ultraviolet: 8 <= x <= 10 ; 1000 -> 1250 Angs
    inter = np.bitwise_and(x >= 8, x <= 10)
    a[inter] = -1.073 - 0.628*(x[inter] - 8.) + 0.137*(x[inter] - 8.)**2 - 0.070*(x[inter] - 8.)**3
    b[inter] = 13.670 + 4.257*(x[inter] - 8.) - 0.420*(x[inter] - 8.)**2 + 0.374*(x[inter] - 8.)**3
    # Ultraviolet: 3.3 <= x <= 8 ; 1250 -> 3030 Angs
    inter =  np.bitwise_and(x >= 5.9, x < 8)
    F_a[inter] = -0.04473*(x[inter] - 5.9)**2 - 0.009779*(x[inter] - 5.9)**3
    F_b[inter] =  0.2130*(x[inter] - 5.9)**2 + 0.1207*(x[inter] - 5.9)**3
    inter =  np.bitwise_and(x >= 3.3, x < 8)
    a[inter] =  1.752 - 0.316*x[inter] - 0.104/((x[inter] - 4.67)**2 + 0.341) + F_a[inter]
    b[inter] = -3.090 + 1.825*x[inter] + 1.206/((x[inter] - 4.62)**2 + 0.263) + F_b[inter]
    # Optical/NIR: 1.1 <= x <= 3.3 ; 3030 -> 9091 Angs ;
    inter = np.bitwise_and(x >= 1.1, x < 3.3)
    a[inter] = 1.+ 0.17699*y[inter] - 0.50447*y[inter]**2 - 0.02427*y[inter]**3 + 0.72085*y[inter]**4 + 0.01979*y[inter]**5 - 0.77530*y[inter]**6 + 0.32999*y[inter]**7
    b[inter] = 1.41338*y[inter] + 2.28305*y[inter]**2 + 1.07233*y[inter]**3 - 5.38434*y[inter]**4 - 0.62251*y[inter]**5 + 5.30260*y[inter]**6 - 2.09002*y[inter]**7
    # Infrared: 0.3 <= x <= 1.1 ; 9091 -> 33333 Angs ;
    inter = np.bitwise_and(x >= 0.3, x < 1.1)
    a[inter] =  0.574*x[inter]**1.61
    b[inter] = -0.527*x[inter]**1.61
    q_l = a + b/R_V
    return q_l

def Charlot_extlaw(l, mu=0.3):
    """
    Returns two-component dust model by Charlot and Fall 2000.

    Parameters
    ----------
    l : array like
        The \lambda wavelength array (in Angstroms).

    mu : float, optional
         Fraction of extinction contributed by the ambient ISM.
         Default value is 0.3.

    Returns
    -------
    array like
        \tau_lambda for t <= 10^7 yr.
    array like
        \tau_lambda for t > 10^7 yr.
    """

    t_lambda_Y = np.power((l/5500.), -0.7)
    t_lambda_O = mu*t_lambda_Y
    return t_lambda_Y, t_lambda_O

def Calzetti_extlaw(l, R_V=4.05):
    """
    Calculates the reddening law by Calzetti et al. (1994).
    Original comments in the .for file:

    q = A_lambda/A_V for Calzetti et al reddening law (formula from hyperz-manual).
    l = lambda, in Angstrons
    x = 1/lambda in 1/microns

    Parameters
    ----------
    l : array like
        Wavelength in Angstroms.
    R_V : float, optional
        Selective extinction parameter (roughly "slope").
        Default value is 4.05.

    Returns
    -------
    array like
        Extinction A_lamda/A_V. Array of same length as l.

    """
    if not isinstance(l, np.ma.MaskedArray):
        l = np.asarray(l, 'float64')
    x = 1e4/l
    q = np.zeros_like(l)
    # UV -> Optical: 1200 -> 6300 Angs
    inter = (l >= 1200.) & (l <= 6300.)
    q[inter] = (2.659/R_V)*(-2.156 + 1.509*x[inter] - 0.198*x[inter]**2 + 0.011*x[inter]**3) + 1.
    # Red -> Infrared
    inter = (l >= 6300.) & (l <= 22000.)
    q[inter] = (2.659/R_V)*(-1.857 + 1.040*x[inter]) + 1.
    # Issue a warning if lambda falls outside 1200->22000 Angs range
    if ((l < 1200.) | (l > 22000.)).any():
        print('[Calzetti_extlaw] WARNING! some lambda outside valid range (1200, 22000.)')
    return q

def calc_extlaw(l, R_V, extlaw=None):
    """
        A wrapper to select which extintion law to use if needed by user.

    Parameters
    ----------
    l : array like
        Wavelength in Angstroms.

    R_V : float
        Selective extinction parameter (roughly "slope").

    extlaw : str {'CCM', 'CAL'}, optional
        Which extinction function to use.
        CCM will call `Cardelli_extlaw`.
        CAL will call `Calzetti_extlaw`.
        Default value is CCM.

    Returns
    -------
    array like
        q(lambda) of the `extlaw` extinction law.
    """
    if (extlaw == 'CCM'):
        return Cardelli_extlaw(l, R_V)
    elif (extlaw == 'CAL'):
        return Calzetti_extlaw(l, R_V)
    else:
        raise ValueError((
            f'[{__name__}]: extlaw {extlaw} not inexistent. Try'
            f'CCM (Cardelli extinction law) or'
            f'CAL (Calzetti extinction law).'
        ))

def spec_apply_dust(wave, flux, AV, R_V=None, extlaw=None):
    """
    Apply the dust extinction law factor to a spectrum `flux` at wavelenghts
    `wave`.

        flux_intrinsic = `flux` * 10^(-0.4 * `AV` * q_l)

    Parameters
    ----------
    wave : array like
        Wavelenghts at restframe.

            (1 + redshift) = `wave` / wave_at_restframe

    flux : array like
        Fluxes to be dust-corrected.

    AV : float or array like
        Dust extinction in mag. If AV is an array, will create
        an (N_AV, N_wave) array of dust spectra.

    R_V : float or None
        Selective extinction parameter (roughly "slope"). Default value 3.1.

    extlaw : str {'CCM', 'CAL'} or None
        Which extinction function to use.
        CCM will call `Cardelli_extlaw`.
        CAL will call `Calzetti_extlaw`.
        Default value is CCM.

    Returns
    -------
    array like
        The `flux` correct by dust extinction.
    """
    if R_V is None:
        R_V = 3.1
    if extlaw is None:
        extlaw = 'CCM'
    q_l = calc_extlaw(wave, R_V, extlaw)
    dust_factor = 10**(-0.4*AV*q_l)
    return flux*dust_factor
