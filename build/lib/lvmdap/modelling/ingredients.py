
import numpy as np
from astropy.io import fits

from lvmdap.pyFIT3D.modelling.dust import spec_apply_dust
from lvmdap.pyFIT3D.common.stats import shift_convolve

class StellarModels(object):
    def __init__(self, filename):
        self.rsp = fits.open(filename)
        self.params = self.rsp[1].data
        self.n_wave = self.rsp[0].header["NAXIS1"]
        self.n_models = self.rsp[0].header["NAXIS2"]

        self.wavelength = self.get_wavelength()
        self.flux_models = self.rsp[0].data
        self.wavenorm = self.rsp[0].header["WAVENORM"]

        self.teff_models = self.params["TEFF"]
        self.logg_models = self.params["LOGG"]
        self.meta_models = self.params["MET"]
        self.alph_models = self.params["ALPHAM"]
        try:
            self.mass_to_light = 1 / self.params["FNORM"]
        except KeyError:
            self.mass_to_light = 1 / self.params["NORM"]

    def get_wavelength(self, mask=None):
        """ Creates wavelength array from FITS header. Applies a mask to
        wavelengths if `mask` is set.

        Parameters
        ----------
        redshift : float, optional
            Redshift to correct the generated wavelength array.
        mask : array_like, optional
            Masked wavelengths.

        Returns
        -------
        array_like
            Wavelenght array from FITS header.
        """
        crval = self.rsp[0].header["CRVAL1"]
        cdelt = self.rsp[0].header["CDELT1"]
        crpix = self.rsp[0].header["CRPIX1"]
        pixels = np.arange(self.n_wave) + 1 - crpix
        w = crval + cdelt*pixels
        if mask is None:
            return w
        return w[mask]

    def moments_from_coeffs(self, coeffs):
        coeffs = np.array(coeffs)
        coeffs[~np.isfinite(coeffs)] = 0
        if (coeffs == 0).all():
            teff, logg, meta, alph, teff_mass, logg_mass, meta_mass, alph_mass = 0, 0, 0, 0, 0, 0, 0, 0
        else:
            coeffs_normed = coeffs/coeffs.sum()
            teff = np.dot(coeffs_normed, self.teff_models)
            logg = np.dot(coeffs_normed, self.logg_models)
            meta = np.dot(coeffs_normed, self.meta_models)
            alph = np.dot(coeffs_normed, self.alph_models)

            coeffs_normed_mass = coeffs/np.dot(coeffs, self.mass_to_light)
            teff_mass = np.dot(coeffs_normed_mass*self.mass_to_light, self.teff_models)
            logg_mass = np.dot(coeffs_normed_mass*self.mass_to_light, self.logg_models)
            meta_mass = np.dot(coeffs_normed_mass*self.mass_to_light, self.meta_models)
            alph_mass = np.dot(coeffs_normed_mass*self.mass_to_light, self.alph_models)
        return teff, logg, meta, alph, teff_mass, logg_mass, meta_mass, alph_mass

    def get_model_from_coeffs(self, coeffs, wavelength, sigma=None, redshift=None, AV=None, sigma_inst=None, R_V=None, extlaw=None, return_tZ=False):
        """
        Shift and convolves stellar template model fluxes (i.e. `self.flux_models`) to
        wavelengths `wave_obs` using `sigma` and `sigma_inst`. After this,
        applies dust extinction to the stellar templates following the extinction law
        `extlaw` with `AV` attenuance. At the end, returns the stellar template model
        spectra using `coeffs`.

        Parameters
        ----------
        coeffs : array_like
            Coefficients of each stellar template model. If `fit` is True get coeffs from fit.
        wavelength : array_like
            Wavelenghts at observed frame. if `fit` is True uses the one set by
            `fit_kwargs`.
        sigma : float or None
            Velocity dispersion (i.e. sigma) at observed frame.
            If the sigma is not set None, is used `cf.sigma`.
        redshift : float or None
            Redshift of the Observed frame. If the `redshift` is None, is used `cf.redshift`.
        AV : float or array_like
            Dust extinction in mag.

            TODO: If AV is an array, will create an (n_AV, n_wave) array of dust spectra.
        sigma_inst : float or None
            Sigma instrumental. If the `sigma_inst` is None, is used `self.sigma_inst`.
        R_V : float, optional
            Selective extinction parameter (roughly "slope"). Default value 3.1.
        extlaw : str {"CCM", "CAL"}, optional
            Which extinction function to use.
            CCM will call `Cardelli_extlaw`.
            CAL will call `Calzetti_extlaw`.
            Default value is CCM.
        return_tZ : bool, optional
            Also returns the age and metallicity for the model.

        Returns
        -------
        array_like
            stellar template model spectrum created by coeffs.

        list of floats
            Only returned if `return_tZ` is True.

            The list carries:
            [t_LW, Z_LW, t_MW, Z_MW]
            Age, metallicity light- and mass-weighted.
        """
        coeffs = np.array(coeffs)
        coeffs[~np.isfinite(coeffs)] = 0
        if (coeffs == 0).all():
            model = np.zeros(wavelength.size)
        else:
            self.to_observed(wavelength, sigma_inst=sigma_inst, sigma=sigma, redshift=redshift, coeffs=coeffs)
            self.apply_dust_to_flux_models_obsframe(wavelength/(1 + redshift), AV, R_V=R_V, extlaw=extlaw)
            model = np.dot(coeffs[coeffs > 0], self.flux_models_obsframe_dust)

        if return_tZ:
            return model, self.moments_from_coeffs(coeffs)
        return model

    def to_observed(self, wavelength, sigma_inst, sigma, redshift, coeffs=None):
        """
        Shift and convolves `self.flux_models` to wavelengths `wavelength` using `sigma`
        and `sigma_inst`.

        Parameters
        ----------
        wavelength : array_like
            Wavelenghts at observed frame.
        sigma_inst : float
            Sigma instrumental defined by user in the program call.
        sigma : float
            Sigma of the Observed frame.
        redshift : float
            Redshift of the Observed frame.
        """
        if coeffs is None:
            flux_models_obsframe = np.asarray([
                shift_convolve(wavelength, self.wavelength, self.flux_models[i], redshift, sigma_inst, sigma)
                for i in range(self.n_models)
            ])
        else:
            # generates a dictionary with i_tZ as key and the coeff as the dict[i_tZ] = coeff
            tZcoeff_dict = {i: c for i, c in enumerate(coeffs) if c > 0}
            flux_models_obsframe = np.asarray([
                shift_convolve(wavelength, self.wavelength, self.flux_models[i], redshift,
                               sigma_inst, sigma)
                for i in tZcoeff_dict.keys()
            ])
        self.flux_models_obsframe = flux_models_obsframe
        m = flux_models_obsframe[0] > 0
        self._msk_wavelength_obsframe = wavelength[m]

    # POSSIBLE BUG: to apply the *internal* dust extinction in the observed frame is incorrect
    def apply_dust_to_flux_models_obsframe(self, wavelength_rest_frame, AV, R_V=3.1, extlaw="CCM"):
        self.flux_models_obsframe_dust = spec_apply_dust(
            wavelength_rest_frame, self.flux_models_obsframe, AV, R_V=R_V, extlaw=extlaw
        )

    def apply_dust_to_flux_models(self, wavelength_rest_frame, AV, R_V=3.1, extlaw="CCM"):
        self.flux_models_dust = spec_apply_dust(
            wavelength_rest_frame, self.flux_models, AV, R_V=R_V, extlaw=extlaw
        )
