# ToDo list

## Priorities

- ** Target selected redshift **: 
   
  1) Introduce a yaml variable
  2) Use Gaia vel-map as prior
  3) Include LV targets (lvmcore?)
  
- ** Modify the smoothing residual procedure **
 
  1) Use the code already intro

- ** Modify the stellar velocity analysis **

  1) Use the Ca H+K and NaID fitting method
  2) Include other spectral features? MgII? Fe?
  
- ** Better emission line masking **

  1) Use the method introduced by the sky-subtraction:
  
  - Define the continuum level with a broad filtering
  - Auto-detect emission lines
  - Match with the list: Velocity
  - Substract adopting intrumental dispersion.

- ** Fitting all emission lines with Gaussian functions **
  
   1) Fixing velocity and velocity dispersion to Halpha
   
- ** Improved stellar library **

   1) We matched GAIA stars and LVM stellar spectra.
   2) May we assign the stellar spectra to GAIA parameters?
   3) Should we re-run the DAP?
   
   
   
