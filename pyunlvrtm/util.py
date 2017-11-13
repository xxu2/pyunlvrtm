"""
The util module contains a group of functions to process 
model outputs

"""
import sys
import numpy as np

__all__ = ['dolp', 'dolp_l', \
           'geometry_index', \
           'radiance2bt', \
           'scattering_angle']

###############################################################################
# Private utility functions.


###############################################################################
# Public functions

######
def dolp( Stokes ):
   """
   Function delp() calculates the degree of linear polarization with the input
   of Stokes vector.

   Parameters
   ----------
   Stokes: A Stokes vector containing at least the first two of [I,Q,U,V]

   Returns:
   --------
   P: degree of linear polarization

   """

   # number of Stokes elements
   nS = np.size(Stokes)

   if (nS in [2]):
      # DOLP = -Q/I
      P = - Stokes[1] / Stokes[0] 
   elif (nS in [3,4]):
      #          _______
      # DOLP = _/Q^2+U^2 / I
      P = np.sqrt(Stokes[1]*Stokes[1] + Stokes[2]*Stokes[2]) / Stokes[0]
   else:
      sys.exit("dolp:Number of Stokes elements is incorrect: "+str(nS))

   return P


######
def dolp_l(Stokes, l_Stokes):
   """
   Function delp_l() calculates the Jacobian of DOLP, given the Jacobian 
   of I, Q, and U.
   
   Parameters
   ----------
   Stokes: A Stokes vector containing at least the first two of [I,Q,U,V]
   l_Stokes: Jacobian of the Stokes vector 

   Returns:
   --------
   P: degree of linear polarization
   l_P: Jacobian of degree of linear polarization

   """
   
   # Check input
   if (np.size(Stokes) != np.size(l_Stokes)):
      sys.exit( "dolp_l:l_Stokes contains different number of elements from Stokes!")

   # Get DLOP
   P = dolp(Stokes)

   # Now calculate l_P
   nS = np.size(Stokes)
   if (nS == 2):
      l_P = ( Stokes[1]*l_Stokes[0] - Stokes[0]*l_Stokes[1] ) / (Stokes[0]*Stokes[0])
   else:
      l_P = l_Stokes[0] * ( - P / Stokes[0] ) \
          + (Stokes[1]*l_Stokes[1] + Stokes[2]*l_Stokes[2]) / (P*Stokes[0]*Stokes[0])

   return P, l_P

######
def geometry_index( nsza, nvza, nraz, \
                    isza, ivza, iraz ):
   """
   Calculate the geometry index for UNL-VRTM output of STOKES parameter

   Parameters
   ----------
      nsza: number of solar zenith angles
      nvza: ......... viewing zenith angles
      nraz: ......... relative azimuthal angles
      isza: subscripts (index) of the specified solar zenith angle 
      ivza: ................................... viewing zenith angle
      iraz: ................................... relative azimuthal angle
  
     Note that isza, (ivza, and iraz) should be a number between 0 
     and nsza-1, (nvza-1, and nraz-1), following Python array's 
     indixing rule.
 
   Returns
   -------
      i_geom: index for the geometry dimension of variable STOKES.

   """
   # total number of geometry combinations
   ngeom = nsza * nvza * nraz

   # vza offset
   vza_offset = isza * nvza + ivza

   # geometry index
   igeom = nraz * vza_offset + iraz

   # return to the calling routine
   return igeom, ngeom


######
def scattering_angle( sza, vza, phi, Expand=False, Degree=False ):
   """
   Function scattering_angle() calculates the scattering angle.
   cos(pi-THETA) = cos(theta)cos(theta0) + sin(theta)sin(theta0)cos(phi)
   Input and output are in the unit of PI

   Parameters
   ----------
   sza: solar zenith angle is radian
   vza: viewing zenith angle in radian
   phi: relative azimuth angle in radian
   Expand: (optional) Ture/False to expand the dimension of calculated THETA

   Returns
   -------
   THETA: scattering angle in radian
   
   """

   # Change angle from degree to radian if needed
   if Degree:
     angle2rad = np.pi / 180.
     sza = sza * angle2rad
     vza = vza * angle2rad
     phi = phi * angle2rad

   # define the 
   m,n,l = np.size(sza),np.size(vza),np.size(phi)

   if Expand:
      THETA = np.zeros( (m,n,l) )
      for k in range(l):
         for j in range(n):
            for i in range(m):
               t1 = np.cos(vza[j]) * np.cos(sza[i]) \
                  + np.sin(vza[j]) * np.sin(sza[i]) * np.cos(phi[k])
               t2 = np.arccos(t1)
               THETA[i,j,k] = np.pi - t2
   else:
      # Check the dimension
      if (( m != n) | (m != l )):
         sys.ext("scattering_angle() error #1 in util.py")
      t1 = np.cos(vza) * np.cos(sza) \
         + np.sin(vza) * np.sin(sza) * np.cos(phi)
      t2 = np.arccos(t1)
      THETA = np.pi - t2

   if Degree:
      THETA = THETA * 180. / np.pi

   return THETA


######
def radiance2bt( Radiance, Spectra, unit='wavenumber' ):
   '''
   Function radiance2bt calculates brightness temperature fron the input 
   spectral radiances on either wavenumber [default] and wavelength unit.
   (xxu, 8/16/16, 6/26/17)

   Parameters
   ----------
   Radiance:  Radiance for which Planck temperature is required.
              Units: mW/(m2.sr.cm-1) [DEFAULT]
                     W/(m2.sr.um)    [If unit='wavelength']
              Dimension: Scalar or 1-D array
    
   Spectra: Spectral originate at which Planck temperature is calculated.
            Units: Inverse centimeters (cm^-1) [DEFAULT]
                   Microns (um)                [If unit='wavelength'] 

   unit: optional = 'wavenumber'(default) or 'wavelength'

   Returns 
   -------
   Brightness (Planck) temperature in Kelvin, dimension is same as inputs

   Notes
   -----
   6/26/17: add keyword "unit" to extend the calculation from radiance in
            wavelength space. (xxu)

   '''

   # unit_id 0 for wavenumber and 1 for wavelength
   if unit == 'wavenumber':
      unit_id = 0
   elif unit == 'wavelength':
      unit_id = 1
   else:
      print( 'unit should be either wavenumber or wavelength' )
      return -1

   # unit_id 0 for wavenumber and 1 for wavelength
   if unit == 'wavenumber':
      unit_id = 0
   elif unit == 'wavelength':
      unit_id = 1
   else:
      print( 'unit should be either wavenumber or wavelength' )
      return -1

   # Some constants
   Planck_Constant = 6.626068e-34 # [joule sec]
   #Boltzmann_Constant = 1.38066e-12  # [joule deg^-1]
   Light_Speed = 2.997925e+8  # [m s^-1]
   Avogadro_Constant =  6.02214199e+23  #[mole^-1]  
   Molar_Gas_Constant = 8.314472 # [joule/mole/K]
   Boltzmann_Constant = Molar_Gas_Constant / Avogadro_Constant

   # units conversions
   # Radiance scale factor
   #    Frequency:  Scaling factor to convert mW/(m2.sr.cm-1) -> W/(m2.sr.cm-1)
   #    Wavelength: Scaling factor set to 1.0 since for wavelength we want W/(m2.sr.um)
   R_scale = [1.0e+03, 1.0]
   # C1 derived constant scale factor
   #    Frequency:   W.m2 to W/(m2.cm-4) => multiplier of 1.0e+08 is required.
   #    Wavelength:  W.m2 to W/(m2.um-4) => multiplier of 1.0e+24 is required.
   C1_scale = [1.0e+08, 1.0e+24]
   # C2 derived constant scale factor
   #    Frequency:   K.m to K.cm => multiplier of 100 is required
   #    Wavelength:  K.m to K.um => multiplier of 1.0e+06 is required.
   C2_scale = [1.0e+02, 1.0e+06]

   # First Planck function constant
   #   Symbol:c1,  Units:W.m^2.sr^-1; c1 = 1.191042722(93)e-16
   C1 = 2.0 * Planck_Constant * Light_Speed ** 2

   # Second Planck function constant
   #   Symbol:c2,  Units:K.m; c2 = 1.4387752(25)e-02
   C2 = Planck_Constant * Light_Speed / Boltzmann_Constant

   # Comput FK1 and FK2 quantities
   if unit == 'wavenumber':
      Fk1 = C1_scale[unit_id] * C1 * Spectra**3
      Fk2 = C2_scale[unit_id] * C2 * Spectra
   elif unit == 'wavelength':
      Fk1 = C1_scale[unit_id] * C1 / Spectra**5
      Fk2 = C2_scale[unit_id] * C2 / Spectra

   # Compute the Planck temperature
   BT = Fk2 / np.log( Fk1 / ( Radiance / R_scale[unit_id] ) + 1.0 )

   # Return to the calling routine
   return BT
