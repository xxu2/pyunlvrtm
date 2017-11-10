"""
The io module contains a group of functions to read UNL-VRTM model output and
prepare model input files. 

"""
import sys
import numpy as np
from netCDF4 import Dataset

__all__ = ['read_unlvrtm','create_spectra','make_spectra_dat', 'make_atmos_nc']

###############################################################################
# Private utility functions.

# Routine to check inputs
def _check_inputs(var_dim, var, var_name):
   array_type = (list, tuple, np.ndarray)
   if isinstance(var,array_type):
      if len(np.squeeze(var)) != var_dim:
         sys.exit("make_spectra_dat: The input "+var_name+" should have same size of spectra... Please check!")
      return np.squeeze(var)
   else:
      return np.zeros(var_dim) + var


###############################################################################
# Public functions

def read_unlvrtm( filename, var=['Stokes'] ):
   """
   Read variables from unl-vrtm output a netCDF file.

   Parameters
   ----------
   filename: a string, the filename of unl-vrtm output netCDF file
        var: list of variable names to be read
   
   Returns
   -------
   outdata: Dict of variables read from the file

   """
   from netCDF4 import Dataset
   #import numpy as np

   # variables to be obtained
   varnames = ['Lamdas', 'Wavenum', 'SZA']
   varnames.extend( var )

   # open file to read
   ncf = Dataset( filename, 'r' )

   # define a DICTIONARY to return
   outdata = {'Source':filename}

   for vname in varnames:
      if vname in ncf.variables.keys():

         if vname in ['Gas', 'LinPar', 'BRDFKernel']:
            # Convert byte to strings 
            nv = len( ncf.variables[vname][:,0] )
            outdata[vname] = np.array([ ncf.variables[vname][iv,:].tostring().decode(encoding='UTF-8').strip() for iv in range(nv) ])
         else:
            # Regular variables
            outdata[vname] = np.squeeze( ncf.variables[vname][:] )
      else:
         print('Warning: '+vname+' is not found in the file. Skipped!')

   # close netcdf file
   ncf.close()

   return outdata

###
def create_spectra(mins, maxs, ns_or_interval, interval=False, freq=False ):
   """
   Create a set of equal-interval spectra based on min, max, and interval(or nspectra) 

   Parameters
   ----------
       mins, maxs: min and max of spectral values
   ns_or_interval: nspectra or interval value
         interval: If true, ns_or_interval is interval value, if false (default), nspectra
             freq: If true, inputs are wavenumber in cm^-1, else (defalut), wavelength in nm   

   Returns
   -------
   outdata: Dict of wvn and lam

   """

   # determine nspectra
   if interval:
      ns = int( (maxs-mins)/ns_or_interval )
   else:
      ns = ns_or_interval

   # Calculate 
   if freq:
      wvn = np.linspace(mins, maxs, ns)
      lam = 1e7 / wvn
   else:
      lam = np.linspace(mins, maxs, ns)
      wvn = 1e7 / lam
   return dict(wvn=wvn, lam=lam)

###
def make_spectra_dat(spectra, nr1=1.33, ni1=0.0, nr2=1.33, ni2=0.0, s1=0.0, s2=0.0, s3=0.0,
                     filename="spectra.dat", casename='Default'):
   """
   Create a 'spectra.dat' file based on the input refractive index and surface parameters
    
   Parameters
   ----------
    spectra: A set of spectral wavelength in nm or wavenumber in cm^-1
    nr1,ni1: Real and imaginary parts of refractive index of the 1st aerosol mode.
             Default values are 1.33 and 0, respectively.
    nr2,ni2: Real and imaginary parts of refractive index of the 2nd aerosol mode,
             Default values are 1.33 and 0, respectively.
   s1,s2,s3: The 3 surface parameters (see UNL-VRTM User's Guide for detail).
   filename: The generated filename, default is 'spectra.dat'
   casename: A string of casename to be writen in the file header.

   Returns
   -------
   None.

   Note
   ----
   nr1, nr2, ni1, ni2, s1, s2, s3 should have same size to spectra.
   
   """ 


   # Number of spectrum
   spectra = np.squeeze(spectra)
   ns = len(spectra) 
   #print('ns = ', ns, spectra.shape)

   # Routine to check inputs
   #def check_inputs(var_dim, var, var_name):
   #   array_type = (list, tuple, np.ndarray)
   #   if isinstance(var,array_type):
   #      if len(np.squeeze(var)) != var_dim:
   #         sys.exit("make_spectra_dat: The input "+var_name+" should have same size of spectra... Please check!")
   #      return np.squeeze(var)
   #   else:
   #      return np.zeros(var_dim) + var
  
   # Check each input 
   wrt_nr1 = _check_inputs( ns, nr1, 'nr1' )
   wrt_ni1 = _check_inputs( ns, ni1, 'ni1' )
   wrt_nr2 = _check_inputs( ns, nr2, 'nr2' )
   wrt_ni2 = _check_inputs( ns, ni2, 'ni2' )
   wrt_s1  = _check_inputs( ns,  s1,  's1' )
   wrt_s2  = _check_inputs( ns,  s2,  's2' )
   wrt_s3  = _check_inputs( ns,  s3,  's3' )
   #print(type(wrt_nr1))

   # Current time
   import datetime
   now = datetime.datetime.now()

   # Open a file to write
   with open(filename, 'w') as f:
      f.write('#1=============================================================================\n')
      f.write('#2 $id:spectra.dat, specifically for multi-spectral simulations by UNL-VRTM\n')
      f.write('#3   - first 10 lines are used for comments\n')
      f.write('#4   - 11th line: specify number of spectral\n')
      f.write('#5   - following lines specify wavelength(nm), n_real, n_img, n_real, n_img:\n')
      f.write('#6        - column#1  : wavelength(nm)\n')
      f.write('#7        - column#2-5:  n_real, n_img for 1st mode, and n_real, n_img for 2nd\n')
      f.write('#8        - column#6-8:  amplification foactor for BRDF model\n')
      f.write('#9 Version 1.3 (xxu, 8/26/13); Note: '+ casename + now.strftime(" %Y-%m-%d %H:%M") + '\n')
      f.write('#10============================================================================ \n')
      f.write('{:6d}'.format(ns) + '\n')
      line_style = '{:10.4f}{:8.3f}{:11.3E}{:8.3f}{:11.3E}{:8.3f}{:8.3f}{:8.3f}'
      #print('ddddddddddddddd')
      for il in range(ns): 
         #print( spectra[il], wrt_nr1[il], wrt_ni1[il], wrt_nr2[il], wrt_ni2[il], wrt_s1[il], wrt_s2[il], wrt_s3[il] ) 
         f.write( line_style.format(spectra[il], wrt_nr1[il], wrt_ni1[il], wrt_nr2[il], wrt_ni2[il], wrt_s1[il], wrt_s2[il], wrt_s3[il]) + '\n')

   print("Make spectra data file: "+filename)

###
def nmlist_replace(oldfile, oldstr, newfile, newstr):
   """
   Replace a string (or a list of strings) in a file with a new string (or a list of new strings)

   Parameters
   ----------
   oldfile: The name of the file with text to be replaced
    oldstr: The string (or a list of strings) in the oldfile to be replaced
   newfile: The name of for a new file to be generated
    newstr: The text string to be used for replacing the oldstr

   Returns
   -------
   None.

   """

   # open the oldfile
   fold = open( oldfile, 'r')
   filedata = fold.read()
   fold.close()

   # replace string in the old file
   if np.isscaler(oldstr):
     filedata = filedata.replace( oldstr, newstr )
   else:
     if np.size(oldstr) != np.size(newstr):
        print( 'size(oldstr) = ', np.size(oldstr) )
        print( 'size(newstr) = ', np.size(newstr) )
        sys.exit('Their size should be same, but not ...io.py:nmlist_replace() ') 
     for istr in range(len(oldstr)):
        filedata = filedata.replace( oldstr[istr], newstr[istr] )
  
   # open a new file to save updates
   fnew = open(newfile,'w')
   fnew.write(filedata)
   fnew.close()  

   return 


### 
def make_atmos_nc( var_dict, filename='atmos.nc' ):
   """
   Currently under development
   """

   # To be added from fimchem_clarreo_for_unlvrtm.py ->
   # /Dedicated/jwang-data/xxu69/CLARREO/scripts/fromcranehome/orbit2/py
   return -1
