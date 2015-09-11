# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 13:11:37 2015

@author: dp11

Script to warp the source sequence to the target sequence given the input dof
rigid transformation

"""

import numpy as np
import SimpleITK as sitk
from registration import registration
import os

N_DOF           = 6         # Rigid registration

os.environ["SITK_SHOW_COMMAND"] = "itksnap"

def usage( sname ):
    """ Define input and output """
    os.sys.stderr.write("\nWarp source sequence to target sequence given the input dof file\n")
    os.sys.stderr.write("\nUsage: {0} -target_path <path_to_target_seq> "\
        "-source_path <path_to_source_seq> -dofin <dof_file> -out_path"\
        " <out_path> -v\n\n".format( os.path.split(sname)[-1] ) )
    os.sys.stderr.write("Where:\t-data_path\t\tis the path to target and source sequences\n")
    os.sys.stderr.write("\t-dofin\t\tname of input file with DOFs\n")
    os.sys.stderr.write("\t-out_path\t\tpath where transformed sequence is written\n")
    os.sys.stderr.write("\t<opt>-v\t\tverbose mode\n\n")
    return True
    
def read_inputs( argv, argc ):
    """ Parse input arguments """
    argc = len(argv)
    tpath, spath, dfile, opath, verbose = 0, 0, 0, 0, False
    for a in range( argc ):
        if argv[a] == '-target_path':
            tpath = a
        elif argv[a] == '-source_path':
            spath = a
        elif argv[a] == '-dofin':
            dfile = a
        elif argv[a] == '-out_path':
            opath = a
        elif argv[a] == '-v':
            verbose = True
    if tpath==0 or spath==0 or dfile==0 or opath==0:
        os.sys.stderr.write('Can''t parse input arguments')
        return True
    return argv[tpath+1], argv[spath+1], argv[dfile+1], argv[opath+1], verbose 
    
def write_registered_sequence( sfiles, tref, sseq, dof, out_path ):
    """ Function to write registered images to folder """
    if not os.path.exists( out_path ):
        os.mkdir( out_path )
#==============================================================================
#   Compute centre of target image  
#==============================================================================
    centre = np.dot( np.array( tref.GetDirection() ).reshape((3,3)),\
        (np.array(tref.GetSize())-1)*tref.GetSpacing()/2.0 )
    centre += tref.GetOrigin()
#==============================================================================
#   Set up the transform      
#==============================================================================
    etransf = sitk.Transform( 3, sitk.sitkEuler )
    etransf.SetFixedParameters( centre )
    etransf.SetParameters( dof )
#    rot_matrix = registration.SetRotation( dof[0], dof[1], dof[2] )
#==============================================================================
#   Warp source sequence given the input dof file   
#==============================================================================
    padder = sitk.ConstantPadImageFilter()
    padded = padder.Execute( tref, [20,20,20], [20,20,20], 0 )
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage( padded )
    resampler.SetInterpolator( sitk.sitkLinear )  # sitkBSpline
    resampler.SetTransform( etransf )
    os.sys.stdout.write('Writing sequence to folder\n')
    for name, image in zip( sfiles, sseq ):
        sitk.WriteImage( resampler.Execute( image ), '{0}{1}'.format(out_path, name) )
    return

def main(argv):
    """ Check input arguments """
    argc = len(argv)
    if argc < 9:
        return usage(argv[0])
    else:
        target_dir, source_dir, doffile, out_path, verbose = read_inputs(argv, argc)
#==============================================================================
#   Read names of input target and source sequences        
#==============================================================================
    target_files = sorted(os.listdir( target_dir ))
    source_files = sorted(os.listdir( source_dir ))
    if verbose:
        os.sys.stdout.write('Reading target sequence from {0}\n'.format(target_dir))
        os.sys.stdout.write('Reading source sequence from {0}\n'.format(source_dir))
#==============================================================================
#   Read and resample target sequence to a 2x2x2 isotropic volume  
#==============================================================================
    target_reference = sitk.ReadImage('{0}{1}'.format( target_dir, target_files[0] ))
#==============================================================================
#   If images are metadata one file is for metadata and one for raw
#==============================================================================
    extension = target_files[0].split('.')[1]
    if extension == 'mhd' or extension=='mha':
        target_files = target_files[::2]
        source_files = source_files[::2]
#==============================================================================
#   Read and resample source sequence to target image   
#==============================================================================
    osource_sequence = []
    for n, f in enumerate( source_files ):
        us_image = sitk.ReadImage('{0}{1}'.format( source_dir, f ))
        osource_sequence.append( us_image )
#==============================================================================
#   Read dof file
#==============================================================================
    idof = np.zeros(N_DOF)
    tdof = np.loadtxt( doffile, skiprows=1 )[:,2]
    idof[0:3] = np.pi*(tdof[3:]/180.0)
    idof[3:] = tdof[0:3]
#==============================================================================
#   Write transformed sequence to folder
#==============================================================================
    write_registered_sequence( source_files, target_reference, osource_sequence,\
        idof, out_path )
    
if __name__ == "__main__":
    os.sys.exit(main(os.sys.argv))
