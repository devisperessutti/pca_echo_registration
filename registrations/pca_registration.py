# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 11:28:57 2015

@author: dp11

This code implements PCA-based registration of image sequences
"""
#==============================================================================
# Imports
#==============================================================================
import numpy as np
import SimpleITK as sitk
import time
from pca.pca import PCA
import os
#==============================================================================
# Hard-coded variables  TODO - add as input arguments
#==============================================================================
N_DOF           = 6         # Rigid registration
MIN_OVERLAP     = 10000
MAX_DIST        = 1e15
spacing         = [2, 2, 2]  # resampling spacing
fraction        = .99        # fraction of retained variance
_GAMMA          = .1
_SIGMA          = 1

os.environ["SITK_SHOW_COMMAND"] = "itksnap"

def usage( sname ):
    """ Define input and output """
    os.sys.stderr.write("\nPCA-based rigid registration of image sequences\n")
    os.sys.stderr.write("\nUsage: {0} -target_path <path_to_target_seq> "\
        "-source_path <path_to_source_seq> -dofout <dof_file> -out_path"\
        " <out_path> -out_time <otime> -v\n\n".format( os.path.split(sname)[-1] ) )
    os.sys.stderr.write("Where:\t-data_path\t\tis the path to target and source sequences\n")
    os.sys.stderr.write("\t-dofout\t\tname of output file with resulting DOFs\n")
    os.sys.stderr.write("\t-out_path\t\tpath where registered sequence is written\n")
    os.sys.stderr.write("\t-out_time\t\tfile where execution time is written\n")
    os.sys.stderr.write("\t<opt>-v\t\tverbose mode\n\n")
    return True
    
def read_inputs( argv, argc ):
    """ Parse input arguments """
    argc = len(argv)
    tpath, spath, dfile, opath, otime, verbose = 0, 0, 0, 0, 0, False
    for a in range( argc ):
        if argv[a] == '-target_path':
            tpath = a
        elif argv[a] == '-source_path':
            spath = a
        elif argv[a] == '-dofout':
            dfile = a
        elif argv[a] == '-out_path':
            opath = a
        elif argv[a] == '-out_time':
            otime = a
        elif argv[a] == '-v':
            verbose = True
    if tpath==0 or spath==0 or dfile==0 or opath==0 or otime==0:
        os.sys.stderr.write('Can''t parse input arguments')
        return True
    return argv[tpath+1], argv[spath+1], argv[dfile+1], argv[opath+1], \
        argv[otime+1], verbose 
    
def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

def compute_info( image ):
    """ Compute information on resampled and original images """
    old_size, old_spac, old_orig = image.GetSize(), image.GetSpacing(), \
        image.GetOrigin()
    new_size, new_spac, new_orig  = [], [], old_orig
    for i in range(len(old_size)): 
        new_size.append( int(np.ceil( 1.0*(old_size[i]-1)*old_spac[i]/spacing[i] ) + 1) ) 
    for i in range(len(new_size)):
        new_spac.append( 1.0*old_spac[i]*old_size[i]/new_size[i] )
#==============================================================================
#   Store information in two dictionaries
#==============================================================================
    old_image_info = {'Size':old_size, 'Spacing':old_spac, 'Origin':old_orig, \
        'Direction':image.GetDirection()}
    new_image_info = {'Size':new_size, 'Spacing':new_spac, 'Origin':new_orig, \
        'Direction':image.GetDirection()}
    return new_image_info, old_image_info

def get_overlapping_mask( image_1, image_2 ):
    """ Returns the indeces of the overlapping region between image_1 and image_2 """
    ref_nda_seq_1 = sitk.GetArrayFromImage( image_1 ).ravel()
    ref_nda_seq_2 = sitk.GetArrayFromImage( image_2 ).ravel()
    mask_1, mask_2 = np.zeros(len(ref_nda_seq_1),dtype=bool),\
        np.zeros(len(ref_nda_seq_2),dtype=bool) 
    mask_1[ np.where(ref_nda_seq_1>0)[0] ] = True
    mask_2[ np.where(ref_nda_seq_2>0)[0] ] = True
    mask = mask_1&mask_2
    if np.sum(mask)>MIN_OVERLAP:
        overlap = True
    else:
        overlap = False
    return mask, overlap

def get_length_from_list( image_sequence ):
    """ Function to get PCs of given image sequence """
    N, D = len(image_sequence), len(sitk.GetArrayFromImage( image_sequence[0] ).ravel())
    X = np.zeros([D,N])
    for n in range(N):
        X[:,n] = sitk.GetArrayFromImage( image_sequence[n] ).ravel()
    ipca = PCA( X.T, np.eye(N), fraction=fraction, ptype='dual' )
    iZ = ipca.transform()
    length = 0
    for n in range(N-1):
        length += np.linalg.norm( iZ[n,:]-iZ[n+1,:] )
    del X, ipca
    return length

def get_pca_distance( tseq, wseq, mask, perimeter, vis ):
    """ Function to compute the distance between PCA manifolds """
#==============================================================================
#   Create matrices to store the sequences  
#==============================================================================
    n_t, n_w, D = len(tseq), len(wseq), len(sitk.GetArrayFromImage( tseq[0] ).ravel())
    tX, wX = np.zeros([D,n_t]), np.zeros([D,n_w])
    for n in range(n_t):
        tX[:,n] = sitk.GetArrayFromImage( tseq[n] ).ravel()
    for n in range(n_w):
        wX[:,n] = sitk.GetArrayFromImage( wseq[n] ).ravel()
#==============================================================================
#   Reduce dimensionality of target sequence  
#==============================================================================
    tpca = PCA( tX[mask,:].T, np.eye(n_t), fraction=fraction, ptype='dual' )
    tZ = tpca.transform()
    U = np.dot( np.dot( tpca.psi, tpca.eigVec ), \
        np.linalg.inv(np.diag(np.sqrt( tpca.eigVal )) ) )
#==============================================================================
#   Project warped sequence onto reduced subspace of target  
#==============================================================================
    wpca = PCA( wX[mask,:].T, np.eye(n_w), fraction=fraction, ptype='dual' )
    wtZ = np.real( np.dot( U.T, wpca.A ).T )
    length = 0
    for n in range(n_w-1):
        length += np.linalg.norm( wtZ[n,:]-wtZ[n+1,:] )
#==============================================================================
#   Find correspondences if sequences have different number of frames  
#==============================================================================
    n_vertices = np.max( [n_t, n_w])
    idx_t, idx_w = np.zeros( n_vertices, dtype = np.uint ),\
        np.zeros( n_vertices, dtype = np.uint )
    at, aw = np.linspace(0,1,num=n_t), np.linspace(0,1,num=n_w)
    if n_t>n_w:
        idx_t = np.arange( n_vertices )
        for ni in np.arange( n_vertices ):
            idx_w[ni] = find_nearest( aw, at[ni] )
    elif n_t<n_w:
        idx_w = np.arange( n_vertices )
        for ni in np.arange(n_vertices):
            idx_t[ni] = find_nearest( at, aw[ni] )
    else:
        idx_t = np.arange( n_vertices )
        idx_w = np.arange( n_vertices )
    return _GAMMA*np.sum( np.linalg.norm(wtZ[idx_w,:]-tZ[idx_t,:], axis=1) ) +\
        (1-_GAMMA)*(perimeter-length)

def compute_manifold_distance( tseq, sseq, dof, perimeter, visualise ):
    """ Function that projects the warped source sequence onto the target sequence
        and computes the distance between projections """
#==============================================================================
#   Compute centre of target image  
#==============================================================================
    centre = np.dot( np.array( tseq[0].GetDirection() ).reshape((3,3)),\
        (np.array(tseq[0].GetSize())-1)*tseq[0].GetSpacing()/2.0 )
    centre += tseq[0].GetOrigin()
#==============================================================================
#   Set up the transform      
#==============================================================================
    etransf = sitk.Transform( 3, sitk.sitkEuler )
    etransf.SetFixedParameters( centre )
    etransf.SetParameters( dof )
#==============================================================================
#   Warp source sequence given the input dof file   
#==============================================================================
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage( tseq[0] )
    resampler.SetInterpolator( sitk.sitkLinear ) 
    resampler.SetTransform( etransf )
    wseq = []
    for s in sseq:
        wseq.append( resampler.Execute( s ) )
#==============================================================================
#   Compute overlapping region  
#==============================================================================
    mask, overlap = get_overlapping_mask( tseq[0], wseq[0] )
    if overlap:
        distance = get_pca_distance( tseq, wseq, mask, perimeter, visualise )        
    else:
        distance = MAX_DIST
    del etransf, resampler, wseq, mask, overlap
    return distance
    
def optimise_dofs( target_sequence, source_sequence, perimeter, verbose ):
    """ Function to find optimal DOFs """
    dof, dirn, step = np.zeros(N_DOF), 1, 10
#==============================================================================
#   Hill climbing optimisation
#==============================================================================
    std_dev = np.array([np.pi/450.0, np.pi/450.0, np.pi/450.0, .4, .4, .4])
    while (dirn!=-1 or step>0.1):
        incr_step = step*std_dev
        best_dist = compute_manifold_distance( target_sequence, source_sequence,\
            dof, perimeter, True )
        if verbose:
            print('Current dof estimate: {0}'.format(dof))
            print('Lowest manifold distance: {0}'.format( best_dist ))
        dirn, pom = -1, 0
        for n_dof in range( N_DOF ):
            dof[n_dof] += incr_step[n_dof]
            dist = compute_manifold_distance( target_sequence, source_sequence,\
                dof, perimeter, False )
            if dist<best_dist:
                dirn = n_dof
                pom = 1
            dof[n_dof] -= 2*incr_step[n_dof]
            dist = compute_manifold_distance( target_sequence, source_sequence,\
                dof, perimeter, False )
            if dist<best_dist:
                dirn = n_dof
                pom = -1
            dof[n_dof] += incr_step[n_dof]
        if dirn!=-1:
            dof[dirn] = dof[dirn]+pom*incr_step[dirn]
        else:
            step /= 2.0
    if verbose:
        print('Final dof estimate: {0}'.format(dof))
        print('Lowest manifold distance: {0}'.format( best_dist ))
    return dof

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
#==============================================================================
#   Warp source sequence given the input dof file   
#==============================================================================
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage( tref )
    resampler.SetInterpolator( sitk.sitkLinear )  # sitkBSpline
    resampler.SetTransform( etransf )
    os.sys.stdout.write('Writing sequence to folder\n')
    for name, image in zip( sfiles, sseq ):
        sitk.WriteImage( resampler.Execute( image ), '{0}{1}'.format(out_path, name) )
    return
    
def write_dof_to_file( dof, doffile, verbose ):
    """ Function to write DOF parameters to file """
    ndof = np.zeros([N_DOF,3])
    ndof[3:,2] = (180/np.pi)*dof[:3]
    ndof[:3,2] = dof[3:]
    if verbose:
        os.sys.stdout.write('Final DOF: {0}\n'.format(ndof))
    os.sys.stdout.write('Writing DOF to file\n')
    np.savetxt( doffile, ndof, fmt='%.2f', delimiter='\t', header='DOF: 6',\
        comments='' )
    return
    
def main(argv):
    """ Check input arguments """
    argc = len(argv)
    if argc < 11:
        return usage(argv[0])
    else:
        target_dir, source_dir, doffile, out_path, otime, verbose = read_inputs(argv, argc)
#==============================================================================
#   Read names of input target and source sequences        
#==============================================================================
    target_files = sorted(os.listdir( target_dir ))
    source_files = sorted(os.listdir( source_dir ))
    if verbose:
        os.sys.stdout.write('Reading target sequence from {0}\n'.format(target_dir))
        os.sys.stdout.write('Reading source sequence from {0}\n'.format(source_dir))
#==============================================================================
#   If images are metadata one file is for metadata and one for raw
#==============================================================================
    extension = target_files[0].split('.')[1]
    if extension == 'mhd' or extension=='mha':
        target_files = target_files[::2]
        source_files = source_files[::2]
#==============================================================================
#   Read and resample target sequence to a 2x2x2 isotropic volume  
#==============================================================================
    target_sequence = []
    resampler = sitk.ResampleImageFilter()
    target_reference = sitk.ReadImage('{0}{1}'.format( target_dir, target_files[0] ))
    resampled_info, original_info = compute_info( target_reference )
    smoother = sitk.SmoothingRecursiveGaussianImageFilter()
    smoother.SetSigma( _SIGMA )
    for n, f in enumerate( target_files ):
        us_image = sitk.ReadImage('{0}{1}'.format( target_dir, f ))
#==============================================================================
#       Smooth images
#==============================================================================
        smus_image = smoother.Execute( us_image )
        res_us_image = resampler.Execute( smus_image, resampled_info['Size'],\
            sitk.Transform(), sitk.sitkBSpline, resampled_info['Origin'],\
            resampled_info['Spacing'], resampled_info['Direction'], int(), sitk.sitkInt16 )
        target_sequence.append( res_us_image )
    perimeter = get_length_from_list( target_sequence )
#==============================================================================
#   Read and resample source sequence to a 2x2x2 isotropic volume 
#==============================================================================
    source_sequence = []
    osource_sequence = []
    source_reference = sitk.ReadImage('{0}{1}'.format( source_dir, source_files[0] ))
    resampled_info, original_info = compute_info( source_reference )
    for n, f in enumerate( source_files ):
        us_image = sitk.ReadImage('{0}{1}'.format( source_dir, f ))
        osource_sequence.append( us_image )
#==============================================================================
#       Smooth images
#==============================================================================
        smus_image = smoother.Execute( us_image )
        res_us_image = resampler.Execute( smus_image, resampled_info['Size'],\
            sitk.Transform(), sitk.sitkBSpline, resampled_info['Origin'],\
            resampled_info['Spacing'], resampled_info['Direction'], int(), sitk.sitkInt16 )
        source_sequence.append( res_us_image )
#==============================================================================
#   Reduce sequence  
#==============================================================================
    start_time = time.time()
    dof = optimise_dofs( target_sequence, source_sequence, perimeter, verbose )
    stop_time = time.time()
    write_registered_sequence( source_files, target_reference, osource_sequence,\
        dof, out_path )
    write_dof_to_file( dof, doffile, verbose )
    np.savetxt( otime, [stop_time-start_time], fmt='%.1f' )
    
if __name__ == "__main__":
    os.sys.exit(main(os.sys.argv))
