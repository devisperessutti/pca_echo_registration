# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 14:43:17 2014

@author: dp11
"""

from __future__ import print_function
from scipy.optimize import fmin
import numpy as np
import SimpleITK as sitk

n_dim = 3
deg2rad = np.pi/180.0
rad2deg = 180/np.pi
scale_t = 30000
scale_r = 400
n_dof = 6

def check_fov( s_vc, t_size ):
    if min(s_vc)<0 or s_vc[0]>=t_size[0] or s_vc[1]>=t_size[1] or s_vc[2]>=t_size[2]:
        return False
    else:
        return True

def dof2tform( dof ):
    """ This function computes the rotation matrix given the rotation angles.
        Angles are in degrees. """
    rx, ry, rz = np.identity(n_dim), np.identity(n_dim), np.identity(n_dim)
    tx, ty, tz, a, b, c = dof[0], dof[1], dof[2], dof[3]*deg2rad,\
        dof[4]*deg2rad, dof[5]*deg2rad 
    ca, sa, cb, sb, cc, sc = np.cos( a ), np.sin( a ), np.cos( b ),\
        np.sin( b ), np.cos( c ), np.sin( c ),
    rx[1,1], rx[1,2], rx[2,1], rx[2,2] = ca, sa, -sa, ca
    ry[0,0], ry[0,2], ry[2,0], ry[2,2] = cb, -sb, sb, cb
    rz[0,0], rz[0,1], rz[1,0], rz[1,1] = cc, sc, -sc, cc
    mat = np.dot( rx, np.dot(ry, rz) )
    t = [tx, ty, tz]
    tform = {'rotation':mat, 'translation':t}
    return tform
    
def SetRotation( a, b, c ):
    """ Function that returns the rotation matrix corrensponding to the input 
        angles. Angles are in radiants. """
    rx, ry, rz = np.identity(n_dim), np.identity(n_dim), np.identity(n_dim)
    ca, sa, cb, sb, cc, sc = np.cos( a ), np.sin( a ), np.cos( b ),\
        np.sin( b ), np.cos( c ), np.sin( c ),
    rx[1,1], rx[1,2], rx[2,1], rx[2,2] = ca, sa, -sa, ca
    ry[0,0], ry[0,2], ry[2,0], ry[2,2] = cb, -sb, sb, cb
    rz[0,0], rz[0,1], rz[1,0], rz[1,1] = cc, sc, -sc, cc
    mat = np.dot( rx, np.dot(ry, rz) )
    return mat
    
def mat2dof( mat, invert=False ):
    """ This function computes the dof given the rotation matrix. 
    The matrix is in homogeneous coordinates 4x4. Default inverse is False """
    M, Ipm, Zrot, dof = np.eye( n_dim+1 ), np.eye( n_dim ), np.eye( n_dim ), np.zeros( 12 )   
    if invert:
        M = np.linalg.inv( mat );
    else:
        M = mat
    subM = M[0:n_dim,0:n_dim]
#==============================================================================
#   QR decomposition
#==============================================================================
    Q, R = np.linalg.qr( subM )
#==============================================================================
#   Find negative values on diagonal
#==============================================================================
    for i in range( n_dim ):
        if R[i,i] < 0:
            Ipm[i,i] = -1
#==============================================================================
#   Correct the value of Q and R
#==============================================================================
    R, Q = np.dot( Ipm, R ), np.dot( Q, Ipm )
#==============================================================================
#   Find rotation matrix
#==============================================================================
    theta = - np.arctan( ( R[0,1]/R[1,1] ) )
    Zrot = SetRotation( 0, 0, theta )
    R = np.dot( Zrot, R )
    iZrot = np.linalg.inv( Zrot )
    Q = np.dot( Q, iZrot )
#==============================================================================
#   Get angles
#==============================================================================
    tmp = np.cos( np.arcsin( -Q[0,2] ) )
    dof[3], dof[4], dof[5] = np.arctan2( Q[1,2]/tmp, Q[2,2]/tmp )*rad2deg, np.arcsin( -Q[0,2] )*rad2deg,\
        np.arctan2( Q[0,1]/tmp, Q[0,0]/tmp )*rad2deg
#==============================================================================
#   Get translations
#==============================================================================
    dof[0], dof[1], dof[2] = M[0,n_dim], M[1,n_dim], M[2,n_dim]
#==============================================================================
#   Get scaling
#==============================================================================
    dof[6], dof[7], dof[8] = R[0,0], R[1,1], R[2,2]
#==============================================================================
#   Get Skews
#==============================================================================
    dof[9], dof[10], dof[11] = rad2deg*np.arctan( R[0,1]/dof[6] ), rad2deg*np.arctan( R[1,2]/dof[7] ), \
        rad2deg*np.arctan( R[0,2]/dof[6] )
#==============================================================================
#   Return dof
#==============================================================================
    return dof

def compute_ssd( dof, target, source ):
    """ Cost function for the minimisation.
        It is the Sum of Squred Differences between the target and the warped source """
#==============================================================================
#   Read parameters and scale them appropriately
#==============================================================================
    tx, ty, tz, a, b, c = scale_t*dof[0], scale_t*dof[1], scale_t*dof[2], scale_r*dof[3],\
        scale_r*dof[4], scale_r*dof[5]
#==============================================================================
#   Set up rigid body transformation matrix (Maxima script)
#==============================================================================
    M = np.identity(4)
    rx, ry, rz = np.identity(n_dim+1), np.identity(n_dim+1), np.identity(n_dim+1)
    ca, sa, cb, sb, cc, sc = np.cos( a ), np.sin( a ), np.cos( b ),\
        np.sin( b ), np.cos( c ), np.sin( c ),
    rx[1,1], rx[1,2], rx[2,1], rx[2,2] = ca, sa, -sa, ca
    ry[0,0], ry[0,2], ry[2,0], ry[2,2] = cb, -sb, sb, cb
    rz[0,0], rz[0,1], rz[1,0], rz[1,1] = cc, sc, -sc, cc
    M = np.dot( rx, np.dot(ry, rz) )
    M[0,n_dim], M[1,n_dim], M[2,n_dim] = tx, ty, tz
#==============================================================================
#   Get image intensities in target and corresponding world physical coordinates
#==============================================================================
    t_size = target.GetSize()
    t_pts, t_values = np.ones((np.prod(t_size), 4)), np.zeros(np.prod(t_size))
    idx = 0
    for z in range(t_size[-1]):
        for y in range(t_size[1]):
            for x in range(t_size[0]):
                t_values[idx] = target.GetPixel( x, y, z )
                t_pts[idx,0:3] = target.TransformIndexToPhysicalPoint( [x, y, z] )
                idx += 1
#==============================================================================
#   Warp world coordinates given the rigid body matrix
#==============================================================================
    s_pts = np.dot( M, t_pts.T ).T
    t_values_overlap, s_values_overlap = [], []
#==============================================================================
#   Retrieve values in source
#==============================================================================
    for idx in range(np.prod(t_size)):
        s_vc = source.TransformPhysicalPointToIndex( s_pts[idx,0:3] )
#==============================================================================
#       Consider values on overlapping FoV only
#==============================================================================
        if check_fov( s_vc, t_size ):
            s_values_overlap.append( source.GetPixel( s_vc[0], s_vc[1], s_vc[2] ) )
            t_values_overlap.append( t_values[idx] )
    ssd = sum( (np.array(s_values_overlap)-np.array(t_values_overlap))**2 )
    return ssd    

def rigid_registration( target, source ):
    """ This function registers source to target optimising a rigid body transformation. 
        The output are the 6 degrees of freedom. The minimisation uses a downhill simplex
        algorithm """
    params = np.zeros(n_dof)
    # Resample source to target image
    res, res_source = sitk.ResampleImageFilter(), sitk.Image()
    res.SetReferenceImage( target )
    res.SetInterpolator( sitk.sitkNearestNeighbor )
    res_source = res.Execute( source )
    dof0 = np.zeros(6)
    dof = fmin(compute_ssd, dof0, args=(target, res_source), maxiter=200)
    params[0],params[1],params[2],params[3],params[4],params[5] = scale_t*dof[0], \
        scale_t*dof[1], scale_t*dof[2], scale_r*dof[3], scale_r*dof[4], scale_r*dof[5]
    print("Final parameters: {}".format(params))
    return dof2tform(params)
    
# For future versions of sitk
#def euler_ssd_linear( target, source ):
#    R = sitk.ImageRegistrationMethod()
#    R.SetMetricAsMeanSquares()
#    R.SetOptimizerAsRegularStepGradientDescent( 4.0, .01, 200 )
#    R.SetTransform( sitk.Transform( target.GetDimension(), sitk.sitkEuler ))
#    R.SetInterpolator(sitk.sitkLinear)
#
#    outTx = R.Execute( target, source )
#    return outTx
#    
#def euler_ssd_nn( target, source ):
#    R = sitk.ImageRegistrationMethod()
#    R.SetMetricAsMeanSquares()
#    R.SetOptimizerAsRegularStepGradientDescent( 4.0, .01, 200 )
#    R.SetTransform( sitk.Transform( target.GetDimension(), sitk.sitkEuler ) )
#    R.SetInterpolator(sitk.sitkNearestNeighbor )
#
#    outTx = R.Execute( target, source )
#    return outTx
    