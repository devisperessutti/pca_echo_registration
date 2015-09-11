# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 11:36:34 2014

@author: dp11
"""
import SimpleITK as sitk
import numpy as np

def rescale_intensity_from_filenames( input_filename, output_filename, max_value ):
    """ This function applies the RescaleIntensityImage filter to the input image """
    reader, image, rescaler, writer = sitk.ImageFileReader(), sitk.Image(), \
        sitk.RescaleIntensityImageFilter(), sitk.ImageFileWriter()
    reader.SetFileName( input_filename )
    image = reader.Execute()
    rescaler.SetOutputMaximum( max_value )
    rescaler.SetOutputMinimum( 0 )
    writer.SetFileName( output_filename )
    writer.Execute( rescaler.Execute( image ) )
    del reader, image, rescaler, writer
    return 
    
def resample_from_filenames( input_filename, reference_filename, output_filename ):
    """ This function resamples the input image to the reference image """
    reader, reader_ref, image, ref, resampler, writer = sitk.ImageFileReader(), sitk.ImageFileReader(),\
        sitk.Image(), sitk.Image(), sitk.ResampleImageFilter(), sitk.ImageFileWriter()
    reader.SetFileName( input_filename )
    reader_ref.SetFileName( reference_filename )
    image = reader.Execute()
    ref = reader_ref.Execute()
    resampler.SetReferenceImage( ref )
    resampler.SetInterpolator( sitk.sitkNearestNeighbor )
    writer.SetFileName( output_filename )
    writer.Execute( resampler.Execute( image ) )
    del reader, reader_ref, ref, image, resampler, writer
    return 
    
def resample_from_spacing( input_filename, output_filename, spacing ):
    """ This function resamples the input image to the reference image """
    reader, image, resampler, writer = sitk.ImageFileReader(), sitk.Image(),\
        sitk.ResampleImageFilter(), sitk.ImageFileWriter()
    reader.SetFileName( input_filename )
    image = reader.Execute()
    old_orig, old_size, old_spac = image.GetOrigin(), image.GetSize(), image.GetSpacing()
    new_size, new_spac, new_orig  = [], [], old_orig
    for i in range(len(old_size)): 
        new_size.append( int(np.ceil( 1.0*(old_size[i]-1)*old_spac[i]/spacing[i] ) + 1) ) 
    for i in range(len(new_size)):
        new_spac.append( 1.0*old_spac[i]*old_size[i]/new_size[i] )
    resampler.SetInterpolator( sitk.sitkNearestNeighbor )
    resampled = resampler.Execute( image, new_size, sitk.Transform(), sitk.sitkBSpline, \
        new_orig, new_spac, image.GetDirection(), long(), sitk.sitkInt16 )
    writer.SetFileName( output_filename )
    writer.Execute( resampled )
    del reader, image, resampler, resampled, writer
    return 
    
def rescale_window_intensity_from_filenames( input_filename, output_filename, min_value, max_value ):
    """ This function applies the IntensityWindowingImage filter to the input image """
    reader, image, rescaler, writer = sitk.ImageFileReader(), sitk.Image(), \
        sitk.IntensityWindowingImageFilter(), sitk.ImageFileWriter()
    reader.SetFileName( input_filename )
    image = reader.Execute()
    nda = sitk.GetArrayFromImage( image )
    rescaler.SetWindowMaximum( np.percentile( nda, 99.9 ) )
    rescaler.SetWindowMinimum( np.percentile( nda, 0.1 ) )
    rescaler.SetOutputMaximum( max_value )
    rescaler.SetOutputMinimum( min_value )
    writer.SetFileName( output_filename )
    writer.Execute( rescaler.Execute( image ) )
    del reader, image, rescaler, writer
    return 
    
def rescale_window_intensity_from_images( image, min_value, max_value ):
    """ This function applies the IntensityWindowingImage filter to the input image """
    rescaler = sitk.IntensityWindowingImageFilter()
    nda = sitk.GetArrayFromImage( image )
    rescaler.SetWindowMaximum( np.percentile( nda, 99.9 ) )
    rescaler.SetWindowMinimum( np.percentile( nda, 0.1 ) )
    rescaler.SetOutputMaximum( max_value )
    rescaler.SetOutputMinimum( min_value )
    return rescaler.Execute( image )
    