# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 15:09:49 2014

@author: dp11
"""

import vtk
import numpy as np
import SimpleITK as sitk

def ImportArrayToVTKImage( dataImporter, image, array, origin):
    array_string = array.tostring()
    # Import to vtk
    size = image.GetSize()
    dataImporter.CopyImportVoidPointer(array_string, len(array_string))
    dataImporter.SetDataScalarTypeToUnsignedShort()
    dataImporter.SetNumberOfScalarComponents( 1 )
    dataImporter.SetDataSpacing( image.GetSpacing() )
    dataImporter.SetDataOrigin( origin )
    dataImporter.SetWholeExtent(0, size[0]-1, 0, size[1]-1, 0, size[2]-1)
    dataImporter.SetDataExtent(0, size[0]-1, 0, size[1]-1, 0, size[2]-1)
    dataImporter.Update()  
    return dataImporter.GetOutput()
    
def ImportArrayToVTKVectorImage(dataImporter, image, array):
    array_string = array.tostring()
    # Import to vtk
    size = image.GetSize()
    dataImporter.CopyImportVoidPointer(array_string, len(array_string))
    dataImporter.SetDataScalarTypeToFloat64()
    dataImporter.SetNumberOfScalarComponents( 3 )
    dataImporter.SetDataSpacing( image.GetSpacing() )
    dataImporter.SetDataOrigin( image.GetOrigin() )
    dataImporter.SetWholeExtent(0, size[0]-1, 0, size[1]-1, 0, size[2]-1)
    dataImporter.SetDataExtent(0, size[0]-1, 0, size[1]-1, 0, size[2]-1)
    dataImporter.Update()  
    
def ExportVTKImageToArray( image ):
    export = vtk.vtkImageExport()
    if vtk.VTK_MAJOR_VERSION <= 5:
        export.SetInput( image )
    else:
        export.SetInputData( image )
    nCmps = image.GetNumberOfScalarComponents()
    dim = image.GetDimensions()
    # The underlying assumption is that the type is vtk.VTK_UNSIGNED_CHAR
    size = dim[0]*dim[1]*dim[2]*nCmps
    imString = np.zeros((size,), np.uint8).tostring()
    export.Export( imString )
    array = np.fromstring( imString, np.uint8 )
    del imString
    # reshape array appropriately.
    array = np.reshape(array, [dim[2], dim[1], dim[0]])
    return array
    
def import_sitk_image_to_vtk( image_filename ):
    """ This function imports a Simple ITK image as a VTK image data """
#==============================================================================
#   Read image 
#==============================================================================
    image = sitk.Image();
    reader = sitk.ImageFileReader()
    reader.SetFileName( image_filename )   
#==============================================================================
#   Rescale to 0-255
#==============================================================================
    rescaler = sitk.IntensityWindowingImageFilter()
    rescaler.SetWindowMaximum( np.percentile( sitk.GetArrayFromImage( reader.Execute() ), 99.9 ) )
    rescaler.SetWindowMinimum( 0 )
    rescaler.SetOutputMaximum( 255 )
    rescaler.SetOutputMinimum( 0 )
#==============================================================================
#   Cast the type to unsigned int
#==============================================================================
    caster = sitk.CastImageFilter()
    caster.SetOutputPixelType( sitk.sitkUInt8 )
#==============================================================================
#   Execute and assing to an sitk.Image
#==============================================================================
    image = caster.Execute( rescaler.Execute( reader.Execute() ) )
#==============================================================================
#   Get image information
#==============================================================================
    size        = np.array( image.GetSize() )
    spacing     = np.array( image.GetSpacing() )
    origin      = np.zeros( 3 )         # Image origin is set to axes origin
    direction   = np.array( image.GetDirection() )
#==============================================================================
#   Convert the image to numpy array
#==============================================================================
    nda = sitk.GetArrayFromImage( image )
    nda = np.require( nda, dtype=np.uint16 ) 
    return nda, image, size, spacing, origin, direction