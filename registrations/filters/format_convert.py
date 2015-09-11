# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 09:13:36 2014

@author: dp11
"""

import SimpleITK as sitk
import sys

def read_write( file_in, file_out ):
    image = sitk.Image()
    reader = sitk.ImageFileReader()
    reader.SetFileName( file_in )        
    image = reader.Execute()
        
    writer = sitk.ImageFileWriter()
    writer.SetFileName ( file_out )
    writer.Execute( image )
    sys.stdout.write("\nImage {0} converted to {1}\n".format( file_in, file_out ))