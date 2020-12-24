#!/usr/bin/env python

from paraview.simple import *
import os.path

# find the curent directory
current_dir = os.getcwd()

for file in os.listdir(current_dir):
    if file.endswith(".vtk"):
	    r = LegacyVTKReader()
	    r.FileNames = [file]
	    w = servermanager.writers.DataSetWriter()
	    w.FileType = 'Ascii'
	    w.Input = r
	    w.FileName = file
	    w.UpdatePipeline()
