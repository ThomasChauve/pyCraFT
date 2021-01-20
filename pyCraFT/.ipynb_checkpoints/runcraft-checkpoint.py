# -*- coding: utf-8 -*-
'''
.. py:module:: runcraft
Created on 30 sept. 2015
Toolbox to treat data from one run experiment using CraFT

@author: Thomas Chauve
@contact: thomas.chauve@lgge.obs.ujf-grenoble.fr
@license: CC-BY-CC
'''

import pyCraFT.image2d as im2d
import pyCraFT.symetricTensorMap as sTM
import pyCraFT.gammarun as GamR


import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import os
import time

class runcraft(object):
    '''
    The 'run' CraFT folder is build as follow :
    an 'output' folder where the output data from CraFT are store
    in the current folder you should find : *.in, *micro.vtk, *.phase, *.output, *.load  
    
    This toolbox need over toolboxs :
    
    :library: numpy
    :library: os
    :library: vtk
    :library: image2d
    :library: symetricTensorMap
    :library: gammarun
    
    '''
    pass
    
    def __init__(self,run_folder,time_data):
        '''
        Build data for one CraFT run using *micro.vtk, *.phase, and data within 'output' folder at time time_data
        
        :param run_folder: folder were the CraFT run is
        :type run_folder: str
        :param time_data: time write one the name file
        :type time_data: str
        :return: runcraft object
        :rtype: runcraft
        
        runcraft structure :
        
        :element: *.grainID : is an 'image2d' with one value per grains
        :element: *.orientation : is an 'array' with 4 column (id,phi1,phi,phi2)
        :element: *.strain : is an 'symetricTensorMap' for the time time_data 
        :element: *.stress : is an 'symetricTensorMap' for the time time_data
        '''
        
        # set up the vtk reader     
        reader = vtk.vtkDataSetReader() 
        # read the current folder
        current=os.getcwd()
        
        ######################
        # load microstrucure #
        ######################
        os.chdir(run_folder)
        # find name of the files
        data_name=os.listdir()
        # get the microstructure
        # find 'vtk' file
        for k in range(len(data_name)):
                if (data_name[k][len(data_name[k])-4:]=='.vtk'):
                    nb=k
        # set up the name of the file wanted    
        reader.SetFileName(data_name[nb])
        # read the file
        reader.Update()
        # read the file output
        ug  = reader.GetOutput()
        # extract the resolution
        res=ug.GetSpacing()[0]
        # extract the matrix representing the map of this scalar
        map=vtk_to_numpy(ug.GetPointData().GetScalars()).reshape((ug.GetDimensions()[0:2][::-1]))
        
        self.grainId=im2d.image2d(map[::-1],res)
        
        ####################
        # load orientation #
        ####################
        # find 'phase' file
        for k in range(len(data_name)):
                if (data_name[k][len(data_name[k])-6:]=='.phase'):
                    nb=k
        self.orientation=np.transpose(np.loadtxt(data_name[nb],unpack=True, skiprows=9, usecols=(0,2,3,4),dtype='f,f,f,f'))
                  
                    
        # set data to extract
        # line with gamma
        wanted_data=[time_data+'_strain11.vtk',time_data+'_strain22.vtk',time_data+'_strain33.vtk',time_data+'_strain12.vtk',time_data+'_strain13.vtk',time_data+'_strain23.vtk',time_data+'_stress11.vtk',time_data+'_stress22.vtk',time_data+'_stress33.vtk',time_data+'_stress12.vtk',time_data+'_stress13.vtk',time_data+'_stress23.vtk',time_data+'_gamma01.vtk',time_data+'_gamma02.vtk',time_data+'_gamma03.vtk',time_data+'_gamma04.vtk',time_data+'_gamma05.vtk',time_data+'_gamma06.vtk',time_data+'_gamma07.vtk',time_data+'_gamma08.vtk',time_data+'_gamma09.vtk',time_data+'_gamma10.vtk',time_data+'_gamma11.vtk',time_data+'_gamma12.vtk',time_data+'_rotation.vtk']
             
        ###########################
        # load strain/stress data #
        ###########################
        tmp='output/'
        os.chdir(tmp)
        # split data for stress, strain
        tmp_split='/home/chauvet/Documents/Rheolef/CraFT/craft_1.1.0/bin/vtk_split *'+time_data+'_gamma.vtk'
        os.system(tmp_split)
        tmp_split='/home/chauvet/Documents/Rheolef/CraFT/craft_1.1.0/bin/vtk_split *'+time_data+'_strain.vtk'
        os.system(tmp_split)
        tmp_split='/home/chauvet/Documents/Rheolef/CraFT/craft_1.1.0/bin/vtk_split *'+time_data+'_stress.vtk'
        os.system(tmp_split)
            
        # find data name
        data_name=os.listdir()
        # create self
        tmp_strain=list(np.arange(6))
        tmp_stress=list(np.arange(6))
        tmp_gamma=list(np.arange(12))
        for j in range(len(wanted_data)):
            # find the name of the txt file wanted
            for k in range(len(data_name)):
                if (data_name[k][len(data_name[k])-len(wanted_data[j]):]==wanted_data[j]):
                    nb=k
            # set up the name of the file wanted    
            reader.SetFileName(data_name[nb])
            # read the file
            reader.Update()
            # read the file output
            ug  = reader.GetOutput()
            # extract the resolution
            res=ug.GetSpacing()[0]
            dim=ug.GetDimensions()[0:2][::-1]
            
                
            # make the difference between strain and stress
            if j in range(6):
                # extract the matrix representing the map of this scalar
                map=vtk_to_numpy(ug.GetPointData().GetScalars()).reshape((ug.GetDimensions()[0:2][::-1]))
                tmp_strain[j]=im2d.image2d(map[::-1],res) 
            elif j in range(12):
                # extract the matrix representing the map of this scalar
                map=vtk_to_numpy(ug.GetPointData().GetScalars()).reshape((ug.GetDimensions()[0:2][::-1]))
                tmp_stress[j-6]=im2d.image2d(map[::-1],res)
            elif j in range(24):
                # extract the matrix representing the map of this scalar
                map=vtk_to_numpy(ug.GetPointData().GetScalars()).reshape((ug.GetDimensions()[0:2][::-1]))
                tmp_gamma[j-12]=im2d.image2d(map[::-1],res)
            elif j in range(25):
                pd = ug.GetPointData()
                tmp_rotation=vtk_to_numpy(pd.GetArray(0)).reshape([dim[0],dim[1],3])
            else:
                print('error 01 : too much map')
        
        # Build symetricTensorMap for strain and stress            
        self.strain=sTM.symetricTensorMap(tmp_strain[0],tmp_strain[1],tmp_strain[2],tmp_strain[3],tmp_strain[4],tmp_strain[5])
        self.stress=sTM.symetricTensorMap(tmp_stress[0],tmp_stress[1],tmp_stress[2],tmp_stress[3],tmp_stress[4],tmp_stress[5])
        # Build gamma map
        self.gamma=GamR.gammarun(tmp_gamma[0],tmp_gamma[1],tmp_gamma[2],tmp_gamma[3],tmp_gamma[4],tmp_gamma[5],tmp_gamma[6],tmp_gamma[7],tmp_gamma[8],tmp_gamma[9],tmp_gamma[10],tmp_gamma[11])
        
        
        
        self.rotation=tmp_rotation

        # remove split file
        os.system('rm *strain1*.vtk *strain2*.vtk *stress1*.vtk *stress2*.vtk *33.vtk *gamma0*.vtk *gamma1*.vtk *rotation1.vtk *rotation2.vtk *rotation3.vtk' )
            
        # return in the current folder
        os.chdir(current)
        
        return
    
    def strain_energy(self):
        '''
        Strain energy map compute as : w=e_ij*s_ij (Einstein notation)
        
        :return: w strain energy map
        :rtype: im2d.image2d
        '''
        
        tmp=self.strain*self.stress
        
        return (tmp.t11+tmp.t22+tmp.t33+(tmp.t12+tmp.t13+tmp.t23)*2.)*(1./2.)
    
    def elstrain(self,mask=0):
        '''
        :return: elastique strain sigma_{ij}*E_{ijkl}*sigma_{kl}
        :rtype: im2d.image2d
        
        '''
        # L is writen is MPa as in CraFT OC2 law
        L=np.matrix([[ 13930.,7082.,5765.,0.,0.,0.],[7082.,13930.,5765.,0.,0.,0.],[5765.,5765.,15010.,0.,0.,0.],[0.,0.,0.,6028.,0.,0.],[0.,0.,0.,0.,6028.,0.],[0.,0.,0.,0.,0.,6848.]])
        E=np.linalg.inv(L)
        
        [ny,nx]=self.grainId.field.shape
        
        elstrain=im2d.image2d(np.zeros([ny,nx]),self.grainId.res)
        #print(time.strftime('%d/%m/%y %H:%M:%S',time.localtime()))  
        
        if (type(mask)==im2d.mask2d):
            idy,idx=np.where(mask.field==1)
        else:
            idy,idx=np.where(self.grainId.field<2.*np.max(self.grainId.field))

        
        for i in list(range(len(idx))):
            # extract localisation of the grain i
            id_grain=self.grainId.extract_data(pos=np.int32(np.array([[idx[i],idy[i]]])))
            # find the euler angle of grain i
            id_ori=np.where(self.orientation[:,0]==id_grain[0])
            ori=self.orientation[id_ori,1:4]
            phi1=ori[0,0,0]
            phi=ori[0,0,1]
            phi2=ori[0,0,2]
            P1=np.matrix([[np.cos(phi1),-np.sin(phi1),0],[np.sin(phi1),np.cos(phi1),0],[0,0,1]])
            P2=np.matrix([[1,0,0],[0,np.cos(phi),-np.sin(phi)],[0,np.sin(phi),np.cos(phi)]])
            P3=np.matrix([[np.cos(phi2),-np.sin(phi2),0],[np.sin(phi2),np.cos(phi2),0],[0,0,1]])
            # matrix
            P=P3*P2*P1
            # extract sigma data
            [sigma,tmpid]=self.stress.extract_data(pos=np.int32(np.array([[idx[i],idy[i]]])))
            # local basis for sigma
            sigmaloc=np.linalg.inv(P)*sigma*P
            # sigma in Kelvin notation
            sigma_Kelvin=np.matrix([[sigmaloc[0,0]],[sigmaloc[1,1]],[sigmaloc[2,2]],[np.sqrt(2.)*sigmaloc[1,2]],[np.sqrt(2.)*sigmaloc[0,2]],[np.sqrt(2.)*sigmaloc[0,1]]])
                
            elstrain.field[idy[i],idx[i]]=0.5*np.transpose(sigma_Kelvin)*E*sigma_Kelvin
         
        #print(time.strftime('%d/%m/%y %H:%M:%S',time.localtime()))      
        return elstrain