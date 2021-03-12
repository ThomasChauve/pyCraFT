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

import image3d.image3d as im3d

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import os
import time
import scipy
import math
import skimage.segmentation as sks
import skimage.morphology as skm
from tqdm.notebook import tqdm

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
        #current=os.getcwd()
        
        ######################
        # load microstrucure #
        ######################
        #os.chdir(run_folder)
        # find name of the files
        data_name=os.listdir(run_folder)
        # get the microstructure
        # find 'vtk' file
        for k in range(len(data_name)):
                if (data_name[k][len(data_name[k])-4:]=='.vtk'):
                    nb=k
        # set up the name of the file wanted    
        reader.SetFileName(run_folder+data_name[nb])
        # read the file
        reader.Update()
        # read the file output
        ug  = reader.GetOutput()
        # extract the resolution
        res=ug.GetSpacing()[0]
        # extract the matrix representing the map of this scalar
        ugdim=ug.GetDimensions()
        
        if ugdim[2]>1:
            dim_im=3
            fim=im3d.image3d
        elif ugdim[2]==1:
            dim_im=2
            fim=im2d.image2d
        
        map=vtk_to_numpy(ug.GetPointData().GetScalars()).reshape((ug.GetDimensions()[0:dim_im][::-1]))
        
        self.grainId=fim(map[::-1],res)
        
        ####################
        # load orientation #
        ####################
        # find 'phase' file
        for k in range(len(data_name)):
                if (data_name[k][len(data_name[k])-6:]=='.phase'):
                    nb=k
        self.orientation=np.transpose(np.loadtxt(run_folder+data_name[nb],unpack=True, skiprows=9, usecols=(0,2,3,4),dtype='f,f,f,f'))
                  
                    
        # set data to extract
        # line with gamma
        wanted_data=[time_data+'_strain11.vtk',time_data+'_strain22.vtk',time_data+'_strain33.vtk',time_data+'_strain12.vtk',time_data+'_strain13.vtk',time_data+'_strain23.vtk',time_data+'_stress11.vtk',time_data+'_stress22.vtk',time_data+'_stress33.vtk',time_data+'_stress12.vtk',time_data+'_stress13.vtk',time_data+'_stress23.vtk',time_data+'_gamma01.vtk',time_data+'_gamma02.vtk',time_data+'_gamma03.vtk',time_data+'_gamma04.vtk',time_data+'_gamma05.vtk',time_data+'_gamma06.vtk',time_data+'_gamma07.vtk',time_data+'_gamma08.vtk',time_data+'_gamma09.vtk',time_data+'_gamma10.vtk',time_data+'_gamma11.vtk',time_data+'_gamma12.vtk',time_data+'_rotation.vtk']
             
        ###########################
        # load strain/stress data #
        ###########################
        tmp=run_folder+'output/'
        #os.chdir(tmp)
        # split data for stress, strain
        tmp_split='/home/chauvet/Documents/Rheolef/CraFT/craft_1.1.0/bin/vtk_split '+tmp+'*'+time_data+'_gamma.vtk'
        os.system(tmp_split)
        tmp_split='/home/chauvet/Documents/Rheolef/CraFT/craft_1.1.0/bin/vtk_split '+tmp+'*'+time_data+'_strain.vtk'
        os.system(tmp_split)
        tmp_split='/home/chauvet/Documents/Rheolef/CraFT/craft_1.1.0/bin/vtk_split '+tmp+'*'+time_data+'_stress.vtk'
        os.system(tmp_split)
            
        # find data name
        data_name=os.listdir(tmp)
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
            reader.SetFileName(tmp+data_name[nb])
            # read the file
            reader.Update()
            # read the file output
            ug  = reader.GetOutput()
            # extract the resolution
            res=ug.GetSpacing()[0]
            dim=ug.GetDimensions()[0:dim_im][::-1]
            
                
            # make the difference between strain and stress
            if j in range(6):
                # extract the matrix representing the map of this scalar
                map=vtk_to_numpy(ug.GetPointData().GetScalars()).reshape((ug.GetDimensions()[0:dim_im][::-1]))
                tmp_strain[j]=fim(map[::-1],res) 
            elif j in range(12):
                # extract the matrix representing the map of this scalar
                map=vtk_to_numpy(ug.GetPointData().GetScalars()).reshape((ug.GetDimensions()[0:dim_im][::-1]))
                tmp_stress[j-6]=fim(map[::-1],res)
            elif j in range(24):
                # extract the matrix representing the map of this scalar
                map=vtk_to_numpy(ug.GetPointData().GetScalars()).reshape((ug.GetDimensions()[0:dim_im][::-1]))
                tmp_gamma[j-12]=fim(map[::-1],res)
            elif j in range(25):
                pd = ug.GetPointData()
                if dim_im==2:
                    tmp_rotation=vtk_to_numpy(pd.GetArray(0)).reshape([dim[0],dim[1],3])
                elif dim_im==3:
                    tmp_rotation=vtk_to_numpy(pd.GetArray(0)).reshape([dim[0],dim[1],dim[2],3])
            else:
                print('error 01 : too much map')
        
        # Build symetricTensorMap for strain and stress            
        self.strain=sTM.symetricTensorMap(tmp_strain[0],tmp_strain[1],tmp_strain[2],tmp_strain[3],tmp_strain[4],tmp_strain[5])
        self.stress=sTM.symetricTensorMap(tmp_stress[0],tmp_stress[1],tmp_stress[2],tmp_stress[3],tmp_stress[4],tmp_stress[5])
        # Build gamma map
        self.gamma=GamR.gammarun(tmp_gamma[0],tmp_gamma[1],tmp_gamma[2],tmp_gamma[3],tmp_gamma[4],tmp_gamma[5],tmp_gamma[6],tmp_gamma[7],tmp_gamma[8],tmp_gamma[9],tmp_gamma[10],tmp_gamma[11])
        
        
        
        self.rotation=tmp_rotation

        # remove split file
        os.system('rm '+tmp+'*strain1*.vtk '+tmp+'*strain2*.vtk '+tmp+'*stress1*.vtk '+tmp+'*stress2*.vtk '+tmp+'*33.vtk '+tmp+'*gamma0*.vtk '+tmp+'*gamma1*.vtk '+tmp+'*rotation1.vtk '+tmp+'*rotation2.vtk '+tmp+'*rotation3.vtk' )
            
        # return in the current folder
        #os.chdir(current)
        
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
    
    def smooth(self,mat=0,boundary='symm', fillvalue=0):
        '''
        Performed smoothing using convolved 2D function
        :param mat: matrix for the convolution
        :type mat: np.array
        :param boundary: boundary : str {'fill', 'wrap', 'symm'}, optional
                         A flag indicating how to handle boundaries:
                        ``fill``
                        pad input arrays with fillvalue.
                        ``wrap``
                        circular boundary conditions.
                        ``symm``
                        symmetrical boundary conditions. (default) 
        type boundary: str
        :param fillvalue:  fillvalue : scalar, optional
                           Value to fill pad input arrays with. Default is 0.
        :return: imagescipy.signal.fftconvolve¶
        :rtype: runcraft
        '''
        if mat==0 and len(self.grainId.field.shape)==2:
            mat=1/13*np.array([[0,0,1,0,0],[0,1,1,1,0],[1,1,1,1,1],[0,1,1,1,0],[0,0,1,0,0]])
        elif mat==0 and len(self.grainId.field.shape)==3:
            mat=1/25*np.array([[[0,0,0,0,0],[0,0,0,0,0],[0,0,1,0,0],[0,0,0,0,0],[0,0,0,0,0]],[[0,0,0,0,0],[0,0,1,0,0],[0,1,1,1,0],[0,0,1,0,0],[0,0,0,0,0]],[[0,0,1,0,0],[0,1,1,1,0],[1,1,1,1,1],[0,1,1,1,0],[0,0,1,0,0]],[[0,0,0,0,0],[0,0,1,0,0],[0,1,1,1,0],[0,0,1,0,0],[0,0,0,0,0]],[[0,0,0,0,0],[0,0,0,0,0],[0,0,1,0,0],[0,0,0,0,0],[0,0,0,0,0]]])
        
        self.gamma.g01.map.smooth(mat=mat,boundary=boundary, fillvalue=fillvalue)
        self.gamma.g02.map.smooth(mat=mat,boundary=boundary, fillvalue=fillvalue)
        self.gamma.g03.map.smooth(mat=mat,boundary=boundary, fillvalue=fillvalue)
        self.gamma.g04.map.smooth(mat=mat,boundary=boundary, fillvalue=fillvalue)
        self.gamma.g05.map.smooth(mat=mat,boundary=boundary, fillvalue=fillvalue)
        self.gamma.g06.map.smooth(mat=mat,boundary=boundary, fillvalue=fillvalue)
        self.gamma.g07.map.smooth(mat=mat,boundary=boundary, fillvalue=fillvalue)
        self.gamma.g08.map.smooth(mat=mat,boundary=boundary, fillvalue=fillvalue)
        self.gamma.g09.map.smooth(mat=mat,boundary=boundary, fillvalue=fillvalue)
        self.gamma.g10.map.smooth(mat=mat,boundary=boundary, fillvalue=fillvalue)
        self.gamma.g11.map.smooth(mat=mat,boundary=boundary, fillvalue=fillvalue)
        self.gamma.g12.map.smooth(mat=mat,boundary=boundary, fillvalue=fillvalue)
    
        self.strain.t11.smooth(mat=mat,boundary=boundary, fillvalue=fillvalue)
        self.strain.t22.smooth(mat=mat,boundary=boundary, fillvalue=fillvalue)
        self.strain.t33.smooth(mat=mat,boundary=boundary, fillvalue=fillvalue)
        self.strain.t12.smooth(mat=mat,boundary=boundary, fillvalue=fillvalue)
        self.strain.t13.smooth(mat=mat,boundary=boundary, fillvalue=fillvalue)
        self.strain.t23.smooth(mat=mat,boundary=boundary, fillvalue=fillvalue)
        
        self.stress.t11.smooth(mat=mat,boundary=boundary, fillvalue=fillvalue)
        self.stress.t22.smooth(mat=mat,boundary=boundary, fillvalue=fillvalue)
        self.stress.t33.smooth(mat=mat,boundary=boundary, fillvalue=fillvalue)
        self.stress.t12.smooth(mat=mat,boundary=boundary, fillvalue=fillvalue)
        self.stress.t13.smooth(mat=mat,boundary=boundary, fillvalue=fillvalue)
        self.stress.t23.smooth(mat=mat,boundary=boundary, fillvalue=fillvalue)
        
    
    
    
    
    def outputIA(self,RX_image,name,listvar=[]):
        '''
        Export a csv file for IA input
        :param RX_image: Binary images for classification in IA
        :type run_folder: im2d.image2d
        :param listvar: name of the output
        :type time_data: str
        :param listvar: list of variable to export, The name availaible are given bellow
        :type time_data: list
        :return: csv file, name.csv
        
        .. note:: 'Strain_eqVonMises', 'Stress_eqVonMises','elStrain' (elastic strain), 'StrainEnergy','StrainComponant' (eij), 'StressComponant' (sij), 'allGamma' (gi), 'activityGamma', 'systemGamma' sum of gamma for each glide system (3 values), 'disttoTJ/GB', 'misAngle' (misorientation angle of the closest grain boundary), 'SchmidFactor', 'diffStress' (s1-s3), 'diffStrain' (e1-e3)
        '''
        # Add RX data
        IA_data=[]
        IA_name=[]
        IA_data.append(RX_image.field.flatten())
        IA_name.append('RX')
        ss=np.shape(RX_image.field)

        if 'Strain_eqVonMises' in listvar:
            IA_data.append(self.strain.eqVonMises().field.flatten())
            IA_name.append('eqStrain')

        if 'Stress_eqVonMises' in listvar:
            IA_data.append(self.stress.eqVonMises().field.flatten())
            IA_name.append('eqStress')

        if  'elStrain' in listvar :
            IA_data.append(self.elstrain().field.flatten())
            IA_name.append('elStrain')

        if 'StrainEnergy' in listvar:
            IA_data.append(self.strain_energy().field.flatten())
            IA_name.append('StrainEnergy')

        if 'StrainComponant' in listvar:
            IA_data.append(self.strain.t11.field.flatten())
            IA_name.append('exx')
            IA_data.append(self.strain.t22.field.flatten())
            IA_name.append('eyy')
            IA_data.append(self.strain.t33.field.flatten())
            IA_name.append('ezz')
            IA_data.append(self.strain.t12.field.flatten())
            IA_name.append('exy')
            IA_data.append(self.strain.t13.field.flatten())
            IA_name.append('exz')
            IA_data.append(self.strain.t23.field.flatten())
            IA_name.append('eyz')

        if 'StressComponant' in listvar:
            IA_data.append(self.stress.t11.field.flatten())
            IA_name.append('sxx')
            IA_data.append(self.stress.t22.field.flatten())
            IA_name.append('syy')
            IA_data.append(self.stress.t33.field.flatten())
            IA_name.append('szz')
            IA_data.append(self.stress.t12.field.flatten())
            IA_name.append('sxy')
            IA_data.append(self.stress.t13.field.flatten())
            IA_name.append('sxz')
            IA_data.append(self.stress.t23.field.flatten())
            IA_name.append('syz')

        if 'allGamma' in listvar:
            IA_data.append(self.gamma.g01.map.field.flatten())
            IA_name.append('g01')
            IA_data.append(self.gamma.g02.map.field.flatten())
            IA_name.append('g02')
            IA_data.append(self.gamma.g03.map.field.flatten())
            IA_name.append('g03')
            IA_data.append(self.gamma.g04.map.field.flatten())
            IA_name.append('g04')
            IA_data.append(self.gamma.g05.map.field.flatten())
            IA_name.append('g05')
            IA_data.append(self.gamma.g06.map.field.flatten())
            IA_name.append('g06')
            IA_data.append(self.gamma.g07.map.field.flatten())
            IA_name.append('g07')
            IA_data.append(self.gamma.g08.map.field.flatten())
            IA_name.append('g08')
            IA_data.append(self.gamma.g09.map.field.flatten())
            IA_name.append('g09')
            IA_data.append(self.gamma.g10.map.field.flatten())
            IA_name.append('g10')
            IA_data.append(self.gamma.g11.map.field.flatten())
            IA_name.append('g11')
            IA_data.append(self.gamma.g12.map.field.flatten())
            IA_name.append('g12')

        if 'activityGamma' in listvar:
            IA_data.append(self.gamma.gamma_activity(plane='ba').field.flatten())
            IA_name.append('A_ba')
            IA_data.append(self.gamma.gamma_activity(plane='pr').field.flatten())
            IA_name.append('A_pr')
            IA_data.append(self.gamma.gamma_activity(plane='py').field.flatten())
            IA_name.append('A_py')

        if 'systemGamma' in listvar:
            IA_data.append((self.gamma.g01.map.field.flatten()**2+self.gamma.g02.map.field.flatten()**2+self.gamma.g03.map.field.flatten()**2)**.5)
            IA_name.append('Sys_ba')
            IA_data.append((self.gamma.g04.map.field.flatten()**2+self.gamma.g05.map.field.flatten()**2+self.gamma.g06.map.field.flatten()**2)**.5)
            IA_name.append('Sys_pr')
            IA_data.append((self.gamma.g07.map.field.flatten()**2+self.gamma.g08.map.field.flatten()**2+self.gamma.g09.map.field.flatten()**2+self.gamma.g10.map.field.flatten()**2+self.gamma.g11.map.field.flatten()**2+self.gamma.g12.map.field.flatten()**2)**.5)
            IA_name.append('Sys_py')

        if 'disttoTJ/GB' in listvar:
            im=scipy.ndimage.interpolation.zoom(self.grainId.field,2,order=0,mode='nearest')
            BB=skm.skeletonize(sks.find_boundaries(im))

            BB[0,:]=1
            BB[-1,:]=1
            BB[:,0]=1
            BB[:,-1]=1

            idBx,idBy=np.where(BB==1)
            mconv=np.ones([3,3])
            mconv[1,1]=2
            idTx,idTy=np.where(scipy.signal.convolve2d(BB,mconv,mode='same')>4)

            ss=np.shape(self.grainId.field)
            dist_to_GB=np.zeros(ss) # dist to Grain Boundaries
            dist_to_TJ=np.zeros(ss) # dist to Triple Junction

            print('Compute dist to TJ and GB')

            for i in tqdm(range(ss[0])):
                for j in list(range(ss[1])):
                    dist_to_GB[i,j]=np.min(((idBx/2-i)**2+(idBy/2-j)**2)**0.5)
                    dist_to_TJ[i,j]=np.min(((idTx/2-i)**2+(idTy/2-j)**2)**0.5)

            dist_to_GB=im2d.image2d(dist_to_GB,self.grainId.res)
            dist_to_TJ=im2d.image2d(dist_to_TJ,self.grainId.res)
            IA_data.append(dist_to_GB.field.flatten())
            IA_name.append('dist_to_GB')

            IA_data.append(dist_to_TJ.field.flatten())
            IA_name.append('dist_to_TJ')

        if 'misAngle' in listvar:
            def euler2vec(phi1,phi):
                return np.matrix([np.sin(phi1)*np.sin(phi),-np.cos(phi1)*np.sin(phi),np.cos(phi)])

            ori=self.orientation
            ss=np.shape(self.grainId.field)
            CLg=np.zeros(ss)
            misAngle=np.zeros(ss)
            print('Compute misAngle')
            for i in tqdm(range(ss[0])):
                for j in list(range(ss[1])):
                    nb=self.grainId.field[i,j]
                    id=np.where(self.grainId.field!=nb)
                    a=(i-id[0])**2+(j-id[1])**2
                    ix=np.where(a==np.min(a))[0][0]

                    CLg[i,j]=self.grainId.field[id[0][ix],id[1][ix]]

                    id1=np.where(self.orientation[:,0]==CLg[i,j])[0][0]
                    id2=np.where(self.orientation[:,0]==self.grainId.field[i,j])[0][0]

                    misAngle[i,j]=math.acos(np.round(euler2vec(ori[id1,1],ori[id1,2])*np.transpose(euler2vec(ori[id2,1],ori[id2,2])),4))


            misAngle=im2d.image2d(misAngle,self.grainId.res)

            IA_data.append(misAngle.field.flatten())
            IA_name.append('misAngle')   

        if 'SchmidFactor' in listvar:
            Schmid_factor=np.zeros(ss) # dist to Triple Junction
            print('Compute Schmid Factor')
            for i in np.unique(self.grainId.field.flatten()):
                idGx,idGy=np.where(self.grainId.field==i)
                ii=np.where(self.orientation[:,0]==i)[0]
                Schmid_factor[idGx,idGy]=np.abs(0.5*np.sin(2*self.orientation[np.int(ii),2]))

            Schmid_factor=im2d.image2d(Schmid_factor,self.grainId.res)


            IA_data.append(Schmid_factor.field.flatten())
            IA_name.append('Schmid_factor')

        if 'diffStress' in listvar:
            a,v=self.stress.diag(twod=False)
            IA_data.append((a[2,:,:]-a[0,:,:]).flatten())
            IA_name.append('diffStress')

        if 'diffStrain' in listvar:
            a,v=self.strain.diag(twod=False)
            IA_data.append((a[2,:,:]-a[0,:,:]).flatten())
            IA_name.append('diffStrain')

        #Export
        for i in list(range(len(IA_name))):
            if i==0:
                Head=IA_name[i]
            else:
                Head=Head+' '+IA_name[i]

        np.savetxt(name+'.csv',np.transpose(np.array(IA_data)),header=Head,comments='')