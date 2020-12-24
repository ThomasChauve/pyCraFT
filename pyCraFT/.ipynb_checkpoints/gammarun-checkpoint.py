# -*- coding: utf-8 -*-
'''
.. py:module:: gammarun
Created on 26 jan. 2016
This class is build to use gamma output from CraFT

@author: Thomas Chauve
@contact: thomas.chauve@lgge.obs.ujf-grenoble.fr
@license: CC-BY-CC
'''

import pyCraFT.image2d as im2d
import pyCraFT.gammamap as GamM

import numpy as np


class gammarun(object):
    '''
    This class defined gamma structure for CraFT output.
    
    This toolbox is running on python and need various packages :
    
    :library: numpy
    :library: image2d
    :library: gammamap       
    '''
    pass

    def __init__(self, g1,g2,g3,g4,g5,g6,g7,g8,g9,g10,g11,g12):
        '''
        Build gamma object with 12 map (image2d object)
         
        :param gi: map of gi
        :type gi: im2d.image2d
        :return: gamma object
        :rtype: gamma
        
        gamma structure :
        
        :element: *.gi : 12 'image2d'
        '''
        
        # systemes basaux
        self.g01=GamM.gammamap(g1,np.array([1,1,0]),np.array([0,0,1]))
        self.g02=GamM.gammamap(g2,np.array([0,-1,0]),np.array([0,0,1]))
        self.g03=GamM.gammamap(g3,np.array([-1,0,0]),np.array([0,0,1]))
        # systemes prismatiques
        self.g04=GamM.gammamap(g4,np.array([1,0,0]),np.array([0,1,0]))
        self.g05=GamM.gammamap(g5,np.array([1,1,0]),np.array([1,-1,0]))
        self.g06=GamM.gammamap(g6,np.array([0,-1,0]),np.array([-1,0,0]))
        # systemes pyramidaux
        self.g07=GamM.gammamap(g7,np.array([1,1,-1]),np.array([1,1,2]))
        self.g08=GamM.gammamap(g8,np.array([-1,-1,-1]),np.array([-1,-1,2]))
        self.g09=GamM.gammamap(g9,np.array([1,0,-1]),np.array([2,-1,2]))
        self.g10=GamM.gammamap(g10,np.array([-1,0,-1]),np.array([-2,1,2]))
        self.g11=GamM.gammamap(g11,np.array([0,-1,-1]),np.array([1,-2,2]))
        self.g12=GamM.gammamap(g12,np.array([0,1,-1]),np.array([-1,2,2]))
       
        return
 
    def vpstrain(self,craft):
        '''
        :return: viscoplastic strain sum other all the slip system
        :rtype: symetricTensorMap
        '''
        
        res=self.g01.vpstrain(craft)+self.g02.vpstrain(craft)+self.g03.vpstrain(craft)+self.g04.vpstrain(craft)+self.g05.vpstrain(craft)+self.g06.vpstrain(craft)+self.g07.vpstrain(craft)+self.g08.vpstrain(craft)+self.g09.vpstrain(craft)+self.g10.vpstrain(craft)+self.g11.vpstrain(craft)+self.g12.vpstrain(craft)
        
        return res
    
    def Lp(self):
        '''
        :return: plastic portion L^p ! sum other all the slip system
        :rtype: TensorMap
        '''
        
        Lp=self.g01.Lpi()+self.g02.Lpi()+self.g03.Lpi()+self.g04.Lpi()+self.g05.Lpi()+self.g06.Lpi()+self.g07.Lpi()+self.g08.Lpi()+self.g09.Lpi()+self.g10.Lpi()+self.g11.Lpi()+self.g12.Lpi()
        
        return Lp
    
    def NyeTensor(self):
        '''
        Compute the Ny tensor : alpha=sum(-curl(gamma(s)*b(s)(x)n(s)))
        
        :return: Nye Tensor in every pixel
        :rtype: TensorMap
        
        '''
        
        NT=self.g01.Lpi().curl()*(-1.0)+self.g02.Lpi().curl()*(-1.0)+self.g03.Lpi().curl()*(-1.0)+self.g04.Lpi().curl()*(-1.0)+self.g05.Lpi().curl()*(-1.0)+self.g06.Lpi().curl()*(-1.0)+self.g07.Lpi().curl()*(-1.0)+self.g08.Lpi().curl()*(-1.0)+self.g09.Lpi().curl()*(-1.0)+self.g10.Lpi().curl()*(-1.0)+self.g11.Lpi().curl()*(-1.0)+self.g12.Lpi().curl()*(-1.0)
        return NT
    
    def gamma_activity(self,plane='ba'):
        '''
        :param plane: relative activity of 'ba' for basal, 'pr' for prismatic, 'py' for pyramidal
        :type plane: str
        :return: relative basal activity map
        :rtype: im2d.image2d
        '''
        
        norm=self.g01.map.pow(2.)+self.g02.map.pow(2.)+self.g03.map.pow(2.)+self.g04.map.pow(2.)+self.g05.map.pow(2.)+self.g06.map.pow(2.)+self.g07.map.pow(2.)+self.g08.map.pow(2.)+self.g09.map.pow(2.)+self.g10.map.pow(2.)+self.g11.map.pow(2.)+self.g12.map.pow(2.)
                
        if plane=='ba':
            res=((self.g01.map.pow(2.)+self.g02.map.pow(2.)+self.g03.map.pow(2.))/norm).pow(0.5)
        elif plane=='pr':
            res=((self.g04.map.pow(2.)+self.g05.map.pow(2.)+self.g06.map.pow(2.))/norm).pow(0.5)
        elif plane=='py':
            res=((self.g07.map.pow(2.)+self.g08.map.pow(2.)+self.g09.map.pow(2.)+self.g10.map.pow(2.)+self.g11.map.pow(2.)+self.g12.map.pow(2.))/norm).pow(0.5)
            
        return res