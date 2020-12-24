'''
.. py:module:: gammarun
Created on 27 jan. 2016
This class is build to build gamma map

@author: Thomas Chauve
@contact: thomas.chauve@lgge.obs.ujf-grenoble.fr
@license: CC-BY-CC
'''

import pyCraFT.image2d as im2d
import pyCraFT.symetricTensorMap as sTM
import pyCraFT.TensorMap as TM
import numpy as np

class gammamap(object):
    '''
    This toolbox is building a gamma map
    
    This toolbox is running on python and need various packages :
    
    :library: image2d  
    '''
    pass


    def __init__(self, map,burger,plane):
        '''
        :param map: output map from CraFT
        :type map: im2d.image2d
        :param burger: burger vector of the slip system
        :type burger: array
        :param plane: normal vector of the plane of the slip system
        :type plane: array
        
        :return: gammamap object with '.map' (output from CraFT), '.b' burger vector '.plane' planver vector of the slip system
        :rtype: gammamap
        '''
        self.map=map
        self.b=burger
        self.plane=plane
        
    def schmidtensor(self,craft,use_sTM=False):
        '''
        Compute the schmidtensor 1/2*(b*plane+plane*b)
        
        :param sTM: if you want the schmid tensor return under a symetric tensor map
        :type sTM: bool 
        :return: array 3*3
        :rtype: array or sTM.symetricTensorMap
        '''
        
        
        if use_sTM:
            st= 0.5*(np.tensordot(self.plane,self.b,axes=0)+np.tensordot(self.b,self.plane,axes=0))
            
            ss=self.map.field.shape
            res=self.map.res
            
            t11=im2d.image2d(np.zeros(ss),res)
            t12=im2d.image2d(np.zeros(ss),res)
            t13=im2d.image2d(np.zeros(ss),res)
            t22=im2d.image2d(np.zeros(ss),res)
            t23=im2d.image2d(np.zeros(ss),res)
            t33=im2d.image2d(np.zeros(ss),res)
        
            for i in list(range(len(craft.orientation))):
                id_ori=np.int(craft.orientation[i,0])
                idx,idy=np.where(id_ori==craft.grainId.field)
                mr=matrot(craft.orientation[i,1],craft.orientation[i,2],craft.orientation[i,3])
                
                mat=np.transpose(mr)*st*mr
                    
                t11.field[:,:]=mat[0,0]
                t12.field[:,:]=mat[0,1]
                t13.field[:,:]=mat[0,2]
                t22.field[:,:]=mat[1,1]
                t23.field[:,:]=mat[1,2]
                t33.field[:,:]=mat[2,2]
            
            res=sTM.symetricTensorMap(t11,t22,t33,t12,t13,t23)
        else :
            res= 0.5*(np.tensordot(self.plane,self.b,axes=0)+np.tensordot(self.b,self.plane,axes=0))
        
        return res
        
    def bdotn(self,use_TM=False):
        '''
        Compute the dyadic product ou b time n
        
        :param TM: if you want the b X n return under a tensor map
        :type TM: bool 
        :return: array 3*3
        :rtype: array or TM.TensorMap
        '''
        if TM:
            st= np.tensordot(self.b,self.plane,axes=0)
            
            ss=self.map.field.shape
            res=self.map.res
            
            t11=im2d.image2d(np.zeros(ss),res)
            t12=im2d.image2d(np.zeros(ss),res)
            t13=im2d.image2d(np.zeros(ss),res)
            
            t21=im2d.image2d(np.zeros(ss),res)
            t22=im2d.image2d(np.zeros(ss),res)
            t23=im2d.image2d(np.zeros(ss),res)
            
            t31=im2d.image2d(np.zeros(ss),res)
            t32=im2d.image2d(np.zeros(ss),res)
            t33=im2d.image2d(np.zeros(ss),res)
        
            t11.field[:,:]=st[0,0]
            t12.field[:,:]=st[0,1]
            t13.field[:,:]=st[0,2]
            
            t21.field[:,:]=st[1,0]
            t22.field[:,:]=st[1,1]
            t23.field[:,:]=st[1,2]
            
            t31.field[:,:]=st[2,0]
            t32.field[:,:]=st[2,1]
            t33.field[:,:]=st[2,2]
            
            res=TM.TensorMap(t11,t12,t13,t21,t22,t23,t31,t32,t33)
        else :
            res= np.tensordot(self.b,self.plane,axes=0)  
            
        return res 
        
    
    def vpstrain(self,craft):
        '''        
        :return: viscoplastic strain of the slip system
        :rtype: sTM.symetricTensorMap
        '''
        return self.schmidtensor(craft,use_sTM=True)*self.map
    
    def Lpi(self):
        '''
        Compute L^p the plastic portion. (see Le et al. 2016)
        '''
        
        Lpi=self.bdotn(use_TM=True)*self.map
        
        return Lpi
    
def matrot(phi1,phi,phi2):
    Rphi1=np.matrix([[np.cos(phi1), -np.sin(phi1), 0],[np.sin(phi1), np.cos(phi1), 0],[0,0,1]])
    Rphi=np.matrix([[1,0,0],[0,np.cos(phi), -np.sin(phi)],[0,np.sin(phi), np.cos(phi)]])
    Rphi2=np.matrix([[np.cos(phi2), -np.sin(phi2), 0],[np.sin(phi2), np.cos(phi2), 0],[0,0,1]])
    
    return Rphi2*Rphi*Rphi1