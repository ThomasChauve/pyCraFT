# -*- coding: utf-8 -*-
'''
Created on 11 févr. 2016
.. py:module:: TensorMap
This class defined 2d tensor map of 2nd order.

@author: Thomas Chauve
@contact: thomas.chauve@lgge.obs.ujf-grenoble.fr
@license: CC-BY-CC
'''

import pyCraFT.image2d as im2d
import pyCraFT.symetricTensorMap as sTM

import numpy as np
from matplotlib.axis import Axis
import matplotlib.pylab as plt
import pylab

class TensorMap(object):
    '''
    This class defined 2d map of 2nd order tensor.
        
    This toolbox is running on python and need various packages :
    
    :library: image2d   
    '''


    def __init__(self, t11,t12,t13,t21,t22,t23,t31,t32,t33):
        '''
        Build symetricTensorMap object with 9 map (image2d object)
        
        t11 t12 t13
        t21 t22 t23
        t31 t32 t33
         
        :param tii: map of tii composante
        :type tii: im2d.image2d
        :return: symetricTensorMap object
        :rtype: symetricTensorMap
        
        symetricTensorMap structure :
        
        :element: *.tii : 9 'image2d'
        '''
        
        self.t11=t11
        self.t12=t12
        self.t13=t13
        
        self.t21=t21
        self.t22=t22
        self.t23=t23
        
        self.t31=t31
        self.t32=t32
        self.t33=t33

        return
    
    def __add__(self,other):
        '''
        Compute the sum of two map tensor
        
        :return: TensorMap
        :rtype: TensorMap
        '''
        if (type(other) is TensorMap):
            return TensorMap(self.t11+other.t11,self.t12+other.t12,self.t13+other.t13,self.t21+other.t21,self.t22+other.t22,self.t23+other.t23,self.t31+other.t31,self.t32+other.t32,self.t33+other.t33)
        if (type(other) is sTM.symetricTensorMap):
            return TensorMap(self.t11+other.t11,self.t12+other.t12,self.t13+other.t13,self.t21+other.t12,self.t22+other.t22,self.t23+other.t23,self.t31+other.t13,self.t32+other.t23,self.t33+other.t33)
    
    def __mul__(self,other):
        '''
        Compute the sum of two map tensor
        
        :return: TensorMap
        :rtype: TensorMap
        '''
        if (type(other) is float):
            return TensorMap(self.t11*other,self.t12*other,self.t13*other,self.t21*other,self.t22*other,self.t23*other,self.t31*other,self.t32*other,self.t33*other)
        if (type(other) is im2d.image2d):
            return TensorMap(other*self.t11,other*self.t12,other*self.t13,other*self.t21,other*self.t22,other*self.t23,other*self.t31,other*self.t32,other*self.t33)
        
    def extract_data(self,pos=[]):
        '''
        Extract the value at the position 'pos' or where you clic
        
        :param pos: array [x,y] position of the data, if pos==[], clic to select the pixel
        :type pos: array
        :return: 3*3 symetric matrix of the pixel wanted
        :rtype: np.matrix 
        '''
        
        if pos==[]:
            plt.imshow(self.t11.field,aspect='equal')
            plt.waitforbuttonpress()
            print('select the pixel :')
            #grain wanted for the plot
            id=np.int32(np.array(pylab.ginput(1)))
        else:
            id=pos
        plt.close()
        
        res=np.matrix(np.zeros([3,3]))
        print(id)
        res[0,0],tmp=self.t11.extract_data(pos=id)
        res[0,1],tmp=self.t12.extract_data(pos=id)
        res[0,2],tmp=self.t13.extract_data(pos=id)
        
        res[1,0],tmp=self.t21.extract_data(pos=id)
        res[1,1],tmp=self.t22.extract_data(pos=id)
        res[1,2],tmp=self.t23.extract_data(pos=id)
        
        res[2,0],tmp=self.t31.extract_data(pos=id)
        res[2,1],tmp=self.t32.extract_data(pos=id)
        res[2,2],tmp=self.t33.extract_data(pos=id)
        
        return res,id
        
    def curl(self):
        '''
        Compute the curl of the TensorMap
        curl(A)il=e(lik)*Aik,l
        
        :return: curl(A)
        :rtype: TensorMap
        '''
        
        t11=self.t13.diff(axis='y')*LeviCivita(1, 2, 3)
        t22=self.t23.diff(axis='x')*LeviCivita(2, 1, 3)
        t33=self.t31.diff(axis='y')*LeviCivita(3, 2, 1)+self.t32.diff(axis='x')*LeviCivita(3, 1, 2)
        
        t12=self.t13.diff(axis='x')*LeviCivita(2, 1, 3)
        t21=self.t23.diff(axis='y')*LeviCivita(1, 2, 3)
        
        t13=self.t12.diff(axis='x')*LeviCivita(3, 1, 2)+self.t11.diff(axis='y')*LeviCivita(3, 2, 1)
        t31=self.t33.diff(axis='y')*LeviCivita(1, 2, 3)
        
        t23=self.t22.diff(axis='x')*LeviCivita(3, 1, 2)+self.t21.diff(axis='y')*LeviCivita(3, 2, 1)
        t32=self.t33.diff(axis='x')*LeviCivita(2, 1, 3)
        
        return TensorMap(t11,t12,t13,t21,t22,t23,t31,t32,t33)
        
def LeviCivita(i,j,k):
    '''
    Levi-Civita
    '''    
    return float((i - j) * (j - k) * (k - i) / 2)