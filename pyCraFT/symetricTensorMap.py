# -*- coding: utf-8 -*-
'''
.. py:module:: symetricTensorMap
Created on 21 oct. 2015
This class defined map of symetric def positive tensor map.

@author: Thomas Chauve
@contact: thomas.chauve@lgge.obs.ujf-grenoble.fr
@license: CC-BY-CC
'''

import pyCraFT.image2d as im2d

import numpy as np
import matplotlib.pylab as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker


class symetricTensorMap(object):
    '''
    This class defined map of symetric def positive tensor map.
    
    This toolbox is running on python and need various packages :
    
    :library: image2d        
    '''
    pass


    def __init__(self, t11,t22,t33,t12,t13,t23):
        '''
        Build symetricTensorMap object with 6 map (image2d object)
        
        t11 t12 t13
            t22 t23
                t33
         
        :param tii: map of tii composante
        :type tii: im2d.image2d
        :return: symetricTensorMap object
        :rtype: symetricTensorMap
        
        symetricTensorMap structure :
        
        :element: *.tii : 6 'image2d'
        '''
        
        self.t11=t11
        self.t22=t22
        self.t33=t33
        self.t12=t12
        self.t13=t13
        self.t23=t23
        
        return


    def __add__(self,other):
        '''
        Compute the sum of two map tensor
        
        :return: symetricTensorMap
        :rtype: symetricTensorMap
        '''
        
        return symetricTensorMap(self.t11+other.t11,self.t22+other.t22,self.t33+other.t33,self.t12+other.t12,self.t13+other.t13,self.t23+other.t23)
    


    def __mul__(self,other):
        '''
        Compute the multiplication pixel by pixel of two map tensor
        
        :return: symetricTensorMap
        :rtype: symetricTensorMap
        '''
        # if multiply symetricTensorMap with image2d
        if (type(other) is im2d.image2d):
            return symetricTensorMap(self.t11*other,self.t22*other,self.t33*other,self.t12*other,self.t13*other,self.t23*other)
        # if multiply symetricTensorMap with symmetricTensorMap
        elif (type(other) is symetricTensorMap):
            return symetricTensorMap(self.t11*other.t11,self.t22*other.t22,self.t33*other.t33,self.t12*other.t12,self.t13*other.t13,self.t23*other.t23)
        # if multiply symetricTensorMap with number
        elif (type(other) is float):
            return symetricTensorMap(self.t11*other,self.t22*other,self.t33*other,self.t12*other,self.t13*other,self.t23*other)
        
 


    def eqtensor2d(self):
        '''
        Compute the equivalent norm of the tensor as it is a 2D mesurement
        
        teq=sqrt(2/3*(t11^2+t22^2+2*t12^2))
        
        :return: equivalent norm as it is a 2d measurement
        :rtype: im2d.image2d
        '''
        
        eq2d=((self.t11.pow(2)+self.t22.pow(2)+self.t12.pow(2)*2.)*(2./3.)).pow(1./2.)
        
        return eq2d
    


    def eqVonMises(self,normByMedian=False):
        '''
        Compute the Von Mises equivalent norm of the tensor
                
        :param normByMedian: normalized the data by the median
        :type normByMedian: bool
        :return: Von Mises equivalent norm (2/3*eij*eij)^0.5
        :rtype: image2d
        '''
        
        eqVM=((self.t11.pow(2)+self.t22.pow(2)+(self.t12.pow(2)+self.t13.pow(2)+self.t23.pow(2))*2.)*(2./3.)).pow(1./2.)
        
        if normByMedian:
            nn=float(np.nanmedian(eqVM.field.flatten()))
            eqVM=eqVM*(1./nn)
        
        return eqVM
    


    def diag(self,twod=True):
        '''
            Diagonalisation of the symetricTensorMap
            
            :param twod: use a 2d symetric tensor (=> t33=t23=t13=0) (default True)
            :type twod: bool
            :return: a, eigenvalue dim(3*n*m) where n*m is the dimension of the map (self.t11.field.shape)
            :rtype: array (3*n*m)
            :return: v, eigenvector dim(3*3*n*m) where n*m is the dimension of the map (self.t11.field.shape) 
            :rtype: array (3*n*m)
            
            .. note:: the eigenvector v(:,k,n,m) is associated to the eigenvalue a(k,n,m)
        '''
        
        ss=self.t11.field.shape
        a=np.zeros([3,ss[0],ss[1]])
        v=np.zeros([3,3,ss[0],ss[1]])
        for i in list(range(ss[0])):
            for j in list(range(ss[1])):
                if twod:
                    tmp=np.matrix([[self.t11.field[i,j],self.t12.field[i,j],0],[self.t12.field[i,j],self.t22.field[i,j],0],[0,0,0]]) 
                else:
                    tmp=np.matrix([[self.t11.field[i,j],self.t12.field[i,j],self.t13.field[i,j]],[self.t12.field[i,j],self.t22.field[i,j],self.t23.field[i,j]],[self.t13.field[i,j],self.t23.field[i,j],self.t33.field[i,j]]])
                
                [a[:,i,j],v[:,:,i,j]]=np.linalg.eigh(tmp)
                
        #img=plt.figure()
        #plt.subplot(131)
        #img1=plt.imshow(a[0,:,:],cmap=cm.jet)
        #plt.colorbar(img1,orientation='vertical',aspect=4)  
        #plt.subplot(132)
        #img2=plt.imshow(a[1,:,:],cmap=cm.jet)
        #plt.colorbar(img2,orientation='vertical',aspect=4)  
        #plt.subplot(133)
        #img3=plt.imshow(a[2,:,:],cmap=cm.jet)
        #plt.colorbar(img3,orientation='vertical',aspect=4)        
        return a,v
    


    def principal_tensor(self,scale=100,space=5,twod=False,absolutemax=False,colorbarcenter=True):
        '''
            Plot the principal tensor direction
            
            :param scale: scale of the arrow (default 100)
            :type scale: float
            :param space: take only 1 value every 'space' value to not charge the picture to much (default 5)
            :type space: int
            :param twod: use a 2d symetric tensor (=> t33=t23=t13=0) (default False)
            :type twod: bool
            :param colorbarcenter: do you want center the colorbar around 0
            :type colorbarcenter: bool
            :return: image of the principal tensor direction
            :rtype: matplotlib.pyplot
        '''
        a,v=self.diag(twod=twod)
        
        # create X, Y, U, V
        X=[]
        Y=[]
        U=[]
        V=[]
        C=[]
        ss=np.shape(self.t11.field)
        for i in list(range(ss[0]/space)):
            for j in list(range(ss[1]/space)):
                ii=space*i
                jj=space*j
                # eigenvalue absolute max
                if absolutemax:
                    kk=np.where(abs(a[:,ii,jj])==max(abs(a[:,ii,jj])))
                else:
                    kk=np.where(a[:,ii,jj]==max(a[:,ii,jj]))        
                k=kk[0][0]
                
                # first arrow
                X.append(ii*self.t11.res)
                Y.append(jj*self.t11.res)
                U.append(a[k,ii,jj]*v[0,k,ii,jj])
                V.append(a[k,ii,jj]*v[1,k,ii,jj])
                C.append(a[k,ii,jj])
                #symetric arrow
                X.append(ii*self.t11.res)
                Y.append(jj*self.t11.res)
                U.append(-a[k,ii,jj]*v[0,k,ii,jj])
                V.append(-a[k,ii,jj]*v[1,k,ii,jj])
                C.append(a[k,ii,jj])
                                
        X=np.array(X)
        Y=np.array(Y)
        U=np.array(U)
        V=np.array(V)
        C=np.array(C)
        img=plt.quiver(Y, ss[0]*self.t11.res-X, scale*V, scale*U, C, angles='xy', scale_units='xy', scale=1,cmap=cm.bwr)
        #plt.colorbar(img,orientation='horizontal',aspect=15,format=ticker.FuncFormatter(plotpdf_craft.fmt))

        plt.colorbar(img,orientation='horizontal',aspect=15)
        plt.axis('equal')
        if colorbarcenter:
            zcolor=np.max(np.abs(C))
            plt.clim(-zcolor, zcolor)
        return img   

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
        #print(id)
        res[0,0],tmp=self.t11.extract_data(pos=id)
        res[0,1],tmp=self.t12.extract_data(pos=id)
        res[0,2],tmp=self.t13.extract_data(pos=id)
        
        res[1,0],tmp=self.t12.extract_data(pos=id)
        res[1,1],tmp=self.t22.extract_data(pos=id)
        res[1,2],tmp=self.t23.extract_data(pos=id)
        
        res[2,0],tmp=self.t13.extract_data(pos=id)
        res[2,1],tmp=self.t23.extract_data(pos=id)
        res[2,2],tmp=self.t33.extract_data(pos=id)
        
        return res,id

