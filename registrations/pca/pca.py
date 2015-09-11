# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 18:33:27 2014

@author: dp11

class for Principal Component Analysis

Usage:
    p = PCA( A, L=eye(N), fraction=0.90, normalise=False, ptype='normal' )
In:
    A:          an array of N observations x D variables, e.g. N rows x D columns
    fraction:   use principal components that account for e.g. 90 % of the total variance
    normalise:  normalise (True) or not (default) by the standard deviation of A
    type:       'normal' (default) or 'dual' (i.e. using Gram's normalisation) 
    L:          matrix of correlations. If unspecified, the identity matrix and standard 
                PCA are computed. Otherwise, supervised PCA is performed

Members:
    p.npc:      number of principal components explaining >= `fraction` of the total variance.
    p.eigVal:   the first p.npc eigenvalues of A*A, in decreasing order.
    p.eigVec:   the first p.npc eigenvectors of A*A, in decreasing order.
    p.exemplar  mean value
    p.std       standard deviation
    p.C         covariance matrix
    p.Z         projection of the input data into the reduced space
    
Methods:
    p.transform()               returns the projection of A onto the reduced space
    p.inverse_transform( X )    reconstructs the observation X from the reduced space to 
                                    the original space

"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

#...............................................................................
class PCA:
    def __init__( self, A, L, fraction=0.90, normalise=False, ptype='normal' ):
        assert 0 <= fraction <= 1
#==============================================================================
#       Center matrix
#==============================================================================
        self._N, self._D = A.shape
        self.exemplar = A.mean(axis=0)
        self.std = A.std(axis=0)
        self.type = ptype
        A = A - self.exemplar
        if normalise:
            self.std = np.where(self.std, self.std, 1.)
            A = A/self.std
        else:
            self.std = np.ones( A.shape[-1] )  
#==============================================================================
#       Matrix for supervision. Set to identity for unsupervised PCA
#==============================================================================
        self.L = L
        self.A      = A.T
        self.type   = ptype
        self.psi    = self.A
        if self.type.lower()=='normal':
            self.C = np.dot(self.A, np.dot( self.L, self.A.T ) )
        elif self.type.lower()=='dual':
            # LDLT not available, using cholesky if L positive-definite, probelm otherwise
#            d_m, l_m = np.linalg.eig( self.L )
#            P = np.abs( np.dot( np.real(l_m), np.sqrt( np.diag( np.abs( np.real(d_m) ) ) ) ) )
            P = np.linalg.cholesky( self.L )
            self.psi = np.dot( self.A, P.T )
            self.C = np.dot(self.psi.T, self.psi)
        else:
            raise Exception("PCA type not supported!")
#==============================================================================
#       Compute eigenvalues and eigenvectors
#==============================================================================
        eigVal, eigVec = np.linalg.eig( self.C )
#==============================================================================
#       Order eigenvalues and eigenvectors
#==============================================================================
        eigVec = eigVec[:,np.where(eigVal > 0.00001)[0]]
        eigVal = eigVal[np.where(eigVal > 0.00001)[0]]
        self.eigVal = np.sort( eigVal )[::-1]
        self.eigVec = eigVec[:,np.argsort( eigVal )[::-1]]
        self.npc = np.searchsorted( np.cumsum(self.eigVal)/np.sum(self.eigVal), fraction) +1
#==============================================================================
#       Return only the npc eigenvalues and eigenvectors
#==============================================================================
        self.eigVal = self.eigVal[:self.npc]
        self.eigVec = self.eigVec[:, :self.npc]
        
    def transform( self ):
        """ This function projects the training data into the PCA subsace """
        if self.type.lower()=='normal':
            self.Z = np.dot(self.A.T, self.eigVec)
            return self.Z
        elif self.type.lower()=='dual':
            U = np.dot( np.dot( self.psi, self.eigVec ) , np.linalg.inv(np.diag(np.sqrt( self.eigVal )) ) ) 
            self.Z = np.real( np.dot(U.T, self.A).T )
            return self.Z
    
    def inverse_transform( self, Y ):
        """ This function reconstructs the observation Y from the PCA subspace
            to the input space """
        if self.type.lower()=='normal':
            X_norm = np.dot(self.eigVec, Y.T)
            mu, std = np.repeat(self.exemplar.reshape([self._D, 1]), X_norm.shape[-1], axis=1),\
                np.repeat(self.std.reshape([self._D, 1]), X_norm.shape[-1], axis=1)
            return (mu + std*X_norm).T 
        elif self.type.lower()=='dual':
            U = np.dot( np.dot( self.psi, self.eigVec ) , np.linalg.inv(np.diag(np.sqrt( self.eigVal )) ) ) 
            X_norm = np.dot(U,Y.T)
            mu, std = np.repeat(self.exemplar.reshape([self._D, 1]), X_norm.shape[-1], axis=1),\
                np.repeat(self.std.reshape([self._D, 1]), X_norm.shape[-1], axis=1)
            return (mu + std*X_norm).T 
    
    def plot( self ):
        """ This function plots the first three PCs loads"""
        self.transform()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.Z[:,0], self.Z[:,1], self.Z[:,2], s=60, marker='o', alpha=1)
        ax.set_xlabel('1st PC')
        ax.set_ylabel('2nd PC')
        ax.set_zlabel('3rd PC')
        plt.show()

