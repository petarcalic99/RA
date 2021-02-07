#!/usr/local/bin/python

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


class Line:
    def __init__(self, batch_size):
        self.nb_dims = 1
        self.theta = np.zeros(shape=(batch_size, self.nb_dims+1))

    def f(self, x):    #hmmm
        """
        Get the FA output for a given input variable(s)

        :param x: A single or vector of dependent variables with size [Ns] for which to calculate the features

        :returns: the function approximator output
        """
        if np.size(x) == 1:
            xl = np.vstack(([x], [1]))
        else:
            xl = np.vstack((x, np.ones((1, np.size(x)))))
        return np.dot(self.theta, xl)

    # ----------------------#
    # # Training Algorithm ##
    # ----------------------#

    def train(self, x_data, y_data):
        # Finds the Least Square optimal weights
        x_data = np.array([x_data]).transpose()
        y_data = np.array(y_data)
        x = np.hstack((x_data, np.ones((x_data.shape[0], 1))))
        
        #a verifier
        theta_opt = np.dot(np.dot(np.linalg.inv(np.dot(x.transpose(),x)),np.transpose(x)), y_data) 
        self.theta = theta_opt          #maybe  
        print("theta", self.theta)
        # ----------------------#
        # # Training Algorithm ##
        # ----------------------#

    def train_regularized(self, x_data, y_data, coef):
        # Finds the regularized Least Square optimal weights
        x_data = np.array([x_data]).transpose()
        y_data = np.array(y_data)
        x = np.hstack((x_data, np.ones((x_data.shape[0], 1))))
       
        # identite * le coeff
        idc = np.eye((np.dot(x.transpose(),x)).shape[0]) * coef   
       # print(idc)

        theta_opt3 = np.dot(np.dot(np.linalg.inv(np.add(np.dot(x.transpose(),x),idc)),np.transpose(x)), y_data) 
        self.theta = theta_opt3          #update de Theta  
        print("theta3", self.theta)
    # ----------------------#
    # # Training Algorithm ##
    # ----------------------#

    def train_from_stats(self, x_data, y_data):
        # Finds the Least Square optimal weights: python provided version
        slope, intercept, r_value, _, _ = stats.linregress(x_data, y_data)
       # print("theta",self.theta)
        #print("slope",slope)
        #print("intercept",intercept)
        self.theta = np.hstack((slope,intercept))
        print("theta2",self.theta)
            

    # -----------------#
    # # Plot function ##
    # -----------------#

    def plot(self, x_data, y_data, label=None):
        xs = np.linspace(0.0, 1.0, 1000)
        z = self.f(xs)
        if (label is not None):
            plt.plot(x_data, y_data, 'o', markersize=3, color='lightgreen')
            plt.plot(xs, z, lw=2, label = label)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        else:
            plt.plot(x_data, y_data, 'o', markersize=3, color='lightgreen')
            plt.plot(xs, z, lw=2, color='red')
            plt.show()

    def calcerr(self, x_data ,y_data , batchsize):
        sumerr = 0
        for i in range(batchsize):
            sumerr = sumerr + abs(y_data[i] - self.f(x_data[i]))
        return sumerr/ batchsize
