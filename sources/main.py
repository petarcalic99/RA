#!/usr/local/bin/python

import numpy as np
import time
from rbfn import RBFN
from lwr import LWR
from line import Line
from sample_generator import SampleGenerator


class Main:
    def __init__(self):
        self.x_data = []
        self.y_data = []
        self.batch_size = 50

    def reset_batch(self):
        self.x_data = []
        self.y_data = []

    def make_nonlinear_batch_data(self):
        """ 
        Generate a batch of non linear data and store it into numpy structures
        """
        self.reset_batch()
        g = SampleGenerator()
        for i in range(self.batch_size):
            # Draw a random sample on the interval [0,1]
            x = np.random.random()
            y = g.generate_non_linear_samples(x)
            self.x_data.append(x)
            self.y_data.append(y)

    def make_linear_batch_data(self):
        """ 
        Generate a batch of linear data and store it into numpy structures
        """
        self.reset_batch()
        g = SampleGenerator()
        for i in range(self.batch_size):
            # Draw a random sample on the interval [0,1]
            x = np.random.random()
            y = g.generate_linear_samples(x)
            self.x_data.append(x)
            self.y_data.append(y)

    def approx_linear_batch(self):
        model = Line(self.batch_size)

        self.make_linear_batch_data()     #reset des batch inclu

        start = time.process_time()
        model.train(self.x_data, self.y_data)
        print("LLS time:", time.process_time() - start)
        model.plot(self.x_data, self.y_data)

        start = time.process_time()
        model.train_from_stats(self.x_data, self.y_data)
        print("LLS from scipy stats:", time.process_time() - start)
        model.plot(self.x_data, self.y_data)

        #Pour la q3 on modifi le coef
        start = time.process_time()
        model.train_regularized(self.x_data, self.y_data, coef=0.1)
        print("regularized LLS :", time.process_time() - start)
        model.plot(self.x_data, self.y_data)

    def approx_rbfn_batch(self):
        model = RBFN(nb_features=10)     #changer la valeur des features
        self.make_nonlinear_batch_data()

        start = time.process_time()
        model.train_ls(self.x_data, self.y_data)
        print("RBFN LS time:", time.process_time() - start)
        model.plot(self.x_data, self.y_data)

      #  start = time.process_time()
    #    model.train_ls2(self.x_data, self.y_data)
    #    print("RBFN LS2 time:", time.process_time() - start)
    #    model.plot(self.x_data, self.y_data)

    def approx_rbfn_iterative(self):
        max_iter = 50       #testons plusieurs valeurs 5,10,50   mais pas plus!
        model = RBFN(nb_features=10)     #3,5,10..,    40 suraprentissage
        start = time.process_time()
        # Generate a batch of data and store it
        self.reset_batch()
        g = SampleGenerator()
        
        for i in range(max_iter):
            #print(i+1)
            # Draw a random sample on the interval [0,1]
            x = np.random.random()
            y = g.generate_non_linear_samples(x)
            self.x_data.append(x)
            self.y_data.append(y)

            # Comment the ones you don't want to use
            model.train_gd(x, y, alpha=0.5)
            #model.train_rls(x, y)
            #model.train_rls_sherman_morrison(x, y)

        print("RBFN Incr time:", time.process_time() - start)
        model.plot(self.x_data, self.y_data)

    def approx_lwr_batch(self):
        model = LWR(nb_features=10)     #modifier le nombre de segments, pour 40 le temps de calc est assez G..
        self.make_nonlinear_batch_data()

        start = time.process_time()
        model.train_lwls(self.x_data, self.y_data)
        print("LWR time:", time.process_time() - start)
        model.plot(self.x_data, self.y_data)

if __name__ == '__main__':
    m = Main()
    #m.approx_linear_batch()
    m.approx_rbfn_batch()
    #m.approx_rbfn_iterative()
    #m.approx_lwr_batch()
