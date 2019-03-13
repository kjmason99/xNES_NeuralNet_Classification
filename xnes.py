
# An xNES trained Neural Network for classification
# building on the xNES code sourced from https://github.com/chanshing/xnes/blob/master/xnes.py

# This code creates an instance of the neural network class and loads in the data sets.
# The xNES algorithm then sets the weights of the neural network and evaluates its performance on the data set.

import joblib
import random
import numpy as np

import scipy as sp
from scipy import (dot, eye, randn, asarray, array, trace, log, exp, sqrt, mean, sum, argsort, square, arange)
from scipy.stats import multivariate_normal, norm
from scipy.linalg import (det, expm)
from NeuralNetworkClass import NeuralNetwork
import Utils as utils

class XNES(object):
    def __init__(self, f_train, f_test, f_val, mu, A_mat,
                 eta_mu=1.0, eta_sigma=None, eta_B_mat=None,
                 npop=None, use_fshape=True, use_adasam=False, patience=100, n_jobs=1):
        self.f = f_train # function
        self.f_test = f_test
        self.f_val=f_val
        self.mu = mu # mu is the mean
        self.eta_mu = eta_mu # learning rate
        self.use_adasam = use_adasam # boolean weather we use adasam sampling
        self.n_jobs = n_jobs # number of parallel threads

        d = len(mu) # dimensions of the problem
        sigma = abs(det(A_mat))**(1.0/d) # step size (simlar to CMAES)
        B_mat = A_mat*(1.0/sigma) # transformation matrix
        self.d = d # dimensions
        self.sigma = sigma # step size
        self.B_mat = B_mat #transformation matrix 
        self.stop = False

        # default population size and learning rates
        npop = int(4 + 3*log(d)) if npop is None else npop # population size
        # step size learning rate
        eta_sigma = 3*(3+log(d))*(1.0/(5*d*sqrt(d))) if eta_sigma is None else eta_sigma
        # B learning rate
        eta_B_mat = 3*(3+log(d))*(1.0/(5*d*sqrt(d))) if eta_B_mat is None else eta_B_mat
        self.npop = npop # population
        self.eta_sigma = eta_sigma # step size learning rate
        self.eta_B_mat = eta_B_mat # B learning rate

        # compute utilities if using fitness shaping
        if use_fshape:
            a = log(1+0.5*npop)
            utilities = array([max(0, a-log(k)) for k in range(1,npop+1)])
            utilities /= sum(utilities)
            utilities -= 1.0/npop           # broadcast
            utilities = utilities[::-1]  # ascending order
        else:
            utilities = None # refered to as u in Gecco paper (utility function)
        self.use_fshape = use_fshape
        self.utilities = utilities

        # stuff for adasam
        self.eta_sigma_init = eta_sigma # initial step size learning rate
        self.sigma_old = None # old step size

        # logging
        self.fitness_best = None
        self.fitness_val = 0.0
        self.best_testFit = 0.0 
        self.mu_best = None
        self.done = False
        self.counter = 0
        self.patience = patience
        self.history = {'eta_sigma':[], 'sigma':[], 'fitness':[]}

        # do not use these when hill-climbing
        if npop == 1:
            self.use_fshape = False
            self.use_adasam = False

    def step(self, niter):
        """ xNES """
        f = self.f
        mu, sigma, B_mat = self.mu, self.sigma, self.B_mat
        eta_mu, eta_sigma, eta_B_mat = self.eta_mu, self.eta_sigma, self.eta_B_mat
        npop = self.npop
        d = self.d
        sigma_old = self.sigma_old
        
        I_mat = eye(d)

        with joblib.Parallel(n_jobs=self.n_jobs) as parallel:
            for i in range(niter):
                if (i%10==0):
                    
                    print('Itter %s of %s'%(str(i),str(niter)))
                    print('best Train Accuracy ',self.fitness_best )
                if(not self.stop):
                    for k in range (len(mu)):
                        if(not self.stop):
                            # if values over ~1e140, get error as too large
                            if(mu[k]>1e140 or abs(mu[k])<(1e-140)): 
                                print('Alert! Value too large/small, mu[k] = ',mu[k])
                                self.stop = True
                       #        input("Press Enter to continue...")
                    
                    z = randn(npop, d) # random [npop X d] array
                    x = mu + sigma * dot(z, B_mat)  # sampling next generation   # broadcast

                    # evaluate solutions in parallel on function f
                    f_try = parallel(joblib.delayed(f)(z) for z in x)
                    f_try = asarray(f_try)

                    # save if best
                    
                    fitness = mean(f_try) # mean fitness of next generation
                    if(self.fitness_best is not None):
                        if ((fitness - 1e-8) > (self.fitness_best)):
                            self.fitness_best = fitness # update best fitness
                            self.mu_best = mu.copy() # update best solution
                            self.counter = 0
                        else: self.counter += 1
                    else: # best fitness is mean fitness
                        self.fitness_best = fitness
                        self.mu_best = mu.copy()
                        self.counter = 0
                        
                    
                    if(self.fitness_best is None):
                        self.fitness_best = mean(f_try)
                        self.mu_best = mu.copy()
                        self.counter = 0
                        
                    for j in range (len(f_try)):
                    #    validation_fit = self.f_val(x[j])
                     #   if(f_try[j]> self.fitness_best and validation_fit>self.fitness_val):
                        if(f_try[j]> self.fitness_best):
                            self.fitness_best = f_try[j] # update best fitness
                            self.mu_best = x[j].copy() # update best solution
                            self.counter = 0
                      #      self.fitness_val = validation_fit
                            
                    if self.counter > self.patience:
                        self.done = True
                        return
                    self.best_testFit = self.f_test(self.mu_best)
    #                print('best Train Accuracy ',self.fitness_best )
    #                print('best Test Accuracy ',self.best_testFit )
                    # sort next generation w.r.t their fitnesses
                    unsorted_f_try = f_try
                    isort = argsort(f_try)
                    f_try = f_try[isort]
                    z = z[isort]
                    x = x[isort]

                    # u = utility if shaping, otherwise just use ordered fitnesses
                    u = self.utilities if self.use_fshape else f_try

                    # adaptation sampling 
                    if self.use_adasam and sigma_old is not None:  # sigma_old must be available
                        eta_sigma = self.adasam(eta_sigma, mu, sigma, B_mat, sigma_old, x)

                    # G_delta = G_delta = u.s (s is z in gecco paper)
                    G_delta = dot(u, z) 
                    # G_M = G_M = SUM u [dot] (s s_transpose - I)
                    G_M = dot(z.T, z*u.reshape(npop,1)) - sum(u)*I_mat
                    # G_sigma = G_sigma = trace of G_M * (1/d)
                    G_sigma = trace(G_M)*(1.0/d)
                    # G_B = G_B = G_M - G_delta*I
                    G_B = G_M - G_sigma*I_mat
                    # update old sigma
                    sigma_old = sigma

                    # update mu (center) = mu + eta_mu*sigma DOT (B, G_delta)
                    mu += eta_mu * sigma * dot(B_mat, G_delta)
                    # update sigma = sigma * exp(eta_delta * 0.5 * G_sigma)
                    sigma *= exp(0.5 * eta_sigma * G_sigma)
                    # update B = DOT (B, exp(0.5 eta_B * G_B))
                    B_mat = dot(B_mat, expm(0.5 * eta_B_mat * G_B))

                    # logging
                    for j in range (len(unsorted_f_try)):
                        self.history['fitness'].append(unsorted_f_try[j])
                        
                 #   self.history['fitness'].append(fitness)
                    self.history['sigma'].append(sigma)
                    self.history['eta_sigma'].append(eta_sigma)
                    

        # keep last results
        self.mu, self.sigma, self.B_mat = mu, sigma, B_mat
        self.eta_sigma = eta_sigma
        self.sigma_old = sigma_old
        print('best Test Accuracy ',self.best_testFit )
        
            

    def adasam(self, eta_sigma, mu, sigma, B_mat, sigma_old, x):
        """ Adaptation sampling """
        eta_sigma_init = self.eta_sigma_init
        d = self.d
        c = .1
        rho = 0.5 - 1./(3*(d+1))  # empirical

        BB_mat = dot(B_mat.T, B_mat)
        cov = sigma**2 * BB_mat
        sigma_ = sigma * sqrt(sigma*(1./sigma_old))  # increase by 1.5
        cov_ = sigma_**2 * BB_mat

        p0 = multivariate_normal.logpdf(x, mean=mu, cov=cov)
        p1 = multivariate_normal.logpdf(x, mean=mu, cov=cov_)
        w = exp(p1-p0)

        # Mann-Whitney. It is assumed x was in ascending order.
        n = self.npop
        n_ = sum(w)
        u_ = sum(w * (arange(n)+0.5))

        u_mu = n*n_*0.5
        u_sigma = sqrt(n*n_*(n+n_+1)/12.)
        cum = norm.cdf(u_, loc=u_mu, scale=u_sigma)

        if cum < rho:
            return (1-c)*eta_sigma + c*eta_sigma_init
        else:
            return min(1, (1+c)*eta_sigma)


if __name__ == '__main__':
    import time
    import statistics as stats
    np.random.seed(43)
    random.seed(43)
    datasetName="seeds_dataset" # 1) seeds_dataset 2) Iris
    filename = datasetName+".csv"
    n_hidden_nodes = [5] #  5
    X_data, y_data = utils.read_csv(filename)  # read as matrix of floats and int
    utils.normalize(X_data)  # normalize
    N, NNInputs = X_data.shape  # extract shape of X
    n_classes = len(np.unique(y_data))
    k_folds = 5
    evals = 1000 # 100
    xnes_pop = 10 # 5
    split = 0.8
    xnes_steps = int(evals/xnes_pop)
    xnes_eta_B_mat=0.02
    xnes_eta_sigma=0.2
    
    best_train_Fits = list()
    best_test_Fits = list()
    convergences = np.zeros(shape=(k_folds,evals))
    descrip = datasetName+"_"+str(k_folds)+"_folds_"+str(evals)+"_evals_"+str(xnes_pop)+"_popSize_"+str(split)+"split"
    
    for k in range(k_folds):
    
        print('Splitting data======================== ')
        X_train, X_test, y_train, y_test = utils.split_data(X_data, y_data, split)  
     #   X_train, X_val, X_test, y_train,y_val, y_test = utils.split_data_validation(X_data, y_data, 0.6, 0.2)
        
        model_1 = NeuralNetwork(n_input=NNInputs, n_output=n_classes, n_hidden_nodes=n_hidden_nodes)
        numWeights = model_1.countWeights()

        mu = np.random.rand(numWeights)

        
        def f_train(x):
            model = model_1
            model.setWeights(x)
            y_predict = model.predict(X_train)
            r = 100*np.sum(y_train==y_predict)/len(y_train)
            return r

        def f_val(x):
            model = model_1
            model.setWeights(x)
            y_predict = model.predict(X_val)
            r = 100*np.sum(y_val==y_predict)/len(y_val)
            return r
        
        def f_test(x):
            model = model_1
            model.setWeights(x)
            y_predict = model.predict(X_test)
            r = 100*np.sum(y_test==y_predict)/len(y_test)
            return r
            

        A_mat = eye(numWeights)

        xnes = XNES(f_train, f_test, f_val, mu, A_mat, npop=xnes_pop, use_adasam=True, eta_B_mat=xnes_eta_B_mat, eta_sigma=xnes_eta_sigma, patience=9999)
        t0 = time.time()
        xnes.step(xnes_steps)
        
        convergences[k] = xnes.history['fitness']
        best_train_Fits.append(xnes.fitness_best)
        best_test_Fits.append(xnes.best_testFit)
            
    print("Took {} secs".format(time.time()-t0))
    avgTrainFit = mean(best_train_Fits)
    avgTestFit = mean(best_test_Fits)
    Train_stdev = stats.stdev(best_train_Fits)
    Test_stdev = stats.stdev(best_test_Fits)
    avg_Converge = list()
    stdev_Converge = list()
    for i in range(evals):
        curConverge = list()
        for k in range (k_folds):
            curConverge.append(convergences[k][i])
        avg_Converge.append(mean(curConverge))
        stdev_Converge.append(stats.stdev(curConverge))
            
    print ("Avg Train fitness ", avgTrainFit)
    print ("Train fitness Stdev ", Train_stdev)
    print ("Avg Test fitness ", avgTestFit)
    print ("Test fitness Stdev ", Test_stdev)

    save_path = 'C:/Users/CodeLocation/Results/'

    Acc = "_Train_Fit_"+str(avgTrainFit)+"_Test_Fit_"+str(avgTestFit)

    summaryFile = save_path+"Summary_"+descrip+Acc+".txt"
    AvgConvFile = save_path+"AvgConv_"+descrip+Acc+".txt"
    StdevConvFile = save_path+"StdevConv_"+descrip+Acc+".txt"

    with open(summaryFile, "w") as text_file:
        text_file.write("Avg_Train_fitness, %s \n" % avgTrainFit)
        text_file.write("Train_fitness_Stdev, %s \n" % Train_stdev)
        text_file.write("Avg_Test_fitness, %s \n" % avgTestFit)
        text_file.write("Test_fitness_Stdev, %s \n  \n" % Test_stdev)
        for i in range (len(best_train_Fits)):
            text_file.write("%s Train Fitness %s \n" % (str(i), str(best_train_Fits[i])))

        text_file.write("\n")
        for i in range (len(best_test_Fits)):
            text_file.write("%s Test Fitness %s \n" % (str(i), str(best_test_Fits[i])))
        text_file.write("\n\n")
        text_file.write(descrip)
        text_file.write(" \n\neta_B_mat, %s \n" % xnes_eta_B_mat)
        text_file.write("eta_sigma, %s \n" % xnes_eta_sigma)

    with open(AvgConvFile, "w") as text_file:
        for i in range (len(avg_Converge)):
            temp_s = str(avg_Converge[i])
            text_file.write("%s \n" % temp_s)

    with open(StdevConvFile, "w") as text_file:
        for i in range (len(stdev_Converge)):
            temp_s = str(stdev_Converge[i])
            text_file.write("%s \n" % temp_s)
    

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2,1)
    axs[0].plot(xnes.history['fitness'])
    axs[1].plot(avg_Converge)
    axs[0].set_ylabel('fitness')
    axs[1].set_ylabel('Average Convergence')
fig.show()






























