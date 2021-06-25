import numpy as np
from copy import deepcopy as dc
import XPyS
from .helper_functions import index_of, guess_from_data
from IPython import embed as shell

from tqdm.notebook import tqdm

import matplotlib.pyplot as plt

import torch
from torch import nn

class autofit:
    
    def __init__(self,energy,intensity,orbital):
        self.energy = dc(energy)
        self.intensity = dc(intensity)
        self.orbital = dc(orbital)
        self.autofit_pars = self.get_autofit_pars(self.orbital)
        self.guess_params(energy = self.energy,intensity = self.intensity)


    def get_autofit_pars(self,orbital):
        if orbital == 'Nb3d':
            autofitpars_path = '/Users/cassberk/code/XPyS/autofit/autofitNb.txt'
        elif orbital =='Si2p':
            autofitpars_path = '/Users/cassberk/code/XPyS/autofit/autofitSi2p.txt'
        elif orbital =='C1s':
            autofitpars_path = '/Users/cassberk/code/XPyS/autofit/autofitC1s.txt'
        elif orbital =='O1s':
            autofitpars_path = '/Users/cassberk/code/XPyS/autofit/autofitO1s.txt'
        elif orbital =='F1s':
            autofitpars_path = '/Users/cassberk/code/XPyS/autofit/autofitF1s.txt'
        else:
            print('No autofit yet')
            return

        f = open(autofitpars_path, "r")
        comdic = {}
        for line in f.readlines():
            # print(line)
            comdic[line.split(' ')[0]] = [l.rstrip('\n') for l in line.split(' ')[1:]]
        f.close()
        return comdic

    def guess_params(self,energy,intensity):
        
        self.energy = dc(energy)
        self.intensity = dc(intensity)
        guessamp = {}
        for par in self.autofit_pars.keys():
            
            # print(par)
            
            if self.autofit_pars[par][0] == 'lin':
                idx = index_of(self.energy, np.float(self.autofit_pars[par][1]))

                if len(self.autofit_pars[par]) == 3:
                    guessamp[par] = self.intensity[idx]*np.float(self.autofit_pars[par][2])
                elif len(self.autofit_pars[par]) ==4:
                    guessamp[par] = self.intensity[idx]*np.float(self.autofit_pars[par][2]) + np.float(self.autofit_pars[par][3])

            elif self.autofit_pars[par][0] == 'linadj':
                amp,cen = guess_from_data(x= self.energy, y = self.intensity,peakpos = np.float(self.autofit_pars[par][1]))
                idx = index_of(self.energy, cen)

                if len(self.autofit_pars[par]) == 3:
                    guessamp[par] = self.intensity[idx]*np.float(self.autofit_pars[par][2])
                elif len(self.autofit_pars[par]) ==4:
                    guessamp[par] = self.intensity[idx]*np.float(self.autofit_pars[par][2]) + np.float(self.autofit_pars[par][3])

            elif self.autofit_pars[par][0] == 'log':
                idx = index_of(self.energy, np.float(self.autofit_pars[par][1]))
                
                guessamp[par] = np.float(self.autofit_pars[par][2])*np.log(self.intensity[idx]) + np.float(self.autofit_pars[par][3])

            elif self.autofit_pars[par][0] == 'par':
                dep_par = self.autofit_pars[par][1]
                guessamp[par] = guessamp[dep_par]*np.float(self.autofit_pars[par][2])

            elif self.autofit_pars[par][0] == 'mean':
                #  a,c = guess_from_data(self.energy,self.intensity,negative = None,peakpos = np.float(self.autofit_pars[par][1]))
                #  guessamp[par] = c
                guessamp[par] = np.float(self.autofit_pars[par][1])

            elif self.autofit_pars[par][0] == 'guess':
                a,c = guess_from_data(self.energy,self.intensity,peakpos = np.float(self.autofit_pars[par][1]))
                print(c)
                #  guessamp[par] = c
                guessamp[par] = np.float(c)
                
        self.guess_pars = guessamp



class NeuralNet(torch.nn.Module):
    """
         This is a single-layer neural network
         This is the default network for initializing amplitude parameters

    """
    def __init__(self, n_feature, n_hidden1,n_output):
        """
                 Initialization
                 :param n_feature: Feature number
                 :param n_hidden: the number of hidden layer neurons
                 :param n_output: output number
        """
        super(NeuralNet, self).__init__()
        # Parameter one is the number of neurons in the previous network layer, and parameter two is the number of neurons in the network layer
        self.fc1 = torch.nn.Linear(n_feature, n_hidden1)
        self.predict = torch.nn.Linear(n_hidden1, n_output)

    def forward(self, x):
        # relu activation function, the number less than or equal to 0 is directly equal to 0. At this time, the second layer of neural network data is obtained
        x = torch.relu(self.fc1(x))

        # Get the third layer neural network output data
        x = self.predict(x)
        return x

class DeepNeuralNet(torch.nn.Module):
    """
         This is a six-layer neural network.
         This is the default network for initializing sigma and center parameters
    """
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_hidden3, n_hidden4, \
                 n_hidden5, n_hidden6,n_output):
        
        """
                 Initialization
                 :param n_feature: Feature number
                 :param n_hidden: the number of hidden layer neurons
                 :param n_output: output number
        """
        super(DeepNeuralNet, self).__init__()
        # Parameter one is the number of neurons in the previous network layer, and parameter two is the number of neurons in the network layer
        self.fc1 = torch.nn.Linear(n_feature, n_hidden1)
        self.fc2 = torch.nn.Linear(n_hidden1, n_hidden2)
        self.fc3 = torch.nn.Linear(n_hidden2, n_hidden3)
        self.fc4 = torch.nn.Linear(n_hidden3, n_hidden4)
        self.fc5 = torch.nn.Linear(n_hidden4, n_hidden5)
        self.fc6 = torch.nn.Linear(n_hidden5, n_hidden6)
        self.predict = torch.nn.Linear(n_hidden6, n_output)
#         self.predict = torch.nn.Linear(n_hidden1, n_output)

    def forward(self, x):
        # relu activation function, the number less than or equal to 0 is directly equal to 0. At this time, the second layer of neural network data is obtained
        # x = F.relu(self.hidden(x))
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        # Get the third layer neural network output data
        x = self.predict(x)
        return x


class ParameterNet:

    def __init__(self,parameter = None,train_specs = None,train_pars = None,net_model = None,net_info = None):
        """
        

        Parameters
        ----------


        Notes
        -----


        Examples
        --------


        """
        self.train_spectra =  torch.from_numpy(train_specs).float()
        self.param_names = [par for par in train_pars.keys() if parameter in par]
        self.train_params = torch.from_numpy(np.asarray([[train_pars[par][i] for par in self.param_names] for i in range(len(train_pars[self.param_names[0]]))])).float()
        # self.spectra_model = spectra_model

        if net_model == None:
            # if (parameter == 'center') or (parameter =='sigma'):
            self.net_model = DeepNeuralNet(self.train_spectra.shape[1], 500, 1000, 700, 500, 200, 50, 4)

            # elif parameter == 'amplitude':
            #     self.net_model = NeuralNet(self.train_spectra.shape[1], 100, 4)
        
        else:
            self.net_model = net_model


    def train(self,n_epochs = 10,learn_rate = 0.01,optimizer = None):
        """
        Train the model
        """
        if optimizer is None:
            optimizer = torch.optim.Adam(self.net_model.parameters(), lr=learn_rate)

        testloss = []
        trainloss = []

        for epoch in tqdm(range(n_epochs)):
            trainloss.append(self._train_epoch(optimizer,10000,epoch))

        plt.plot(trainloss)

    def _train_epoch(self,optimizer, batch_size, epoch, steps_per_epoch=20):
        """
        Train an epoch of the net_model

        Parameters
        ----------


        Notes
        -----


        Examples
        --------


        """
    # Switch model to training mode. This is necessary for layers like dropout, batchnorm etc which behave differently in training and evaluation mode
        self.net_model.train()
        train_total = 0
        train_correct = 0
        self.loss_func = torch.nn.MSELoss()

    # We loop over the data iterator, and feed the inputs to the network and adjust the weights.
        for batch_idx, i in enumerate(range(0, len(self.train_spectra), batch_size)):
            if batch_idx > steps_per_epoch:
                break
                
            batch_spectra = self.train_spectra[i:i+batch_size]
            batch_params = self.train_params[i:i+batch_size]

            # Reset the gradients to 0 for all learnable weight parameters
            optimizer.zero_grad()

            # Forward pass: Pass spectra batch data from training dataset
            outputs = self.net_model(batch_spectra)

            # Define our loss function, and compute the loss
            loss = self.loss_func(outputs, batch_params)

            # Backward pass: compute the gradients of the loss w.r.t. the model's parameters
            loss.backward()

            # Update the neural network weights
            optimizer.step()

        print('Epoch [{}], Loss: {}, '.format(epoch, loss.item()), end='\n')
        
        return loss

    
    def predict(self,spectra):

        self.net_model.eval()
        _predict_pars = self.net_model(torch.from_numpy(spectra).float()).detach().numpy()
        predict_pars = {}

        for par in enumerate(self.param_names):
            # print(par)
            predict_pars[par[1]] = _predict_pars[par[0]]

        return predict_pars


    # def check_spectra(self,hellaspecs,params,testidx):
    #     """
    #     Check the spectra from the neural network parameter outputs against a spectra from a 
    #     HellaSpectra object

    #     Parameters
    #     ----------


    #     Notes
    #     -----


    #     Examples
    #     --------


    #     """
        
    #     spec_test = hellaspecs.spectra[testidx]

    #     self.net_model.eval()
    #     testpars = self.net_model(torch.from_numpy(spec_test).float()).detach().numpy()

    #     for par in enumerate(self.param_names):
    #         if 'sigma' in par[1]:
    #             print(par[1],hellaspecs.df_params[par[1]].iloc[testidx],testpars[par[0]])
    #             params[par[1]].set(testpars[par[0]])
                
    #         elif 'center' in par[1]:
    #             print(par[1],hellaspecs.df_params[par[1]].iloc[testidx],testpars[par[0]])
    #             params[par[1]].set(testpars[par[0]])
                
    #         else:
    #             print(par[1],hellaspecs.df_params[par[1]].iloc[testidx],testpars[par[0]])
    #             params[par[1]].set(testpars[par[0]])

    #     for par in [p for p in params.keys() if p not in self.param_names]:
    #         params[par].set(hellaspecs.df_params[par].iloc[testidx])


    #     plt.plot(spec_test,'o')
    #     plt.plot(self.spectra_model.eval(params = params,x= hellaspecs.energy))







class SpectraModelNet:

    def __init__(self,spectra_model,parameters,spec_train,par_train):
        """
        

        Parameters
        ----------


        Notes
        -----


        Examples
        --------


        """
        self.spectra_model = spectra_model
        self.params = parameters
        self.spec_train = spec_train
        self.par_train = par_train

        self.amp_net = ParameterNet(parameter = 'amplitude',train_specs = self.spec_train, train_pars = self.par_train)
        self.center_net = ParameterNet(parameter = 'center',train_specs = self.spec_train, train_pars = self.par_train)
        self.sigma_net = ParameterNet(parameter = 'sigma',train_specs = self.spec_train, train_pars = self.par_train)



    def predict(self,spectra):

        predict_pars = {}
        predict_pars.update(self.amp_net.predict(spectra))
        predict_pars.update(self.center_net.predict(spectra))
        predict_pars.update(self.sigma_net.predict(spectra))

        return predict_pars


    def check_spectra(self,spectra,energy):

        predict_pars = self.predict(spectra)
        for par,val in predict_pars.items():
            self.params[par].set(val)

        fig, ax = plt.subplots()
        ax.plot(energy,spectra,'o')
        ax.plot(energy,self.spectra_model.eval(params = self.params,x= energy))

        return fig,ax

    def save_model(self,path):
        torch.save({
            'amplitude': self.amp_net.net_model,\
            'center': self.center_net.net_model,\
            'sigma': self.sigma_net.net_model,\
                }, path)

    def load_model(self,path):

        parameter_models = torch.load(path)

        self.amp_net = ParameterNet(parameter = 'amplitude', net_model = parameter_models['amplitude'])
        self.amp_net.net_model.eval()
        self.center_net = ParameterNet(parameter = 'center', net_model = parameter_models['center'])
        self.center_net.net_model.eval()
        self.sigma_net = ParameterNet(parameter = 'sigma', net_model = parameter_models['sigma'])
        self.sigma_net.net_model.eval()
