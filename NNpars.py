import numpy as np
import lmfit as lm
from copy import deepcopy as dc
import XPyS
from .helper_functions import index_of, guess_from_data
from IPython import embed as shell

from tqdm.notebook import tqdm

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import torch
from torch import nn


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

    def __init__(self,train_energy = None, parameter = None,train_specs = None,train_pars = None,net_model = None,param_names = None):
        """
        

        Parameters
        ----------
        train_energy: array
            the energy values of the spectra used to train the network. This is essential since the energy values of the spectra to predict
            will be different in which case the spectra must be interpolated at the same energy values used to train the network.

        parameter: str
            name of type of parameter, ex: "amplitude","center" or "sigma". This will create a list of the parameter names that are
            fit in the neural network. Only used if building a new net.If a net is loaded the parameter names will be loaded when 
            the net_model is loaded.

        train_specs: 2d array
            array of spectra to train the neural network on. Rows are spectra.

        train_pars: dict
            dictionary of the training parameters. key: parameeter, value: list or array of the parameters corresponding to the training spectra

        net_model = None,
        param_names = None

        Notes
        -----


        Examples
        --------


        """

        # The energy used to train the parameternet must be stored to compare it to the energy for the auto initialization. If they are different
        # the spectra must be interpolated
        self.train_energy = train_energy

        # param_names is used for loading a model. It should be left as None if training
        if param_names:
            self.param_names = param_names
        

        if train_pars:
            self.param_names = [par for par in train_pars.keys() if parameter in par]
            self.train_params = torch.from_numpy(np.asarray([[train_pars[par][i] for par in self.param_names] for i in range(len(train_pars[self.param_names[0]]))])).float()
        
        if np.any(train_specs):
            self.train_spectra =  torch.from_numpy(train_specs).float()

        # If net_model is not specified then the default neural net is loaded
        # if net_model == None:
        #     self.load_net_model(DeepNeuralNet(self.train_spectra.shape[1], 500, 1000, 700, 500, 200, 50, len(self.param_names)))
        
        if net_model != None:
            self.load_net_model(net_model)

    def load_net_model(self,NeuralNetModel):
        self.net_model = NeuralNetModel

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

    
    def predict(self,spectra,energy):
        """
        
        """
        if not np.all(energy == self.train_energy):
            print('interpolating...')
            spectra = self.interp_spectra(spectra,energy)

        self.net_model.eval()
        _predict_pars = self.net_model(torch.from_numpy(spectra).float()).detach().numpy()
        predict_pars = {}

        for par in enumerate(self.param_names):

            predict_pars[par[1]] = _predict_pars[par[0]]

        return predict_pars

    def interp_spectra(self,spectra,energy):
        f = interp1d(energy, spectra,kind = 'cubic',fill_value = 'extrapolate')
        ynew = f(self.train_energy)

        return ynew


class SpectraModelNet:

    def __init__(self,train_energy = None,spectra_model = None,parameters = None,spec_train = None,par_train = None):
        """
        

        Parameters
        ----------


        Notes
        -----


        Examples
        --------


        """

        self.train_energy = train_energy

        if spectra_model:
            self.spectra_model = spectra_model
        if parameters:
            self.params = parameters

        self.spec_train = spec_train
        self.par_train = par_train


    def initialize_parnets(self):
        self.amp_net = ParameterNet(train_energy = self.train_energy,parameter = 'amplitude',train_specs = self.spec_train, train_pars = self.par_train)
        self.center_net = ParameterNet(train_energy = self.train_energy,parameter = 'center',train_specs = self.spec_train, train_pars = self.par_train)
        self.sigma_net = ParameterNet(train_energy = self.train_energy,parameter = 'sigma',train_specs = self.spec_train, train_pars = self.par_train)



    def predict(self,spectra,energy):

        predict_pars = {}
        predict_pars.update(self.amp_net.predict(spectra,energy))
        predict_pars.update(self.center_net.predict(spectra,energy))
        predict_pars.update(self.sigma_net.predict(spectra,energy))

        return predict_pars

    def update_params(self,spectra,energy):
        predict_pars = self.predict(spectra,energy)
        for par,val in predict_pars.items():
            self.params[par].set(val)       

    def check_spectra(self,spectra,energy):

        self.update_params(spectra,energy)

        fig, ax = plt.subplots()
        ax.plot(energy,spectra,'o')
        ax.plot(energy,self.spectra_model.eval(params = self.params,x= energy))

        return fig,ax

    def save_model(self,path):
        torch.save({
            'energy' : self.train_energy,\
            'params' : self.params,\
            'spectra_model' : self.spectra_model.dumps(),\
            'amplitude': [self.amp_net.param_names, self.amp_net.net_model],\
            'center': [self.center_net.param_names, self.center_net.net_model],\
            'sigma': [self.sigma_net.param_names, self.sigma_net.net_model],\
                }, path)

    def load_model(self,path):

        parameter_models = torch.load(path)

        self.train_energy = parameter_models['energy']
        self.params = parameter_models['params']
        self.spectra_model = lm.model.Model(lambda x: x)
        self.spectra_model = self.spectra_model.loads(parameter_models['spectra_model'])
        self.amp_net = ParameterNet(train_energy = parameter_models['energy'], net_model = parameter_models['amplitude'][1],param_names = parameter_models['amplitude'][0])
        self.amp_net.net_model.eval()
        self.center_net = ParameterNet(train_energy = parameter_models['energy'],net_model = parameter_models['center'][1],param_names = parameter_models['center'][0])
        self.center_net.net_model.eval()
        self.sigma_net = ParameterNet(train_energy = parameter_models['energy'],net_model = parameter_models['sigma'][1],param_names = parameter_models['sigma'][0])
        self.sigma_net.net_model.eval()
