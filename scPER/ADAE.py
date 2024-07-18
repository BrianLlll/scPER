#Class for adversarial training
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from sklearn.utils.class_weight import compute_class_weight
from numpy.random import seed
import tensorflow
import keras as ke
import keras.backend as K
from keras.layers import Input, Dense, Dropout
from keras.models import Model
create_gif = False
import sys
import argparse

class AdversarialConfounderAutoencoder(object):

    def __init__(self, n_features, pheno_n, latent_dim, lambda_val, random_seed):
        
        #Set random seeds
        seed(123456 * random_seed)
        tensorflow.random.set_seed(123456 * random_seed)

        #Set values
        self.lambda_val = lambda_val
        self.latent_dim = latent_dim
        
        #Define inputs
        ae_inputs = Input(shape=(n_features,)) 
        adv_inputs = Input(shape=(latent_dim,))
        
         #Define autoencoder net
        [ae_net, encoder_net, decoder_net] = self._create_autoencoder_net(ae_inputs, n_features, latent_dim) 
        print("AE net ")
        ae_net.summary()
        print("Encoder net ")
        encoder_net.summary()
        print("Decoder net ")
        decoder_net.summary()
            
        #Define adversarial net
        adv_net = self._create_adv_net(adv_inputs,pheno_n)
        
        #Turn on/off network weights
        self._trainable_ae_net = self._make_trainable(ae_net) 
        self._trainable_adv_net = self._make_trainable(adv_net) 
        
        #Compile models
        self._ae = self._compile_ae(ae_net) 
        self._encoder =  self._compile_encoder(encoder_net) 
        self._decoder =  self._compile_decoder(decoder_net) 
        self._ae_w_adv = self._compile_ae_w_adv(ae_inputs, ae_net, encoder_net, adv_net) 
        self._adv = self._compile_adv(ae_inputs, ae_net, encoder_net, adv_net)
        
        print("Autoencoder net with adv ")
        self._ae_w_adv.summary()
        
        #Define metrics
        self._val_metrics = None
        self._fairness_metrics = None
        
    #Freeze layers if network
    def _make_trainable(self, net):
        def make_trainable(flag):
            net.trainable = flag
            for layer in net.layers:
                layer.trainable = flag
        return make_trainable
       
    #Method for defining autoencoder network
    def _create_autoencoder_net(self, inputs, n_features, latent_dim):
        
        #Encoder
        dense1 = Dense(500, activation='relu')(inputs)
        dropout1 = Dropout(0.1)(dense1)
        latent_layer = Dense(latent_dim)(dropout1)
        
        #Decoder
        dense2 = Dense(500, activation='relu')
        dropout2 = Dropout(0.1)
        outputs = Dense(n_features)
        
        decoded = dense2(latent_layer)
        decoded = dropout2(decoded)
        decoded = outputs(decoded)
        
        autoencoder = Model(inputs=[inputs], outputs=[decoded], name = 'autoencoder')
        encoder = Model(inputs=[inputs], outputs=[latent_layer], name = 'encoder')
        
        #Define decoder
        decoder_input = Input(shape=(latent_dim, )) 
        decoded = dense2(decoder_input)
        decoded = dropout2(decoded)
        decoded = outputs(decoded)
        decoder = Model(inputs = decoder_input, outputs=[decoded],  name = 'decoder')
        
        return [autoencoder, encoder, decoder]
     
    #Method for defining adversarial network  
    def _create_adv_net(self, inputs, pheno_n):
        dense1 = Dense(100, activation='relu')(inputs)
        dense2 = Dense(100, activation='relu')(dense1)
        outputs = Dense(pheno_n, activation='softmax')(dense2)
        return Model(inputs=[inputs], outputs = [outputs],  name = 'adversary')

    #Compile model
    def _compile_ae(self, ae_net):
        ae = ae_net
        self._trainable_ae_net(True)
        ae.compile(loss='mse', metrics = ['mse'], optimizer='adam')
        return ae
    
    #Compile modelModels)
    
    def _compile_encoder(self, encoder_net):
        ae = encoder_net
        self._trainable_ae_net(True)
        ae.compile(loss='mse', metrics = ['mse'], optimizer='adam')
        return ae
      
    #Compile model
    def _compile_decoder(self, decoder_net):
        ae = decoder_net
        self._trainable_ae_net(True) 
        ae.compile(loss='mse', metrics = ['mse'], optimizer='adam')
        return ae
    
    def auroc(y_true, y_pred):
        return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

    #Compile autoencoder with adv loss
    #The model takes input features as input
    #Outputs classifier prediction + adversarial prediction from the classifier prediction
    def _compile_ae_w_adv(self, inputs, ae_net, encoder_net, adv_net):
        ae_w_adv = Model(inputs=[inputs], outputs = [ae_net(inputs)] + [adv_net(encoder_net(inputs))])
        self._trainable_ae_net(True) #classifier is trainable
        self._trainable_adv_net(False) #Freeze the adversary
        loss_weights = [1., -1 * self.lambda_val] #classifier loss - adversarial loss
        #Now compile the model with two losses and defined weights
        ae_w_adv.compile(loss=['mse', 'categorical_crossentropy'], 
                          metrics=['mse', 'accuracy'], 
                          loss_weights=loss_weights,
                          optimizer='adam')
        return ae_w_adv

    #Compile adversarial model
    #Takes input features and outputs adversarial prediction
    def _compile_adv(self, inputs, ae_net, encoder_net, adv_net):
        adv = Model(inputs=[inputs], outputs=adv_net(encoder_net(inputs)))
        self._trainable_ae_net(False) #Freeze the classifier
        self._trainable_adv_net(True) #adversarial net is trainable
        adv.compile(loss=['categorical_crossentropy'], 
                    metrics = ['accuracy'], optimizer='adam') 
        return adv
        
    #Pretrain all models
    def pretrain(self, x, z, validation_data=None, epochs=10):
        self._trainable_ae_net(True)
        self._ae.fit(x.values, x.values, epochs=epochs)
        self._trainable_ae_net(False)
        self._trainable_adv_net(True)
        
        if validation_data is not None:
            x_val, z_val = validation_data

        self._adv.fit(x.values, z.values,
                      validation_data = (x_val.values, z_val.values),
                        epochs=epochs, verbose=2)
        
    #Now do adversarial training
    def fit(self, x, z, validation_data=None, T_iter=250, batch_size=128):
        
        if validation_data is not None:
            x_val, z_val = validation_data

        self._val_metrics = pd.DataFrame()
        self._train_metrics = pd.DataFrame()
        
        #Go over all iterations
        for idx in range(T_iter):
            print("Iter ", idx)
            
            if validation_data is not None:
                
                #Predict with encoder
                x_pred = pd.DataFrame(self._ae.predict(x_val), index = x_val.index)
                self._val_metrics.loc[idx, 'MSE'] = mean_squared_error(x_val, x_pred)
                
            # train adversary
            self._trainable_ae_net(False)
            self._trainable_adv_net(True)
            print("Training adversary...")
            history = self._adv.fit(x.values, z.values, 
                                    validation_data = (x_val.values, z_val.values),
                                    batch_size=batch_size, epochs=1, verbose=1)
            self._train_metrics.loc[idx, 'Adversary accuracy'] = history.history['accuracy'][0]
            self._val_metrics.loc[idx, 'Adversary accuracy'] = history.history['val_accuracy'][0]
            
            # train autoencoder
            self._trainable_ae_net(True)
            self._trainable_adv_net(False)
            indices = np.random.permutation(len(x))[:batch_size]
            print("Training adversarial autoencoder...")
            history = self._ae_w_adv.fit(x.values[indices],
                                     [x.values[indices]] + [z.values[indices]],
                                     batch_size=batch_size, epochs=1, verbose=1,
                                     validation_data = (x_val.values,
                                     [x_val.values] + [z_val.values]))
            
            print("Autoencoder loss ",  history.history)
            keys = self._ae_w_adv.metrics_names
            
            #Record of interest results
            self._train_metrics.loc[idx, 'Total autoencoder loss'] = history.history['loss'][0]
            self._val_metrics.loc[idx, 'Total autoencoder loss'] = history.history['val_loss'][0]
             
            self._train_metrics.loc[idx, 'Autoencoder MSE'] = history.history['autoencoder_mse'][0]
            self._val_metrics.loc[idx, 'Autoencoder MSE'] = history.history['val_autoencoder_mse'][0]
                
            self._train_metrics.loc[idx, 'Adversary accuracy'] = history.history['adversary_accuracy'][0]
            self._val_metrics.loc[idx, 'Adversary accuracy'] = history.history['val_adversary_accuracy'][0]
              
    
        #Create plot of losses
        fig, ax = plt.subplots()
        fig.set_size_inches(60, 15)

        SMALL_SIZE = 50
        MEDIUM_SIZE = 60
        BIGGER_SIZE = 70

        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        plt.plot(self._train_metrics['Total autoencoder loss'], 
                 label = 'Total autoencoder training loss', lw = 5, color = '#27ae60')
        plt.plot(self._val_metrics['Total autoencoder loss'], 
                 label = 'Total autoencoder validation loss', lw = 5, color = '#f39c12')
        
        plt.xlabel('epochs')
        
        # Don't allow the axis to be on top of your data
        ax.set_axisbelow(True)
        ax.minorticks_on()
        ax.grid(which='major', linestyle='-', linewidth='0.5', color='black')
        ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        ax.legend(bbox_to_anchor=(1.1, 1.05))
        
        plt.show()
        
        
        
        #Create plot of losses
        fig, ax = plt.subplots()
        fig.set_size_inches(60, 15)

        SMALL_SIZE = 50
        MEDIUM_SIZE = 60
        BIGGER_SIZE = 70

        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        plt.plot(self._train_metrics['Autoencoder MSE'], 
                 label = 'Autoencoder training MSE', lw = 5, color = '#3498db')
        plt.plot(self._val_metrics['Autoencoder MSE'], 
                 label = 'Autoencoder validation MSE', lw = 5, color = '#e74c3c')
           
        plt.plot(self._train_metrics['Adversary accuracy'], 
                 label = 'Adversary training accuracy', lw = 5, color = '#16a085')
        plt.plot(self._val_metrics['Adversary accuracy'], 
                 label = 'Adversary validation accuracy', lw = 5, color = '#9b59b6')
         
            
        plt.xlabel('epochs')
        
        # Don't allow the axis to be on top of your data
        ax.set_axisbelow(True)
        ax.minorticks_on()
        ax.grid(which='major', linestyle='-', linewidth='0.5', color='black')
        ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        ax.legend(bbox_to_anchor=(1.1, 1.05))
        
        plt.show()
