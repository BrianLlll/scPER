#!/usr/bin/env python
# coding: utf-8
## Author: Bingrui Li
## Date: Mar 22 2023
### Script for training adversarial deconfounding autoencoder 
### Code is modified from https://gitlab.cs.washington.edu/abdincer/ad-ae/-/blob/master/KMPLOT_BRCA_EXPRESSION/Adversarial_Deconfounder_AE_Generate_Embeddings.py

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

parser = argparse.ArgumentParser("inputs")
parser.add_argument("scRNA", help="Imputed scRNA-seq data from preprocess step.")
parser.add_argument("label", help="labels for the scRNA-seq data that need to remove the bias")
parser.add_argument("bulk", help="Expression matrix for bulk samples")
args = parser.parse_args()


print(f"sklearn: {sk.__version__}")
print(f"pandas: {pd.__version__}")
print(f"keras: {ke.__version__}")


#Read single cell reference data

input_df = pd.read_csv(args.scRNA, index_col=0).T
input_df = pd.DataFrame(input_df.values, index = input_df.index.astype(str))


###onehot encoding of batch label
pheno_df=pd.read_csv(args.label, index_col = 0)
pheno_df=pheno_df.dropna()
pheno_type=len(pheno_df.iloc[:,0].unique().tolist())
pheno_df=pd.get_dummies(pheno_df)


X = input_df.sort_index()
Z = pheno_df.sort_index()

print("X ", X.shape)
print("Y ", Z.shape)

#Train and test splits
X_train, X_test, Z_train, Z_test = train_test_split(X, Z, test_size=0.2, random_state=12345)

#Standardize the data
scaler = StandardScaler().fit(X_train)
scale_df = lambda df, scaler: pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)
X_train = X_train.pipe(scale_df, scaler) 
X_test = X_test.pipe(scale_df, scaler)

print("X ", X_train.shape)
print("Y ", Z_train.shape)


#Class for adversarial training
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

        
adv_model = AdversarialConfounderAutoencoder(n_features=X_train.shape[1], pheno_n=pheno_type,
                                       latent_dim = 100, lambda_val = 1.0, random_seed = 1)

# pre-train both adverserial and classifier networks
adv_model.pretrain(X_train, Z_train, 
                    validation_data=(X_test, Z_test), epochs=10)

# adverserial train on train set and validate on test set
adv_model.fit(X_train, Z_train, 
        validation_data=(X_test, Z_test),
        T_iter = 1000)

#Generate embedding for single cells
embedding = adv_model._encoder.predict(X)
embedding_df = pd.DataFrame(embedding, index = X.index)
embedding_df.to_csv('./ADAE_' + str(100) + '_latents' + '.tsv',sep='\t')


##embeding for bulk samples
bulk_df = pd.read_csv(args.bulk, index_col=0).T


encoded_bulk_df = adv_model._encoder.predict(bulk_df)
encoded_bulk_df = pd.DataFrame(encoded_bulk_df, index=bulk_df.index)

encoded_bulk_df.columns.name = 'latent'
encoded_bulk_df.columns = encoded_bulk_df.columns + 1
encoded_bulk_df.T.to_csv('./bulk_100_latents.tsv', sep='\t')


#Record models
model_json = adv_model._encoder.to_json()
with open('/ADE_models/ADE_encoder_SC' + str(100) + '_Latents'+'.json', "w") as json_file:
    json_file.write(model_json)
adv_model._encoder.save_weights('./ADE_encoder_SC' + str(100) + '_Latents'+'.h5')
print("Saved model to disk")

model_json = adv_model._decoder.to_json()
with open('/ADE_models/ADE_decoder_SC' + str(100) + '_Latents'+'.json', "w") as json_file:
    json_file.write(model_json)
adv_model._decoder.save_weights('./ADE_decoder_SC' + str(100) + '_Latents'+'.h5')
print("Saved model to disk")

