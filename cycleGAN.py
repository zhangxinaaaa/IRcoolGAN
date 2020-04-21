#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 16:58:26 2020

@author: xen
"""

import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv1D, Concatenate, Activation, LeakyReLU, Conv2DTranspose, Lambda
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.utils import plot_model
import numpy as np
from tensorflow import config
gpus = config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the fourth GPU
        config.experimental.set_visible_devices(gpus[0], 'GPU')

        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            config.experimental.set_memory_growth(gpu, True)
        logical_gpus = config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)






# def Conv1DTranspose(self, input_tensor, filters, kernel_size, strides=2, padding='same', 
#                     kernel_initializer = RandomNormal(mean=0., stddev=0.02), name = 'Conv1dTranspose', activation = 'relu'):
#     """
#         input_tensor: tensor, with the shape (batch_size, time_steps, dims)
#         filters: int, output dimension, i.e. the output tensor will have the shape of (batch_size, time_steps, filters)
#         kernel_size: int, size of the convolution kernel
#         strides: int, convolution step size
#         padding: 'same' | 'valid'
#     """
#     x = Lambda(lambda x: K.expand_dims(x, axis=1))(input_tensor)
#     x = Conv2DTranspose(filters=filters, kernel_size=(1, kernel_size), strides=(1, strides), padding=padding,
#                         kernel_initializer = kernel_initializer, name = name, 
#                         activation = activation)(x)
#     x = Lambda(lambda x: K.squeeze(x, axis=1))(x)
#     return x

# def Conv1DTranspose(input_layer, filters, kernel_size, strides=4, padding='same', 
#                     kernel_initializer = RandomNormal(mean=0., stddev=0.02), name = 'Conv1dTranspose', **kwargs): 
#     g = Conv2DTranspose(
#           filters, (1, kernel_size), (1, strides), padding, kernel_initializer = kernel_initializer, **kwargs)

#     def call(self, x):
#         x = K.expand_dims(x, axis=1)
#         x = self.conv2dtranspose(x)
#         x = K.squeeze(x, axis=1)
#         return x
#     def get_config(self):
#         return {'conv2dtranspose': self.conv2dtranspose}

def Conv1DTranspose(input_tensor, filters, kernel_size, strides=2, padding='same', 
                kernel_initializer = RandomNormal(mean=0., stddev=0.02), name = 'Conv1dTranspose', activation = 'relu'):
    """
    input_tensor: tensor, with the shape (batch_size, time_steps, dims)
    filters: int, output dimension, i.e. the output tensor will have the shape of (batch_size, time_steps, filters)
    kernel_size: int, size of the convolution kernel
    strides: int, convolution step size
    padding: 'same' | 'valid'
    """
    
    x = Conv2DTranspose(filters=filters, kernel_size=(1, kernel_size), strides=(1, strides), padding=padding, 
                    kernel_initializer = kernel_initializer, name = name, 
                    activation = activation)(K.expand_dims(input_tensor, axis=1))
    x = Lambda(lambda x: K.squeeze(x, axis=1))(x)
    return x



class cyclegan():
    def __init__(self
        , X_trainA
        , X_trainB
        , working_dir
        , D_learning_rate
        , batch_size
        , sample_rate_audio
        , adversarial_learningrate
        , kernel_size
        , IR_shape
        , epoch
        , total_epochs
        , print_models
        , soft_labels
        , idloss_weight):

        self.name = 'CycleGAN'
        self.working_dir = working_dir
        #critic and generator params
        self.D_learning_rate = D_learning_rate
        self.critic_filters = [64, 128, 256, 512, 1024, 2048]  
        
        self.total_epochs = total_epochs - epoch
        self.idloss_weight = idloss_weight
       
        # training epoch 
        self.epoch = epoch
        # training data
        self.X_trainA = X_trainA
        self.X_trainB = X_trainB
        # training data indices
        self.idsA = np.arange(X_trainA.shape[0])
        self.idsB = np.arange(X_trainB.shape[0])
        # input shape
        self.IR_shape = IR_shape
        # composite model learning rate (default = 0.0002)
        self.adv_LR = adversarial_learningrate
        #kernel size (int)
        self.kernel_size = kernel_size
        #audio sample rate (int)
        self.sample_rate_audio = sample_rate_audio
        #batch size (int)
        self.batch_size = batch_size
        #weights
        # self.grad_weight = 10
        # weight initialization
        self.weight_init = RandomNormal(mean=0., stddev=0.02)
        #losses
        self.d_losses = []
        self.g_losses = []
        # number of residual network layers 
        if self.IR_shape == 16384:
            self.n_resnet = 6
        else: 
            self.n_resnet = 9
            
        self.print_models = print_models

        self.soft_labels = soft_labels
        
        

    
    # generator a resnet block
    def resnet_block(self, n_filters, input_layer, i = ''):

        # first layer convolutional layer
        g = Conv1D(n_filters, 9, padding= 'same', kernel_initializer=self.weight_init, name = 'G_resnet_conv1_' + i)(input_layer)
        g = InstanceNormalization(axis=-1)(g)
        g = Activation('relu')(g)
        # second convolutional layer
        g = Conv1D(n_filters, 9, padding= 'same', kernel_initializer=self.weight_init, name = 'G_resnet_conv2_' + i)(g)
        g = InstanceNormalization(axis=-1)(g)
        # concatenate merge channel-wise with input layer
        g = Concatenate()([g, input_layer])
        return g
    
    # define the discriminator model
    def define_discriminator(self):
        n_ch = 1 # TODO: add channels to audio
        # source image input
        in_audio = Input(shape= (self.IR_shape, n_ch), name = 'Input_D') 
            
        # C64
        d = Conv1D(self.critic_filters[0], self.kernel_size, strides= 4, padding='same', kernel_initializer= self.weight_init , name = 'critic_conv_0')(in_audio) # kernel size (4,4) -> 16
        d = LeakyReLU(alpha=0.2)(d) 
       
        for i in range(1,4):   
            d = Conv1D(
                filters = self.critic_filters[i]
                , kernel_size = self.kernel_size
                , strides = 4
                , padding = 'same'
                , name = 'critic_conv_' + str(i)
                , kernel_initializer = self.weight_init
                )(d)
            d = InstanceNormalization(axis=-1)(d)
            d = LeakyReLU(alpha=0.2)(d)

        
        # second last output layer
        d = Conv1D(self.critic_filters[3], self.kernel_size, padding='same', kernel_initializer = self.weight_init , name = 'critic_conv_4')(d)
        d = InstanceNormalization(axis=-1)(d)
        d = LeakyReLU(alpha=0.2)(d)
        
        # patch output
        patch_out = Conv1D(1, self.kernel_size, padding='same', kernel_initializer = self.weight_init, name = 'Patch_out')(d)
       
        # define model
        model = Model(in_audio, patch_out) # expected output for 16384 input is 16384/16 = 2048
        
        
        # compile model
        model.compile(loss= 'mse', optimizer=Adam(lr= self.D_learning_rate, beta_1=0.5), loss_weights=[0.5])
        
        if self.print_models:
            plot_model(model, to_file= self.working_dir +'/discriminator_model_plot.png', show_shapes=True,
                show_layer_names=True)
    
        return model


    # define the standalone generator model
    def define_generator(self):
        # image input
        n_ch = 1 # TODO: add channels to audio
        in_audio = Input(shape= (self.IR_shape,n_ch), name='generator_input')
        
        # c7s1-64
        g = Conv1D(64, 49, padding= 'same', kernel_initializer=self.weight_init, name = 'c7s1-64')(in_audio)
        g = InstanceNormalization(axis=-1)(g)
        g = Activation('relu')(g)
        # d128
        g = Conv1D(128, 9, strides= 4, padding='same', kernel_initializer=self.weight_init , name = 'd128')(g)
        g = InstanceNormalization(axis=-1)(g)
        g = Activation('relu')(g)
        # d256
        if self.IR_shape == 32768:
            g = Conv1D(256, 9, strides= 2, padding='same', kernel_initializer=self.weight_init, name = 'd256')(g)
            g = InstanceNormalization(axis=-1)(g)
            g = Activation('relu')(g)
        else: 
            g = Conv1D(256, 9, strides= 4, padding='same', kernel_initializer=self.weight_init, name = 'd256')(g)
            g = InstanceNormalization(axis=-1)(g)
            g = Activation('relu')(g)
        # R256
        for i in range(self.n_resnet):
            g = self.resnet_block(256, g, str(i))
        # u128
        if self.IR_shape == 32768:
            g = Conv1DTranspose(g, 128, 9, strides= 2, padding='same', kernel_initializer=self.weight_init, name = 'u128')
            g = InstanceNormalization(axis=-1)(g)
            g = Activation('relu')(g)
        else:
            g = Conv1DTranspose(g, 128, 9, strides= 4, padding='same', kernel_initializer=self.weight_init, name = 'u128')
            g = InstanceNormalization(axis=-1)(g)
            g = Activation('relu')(g)
        # u64
        g = Conv1DTranspose(g, 64, 9, strides= 4, padding='same', kernel_initializer=self.weight_init, name = 'u64')
        g = InstanceNormalization(axis=-1)(g)
        g = Activation('relu')(g)
        # c7s1-1
        g = Conv1D(1, 49, padding='same', kernel_initializer= self.weight_init, name = 'c7s1-1')(g)
        g = InstanceNormalization(axis=-1)(g)
        out_audio = Activation('tanh')(g)
        # define model
        model = Model(in_audio, out_audio)
        
        if self.print_models:
            plot_model(model, to_file= self.working_dir + '/generator_model_plot.png', show_shapes=True,
                    show_layer_names=True) 
        return model

    # define a composite model for updating generators by adversarial and cycle loss
    def define_composite_model(self, g_model_1, d_model, g_model_2):
        n_ch = 1 # TODO: add channels to audio
        # ensure the model we✬re updating is trainable
        g_model_1.trainable = True
        # mark discriminator as not trainable
        d_model.trainable = False
        # mark other generator model as not trainable
        g_model_2.trainable = False
        # discriminator element
        input_gen = Input(shape= (self.IR_shape,n_ch))
        gen1_out = g_model_1(input_gen)
        output_d = d_model(gen1_out)
        # identity element
        input_id = Input(shape= (self.IR_shape,n_ch))
        output_id = g_model_1(input_id)
        # forward cycle
        output_f = g_model_2(gen1_out)
        # backward cycle
        gen2_out = g_model_2(input_id)
        output_b = g_model_1(gen2_out)
        # define model graph
        model = Model([input_gen, input_id], [output_d, output_id, output_f, output_b])
        # define optimization algorithm configuration
        opt = Adam(lr= self.adv_LR, beta_1=0.5)
        # compile model with weighting of least squares loss and L1 loss
        model.compile(loss=['mse', 'mae', 'mae', 'mae'], loss_weights=[1, self.idloss_weight, 10, 10],
        optimizer=opt)
        d_model.trainable = True
        return model    
    
    # select a batch of random samples, returns irs and target
    def generate_real_samples(self, dataset, patch_shape, cur_batch, train_a, n_samples = -1):
        if cur_batch == 0:
            ids = np.arange(dataset.shape[0])
            np.random.shuffle(ids)
            if train_a:
                self.idsA = np.arange(dataset.shape[0])
            else:
                self.idsB = np.arange(dataset.shape[0])
        else: 
            if train_a:
                ids = self.idsA
            else:
                ids = self.idsB
        cur_ids = ids[cur_batch * self.batch_size:(cur_batch+1)*self.batch_size] 
        if n_samples > 0:
            X = dataset[np.random.choice(ids, n_samples)]
        else:
            # real batch
            X = dataset[cur_ids]


        # X = np.array([None, X])

        # generate 'real' class labels (1)
        if self.soft_labels:
            # generate soft labels
            y = np.random.uniform(0.9, 1, size = (self.batch_size, patch_shape, 1))
        else:
            y = np.ones((self.batch_size, patch_shape, 1))
        return X, y
    
    
    # generate a batch of irs, returns irs and targets
    def generate_fake_samples(self, g_model, dataset, patch_shape):
       
        # generate fake instance
        X = g_model.predict(dataset)
        # create 'fake' class labels (0)
        # soft labels
        if self.soft_labels:
            y = np.random.uniform(0, 0.1, size = (len(X), patch_shape, 1))
        else:
            y = np.zeros((self.batch_size, patch_shape, 1))
        return X, y
    
    
    # update IR pool for fake IRs
    def update_ir_pool(self, pool, IRs, max_size=50):
        selected = list()
        for ir in IRs:
            if len(pool) < max_size:
                # stock the pool
                pool.append(ir)
                selected.append(ir)
            elif np.random.random() < 0.5:
                # use ir, but don't add it to the pool
                selected.append(ir)
            else:
                # replace an existing irs and use replaced irs
                ix = np.random.randint(0, len(pool))
                selected.append(pool[ix])
                pool[ix] = ir
        # print(np.size(selected))
        return np.asarray(selected)
                
    
    # train cyclegan models
    def train(self, d_model_A
              , d_model_B
              , g_model_AtoB
              , g_model_BtoA
              , c_model_AtoB
              , c_model_BtoA
              , save_interval
              , sample_interval):

        wdir = self.working_dir + '/weights'
        if os.path.isfile( wdir + '/D_loss.npy'):
            D_loss = list(np.load(wdir + '/D_loss.npy'))
            G_loss = list(np.load(wdir + '/D_loss.npy'))
            Forward = list(np.load(wdir + '/Forward.npy'))
            Backward = list(np.load(wdir + '/Backward.npy'))
            IDloss = list(np.load(wdir + '/IDloss.npy'))
        else:
            D_loss = []
            G_loss = []
            Forward = []
            Backward = []
            IDloss = []

        from tensorboardX import SummaryWriter
        import time
        
        named_tuple = time.localtime() # get struct_time
        time_string = time.strftime("%m-%d-%Y_%H.%M", named_tuple)
        
        log_dir = "logs/{}/".format(time_string)

        
        
        writer = SummaryWriter(log_dir)

        
        # determine the output square shape of the discriminator
        n_patch = d_model_A.output_shape[1]
        # unpack dataset
        trainA, trainB = self.X_trainA, self.X_trainB
        # prepare ir pool for fakes
        poolA, poolB = list(), list()
        # calculate the number of batches per training epoch
        bat_per_epo = int(len(trainA) / self.batch_size)
        #current batch
        cur_batch = 0
        # current epoch
        epoch = self.epoch
        # calculate the number of training iterations
        n_steps = bat_per_epo * self.total_epochs
        
        
        print('-' * 80)

        print('\nNumber of batches per epoch: {}, total iterations to run for: {}, discriminator patch size: {} samples/patch'.format(bat_per_epo, n_steps, n_patch))
            
        print('-' * 80)
        print('Training has started. Please use \'tensorboard --logdir={}\' to monitor.'.format(log_dir))

        print('-' * 80)
        
        # manually enumerate epochs
        for i in range(n_steps):
            # select a batch of real samples
            X_realA, y_realA = self.generate_real_samples(trainA, n_patch, cur_batch, 1)
            X_realB, y_realB = self.generate_real_samples(trainB, n_patch, cur_batch, 0)
            # generate a batch of fake samples
            X_fakeA, y_fakeA = self.generate_fake_samples(g_model_BtoA, X_realB, n_patch)
            X_fakeB, y_fakeB = self.generate_fake_samples(g_model_AtoB, X_realA, n_patch)
            # update fakes from pool
            X_fakeA = self.update_ir_pool(poolA, X_fakeA)
            X_fakeB = self.update_ir_pool(poolB, X_fakeB)
            # update generator B->A via adversarial and cycle loss
            g_loss2, idlossg2, forward2, backward2, _ = c_model_BtoA.train_on_batch([X_realB, X_realA], [y_realA,
                X_realA, X_realB, X_realA])
            # update discriminator for A -> [real/fake]
            dA_loss1 = d_model_A.train_on_batch(X_realA, y_realA)
            dA_loss2 = d_model_A.train_on_batch(X_fakeA, y_fakeA)
            # update generator A->B via adversarial and cycle loss
            g_loss1, idlossg1, forward1, backward1, _ = c_model_AtoB.train_on_batch([X_realA, X_realB], [y_realB,
                X_realB, X_realA, X_realB])
            # update discriminator for B -> [real/fake]
            dB_loss1 = d_model_B.train_on_batch(X_realB, y_realB)
            dB_loss2 = d_model_B.train_on_batch(X_fakeB, y_fakeB)
            # summarize performance
            print('>iteration: %d, d_lossesA[%.3f,%.3f] d_lossesB[%.3f,%.3f] glosses[%.3f,%.3f]' % (i+1, dA_loss1,dA_loss2,
                dB_loss1,dB_loss2, g_loss1,g_loss2))
            print('current batch is: ', cur_batch, '/', bat_per_epo)
            if cur_batch == 0:
                if epoch % save_interval == 0:
                    self.save_models(g_model_AtoB, g_model_BtoA, d_model_A, d_model_B, c_model_AtoB, c_model_BtoA, epoch)

                if epoch % sample_interval == 0:
                    # total discriminator loss of A                
                    dA_loss = 0.5 * np.add(dA_loss1, dA_loss2)
                    # total discriminator loss of B             
                    dB_loss = 0.5 * np.add(dB_loss1, dB_loss2)

                    D_loss.append([dB_loss, dA_loss])
                    G_loss.append([g_loss1, g_loss2])
                    Forward.append([forward1, forward2])
                    Backward.append([backward1,backward2])
                    IDloss.append([idlossg1, idlossg2])
                    self.plot_loss(D_loss, G_loss, Forward, Backward, IDloss, epoch)
                    self.sample_IRs(g_model_AtoB, trainA, 'Image Source Method to Real RIRs', epoch, A_set = True)
                    self.sample_IRs(g_model_BtoA, trainB, 'Real to Image Source Method RIRs', epoch,  A_set = False)
                epoch += 1            
        
            cur_batch = (cur_batch + 1) % bat_per_epo


            
            
            # total discriminator loss of A                
            dA_loss = 0.5 * np.add(dA_loss1, dA_loss2)
            
            # total discriminator loss of B             
            dB_loss = 0.5 * np.add(dB_loss1, dB_loss2)
            
            # Total disciminator loss
            d_loss = 0.5 * np.add(dA_loss, dB_loss)

            
            # use tensorboard to monitor the Discriminator losses
            writer.add_scalar('D loss A', dA_loss, i)
            writer.add_scalar('D loss B', dB_loss, i)
            writer.add_scalar('D loss total', d_loss, i)                
            # use tensorboard to monitor the Generator losses
            writer.add_scalar('Generator loss A to B', g_loss1, i)
            writer.add_scalar('Generator loss B to A', g_loss2, i)
            writer.add_scalar('G loss total', 0.5*np.add(g_loss1,g_loss2), i)

            G_AtoB_rms = np.sqrt(np.mean(np.square(X_fakeB), axis=1))
            G_BtoA_rms = np.sqrt(np.mean(np.square(X_fakeA), axis=1))

            xA_rms = np.sqrt(np.mean(np.square(X_realA), axis=1))
            xB_rms = np.sqrt(np.mean(np.square(X_realB), axis=1))

            
            # monitor rms distributions of G(z) and x on tensorboard to see if they start matching
            writer.add_histogram('G_AtoB_rms_batch', G_AtoB_rms, i)
            writer.add_histogram('G_BtoA_rms_batch', G_BtoA_rms, i)

            writer.add_histogram('xA_rms_batch', xA_rms, i)
            writer.add_histogram('xB_rms_batch', xB_rms, i)

            
            # Add first 3 Audio samples of batch and generated to tensorboard
            writer.add_audio('X Original (A) IR', X_realA, i, self.sample_rate_audio)
            writer.add_audio('X Original (B) IR',  X_realB, i, self.sample_rate_audio)
            writer.add_audio('X Generated (BtoA) IR',  X_fakeA, i, self.sample_rate_audio)
            writer.add_audio('X Generated (AtoB) IR',  X_fakeB, i, self.sample_rate_audio)

                  
    def plot_loss(self, D_loss, G_loss, Forward, Backward, IDloss, epoch):

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        directory = self.working_dir + '/preview'
        subdir = directory + '/epoch_'+ str(epoch)
        weights_direc = self.working_dir + '/weights'

        if not os.path.exists(directory):
            os.makedirs(directory)
    
        if not os.path.exists(subdir):
            os.makedirs(subdir)
    

        epochs = int(epoch/len(D_loss))*np.arange(0, len(D_loss))
        # A -> B
        fig, ax = plt.subplots(1, figsize=(10, 6), dpi = 180)
        ax.grid(axis='both', alpha = 0.4)
        fig.suptitle('Loss Values for Domain A to Domain B - Epoch {}'.format(epoch), fontsize=20)
        ax.plot(epochs, np.array(D_loss)[:,0], label = 'D loss')
        ax.plot(epochs, np.array(G_loss)[:,0], label = 'G loss')
        ax.plot(epochs, np.array(Forward)[:,0], label = 'forward cycle loss')
        ax.plot(epochs, np.array(Backward)[:,0], label = 'backward cycle loss')
        ax.plot(epochs, np.array(IDloss)[:,0], label = 'identity loss')
        ax.set_xlabel('Epoch', fontsize = 13)
        ax.set_ylabel('Loss', fontsize = 13)
        ax.legend()
        fig.savefig(subdir + '/Loss_AtoB.png')
        # B -> A
        fig2, ax2 = plt.subplots(1, figsize=(10, 6), dpi = 180)
        ax2.grid(axis='both', alpha = 0.4)
        fig2.suptitle('Loss Values for Domain B to Domain A - Epoch {}'.format(epoch), fontsize=20)
        ax2.plot(epochs, np.array(D_loss)[:,1], label = 'D loss')
        ax2.plot(epochs, np.array(G_loss)[:,1], label = 'G loss')
        ax2.plot(epochs, np.array(Forward)[:,1], label = 'forward cycle loss')
        ax2.plot(epochs, np.array(Backward)[:,1], label = 'backward cycle loss')
        ax2.plot(epochs, np.array(IDloss)[:,1], label = 'identity loss')
        ax2.set_xlabel('Epoch', fontsize = 13)
        ax2.set_ylabel('Loss', fontsize = 13)
        ax2.legend()
        fig2.savefig(subdir + '/Loss_BtoA.png')
        plt.close('all')

        np.save(weights_direc + '/D_loss', D_loss)
        np.save(weights_direc + '/G_loss', G_loss)
        np.save(weights_direc + '/Forward', Forward)
        np.save(weights_direc + '/Backward', Backward)
        np.save(weights_direc + '/IDloss', IDloss)

    #save model
    def save_models(self, g_model_AtoB, g_model_BtoA, d_model_A, d_model_B, c_model_AtoB, c_model_BtoA, epoch):
        
        weights_direc = self.working_dir + '/weights'
        model_direc = self.working_dir + '/models'
        if not os.path.exists(weights_direc):
            os.makedirs(weights_direc) 
        if not os.path.exists(model_direc):
            os.makedirs(model_direc) 
        
        # save generator models
        print('Saving models at epoch {}...'.format(epoch))
        g_model_AtoB.save(model_direc + '/g_model_AtoB_{}.h5'.format(str(epoch)))
        g_model_BtoA.save(model_direc + '/g_model_BtoA_{}.h5'.format(str(epoch)))
        
        # save weights to resume training
        g_model_AtoB.save_weights(weights_direc + '/g_model_AtoB_weights_{}.h5'.format(str(epoch)))
        g_model_BtoA.save_weights(weights_direc + '/g_model_BtoA_weights_{}.h5'.format(str(epoch)))        
        d_model_A.save_weights(weights_direc + '/d_model_A_weights_{}.h5'.format(str(epoch)))
        d_model_B.save_weights(weights_direc + '/d_model_B_weights_{}.h5'.format(str(epoch)))
        c_model_AtoB.save_weights(weights_direc + '/c_model_AtoB_weights_{}.h5'.format(str(epoch)))
        c_model_BtoA.save_weights(weights_direc + '/c_model_BtoA_weights_{}.h5'.format(str(epoch)))
        print('Models saved.')



    def sample_IRs(self, g_model, trainX, name, epoch, n_samples = 5, A_set = False):
        
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from scipy.io.wavfile import write as wavwrite
        
        # select a sample of input irs (ISM or real)
        data_IRs, _ = self.generate_real_samples(trainX, 1, 1, train_a = A_set, n_samples = n_samples)
        # generate translated irs
        gen_IRs, _ = self.generate_fake_samples(g_model, data_IRs, 0)
        
        for ii in range(len(data_IRs)):
            data_IRs[ii] = data_IRs[ii]/max(abs(data_IRs[ii]))
            gen_IRs[ii] = gen_IRs[ii]/max(abs(gen_IRs[ii]))
                    
        directory = self.working_dir + '/preview'
        subdir = directory + '/epoch_'+ str(epoch)
        
        if not os.path.exists(directory):
            os.makedirs(directory)
    
        if not os.path.exists(subdir):
            os.makedirs(subdir)
    
        fs = self.sample_rate_audio
        t = np.linspace(0, len(gen_IRs[0])/fs, len(gen_IRs[0]))
        
        fig, big_axes = plt.subplots(figsize=(20, 10) , nrows=2, ncols= 1)
        
        for row, big_ax in enumerate(big_axes, start=1):
            if row == 1:
                big_ax.set_title("A - Impulse Responses (original domain) - Epoch: {}\n".format(epoch), fontsize=20)
            else:
                big_ax.set_title("B - Impulse Responses (transfered domain) - Epoch: {} \n".format(epoch), fontsize=20)

            # Turn off axis lines and ticks of the big subplot 
            # obs alpha is 0 in RGBA string!
            big_ax.tick_params(labelcolor=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off')
            # removes the white frame
            big_ax._frameon = False
        
        for i in range(1, 2*n_samples+1):
            ax = fig.add_subplot(2,n_samples,i, sharey = True)
            # ax.set_xlabel('Time [s]', fontsize = 13)
            ax.set_title( name + ' ' + str(i))
            ax.grid(axis='both', alpha = 0.4)
            if i == 1:
                ax.tick_params(labelbottom=False)
            if 1 < i <= n_samples: 
                ax.tick_params(labelbottom=False, labelleft=False)
            elif i > n_samples + 1:
                ax.tick_params(labelleft=False)
            if i <= n_samples:
                ax.plot(t, data_IRs[i-1,:,0])
                # ax.plot(np.arange(0,100))
            elif i <= 2*n_samples:
                ax.plot(t, gen_IRs[i-n_samples-1,:,0])
                # ax.plot(np.arange(0,100))

        fig.set_facecolor('w')
        fig.text(0.5, 0.01, 'Time [s]', fontsize = 13,  ha='center') # x-axis
        fig.text(0.01, 0.5, 'Amplitude',  fontsize = 13, va='center', rotation='vertical')
        plt.tight_layout()         
        plt.savefig(subdir + '/Generated_IRs_epoch_{}_'.format(epoch) + name)
        plt.close()
                
        for ii in range(np.size(gen_IRs, axis = 0)):
            preview_fp = subdir + '/sample_' + str(ii) + '.wav'
            wavwrite(preview_fp, fs, gen_IRs[ii])
        
                
                    
                
                
                
                