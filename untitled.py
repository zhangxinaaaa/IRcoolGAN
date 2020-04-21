#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 19:32:26 2020

@author: xen
"""

import cycleGAN
import os
from data_loader import load_data


def train_cyclegan( args):
    
    ISMdir = args.data_dir + '/ISM'
    realdir = args.data_dir + '/real'
    (X_train_ISM, X_test_ISM) = load_data(ISMdir, args.data_slice_len, \
            args.data_sample_rate, args.data_num_channels, \
            args.data_normalise, save_data = True, load_array = args.loader, tag = 'ISM')
        
    (X_train_real, X_test_real) = load_data(realdir, args.data_slice_len, \
            args.data_sample_rate, args.data_num_channels, \
            args.data_normalise, save_data = True, load_array = args.loader, tag = 'real')    

    print('X_trainA shape: ', X_train_ISM.shape)
    print('X_trainB shape: ', X_train_real.shape)

    
    gan = cycleGAN.cyclegan(X_trainA = X_train_ISM
                          , X_trainB = X_train_real
                          , D_learning_rate = args.D_learning_rate
                          , batch_size = args.train_batch_size
                          , sample_rate_audio = args.data_sample_rate
                          , adversarial_learningrate = args.adv_lr
                          , IR_shape = args.data_slice_len
                          , kernel_size = args.disc_kernel_len
                          , epoch = 0
                          , total_epochs  = args.total_epochs
                          , working_dir = args.train_dir
                          , print_models = True)
    
    g_model_AtoB = gan.define_generator()
    # generator: B -> A
    g_model_BtoA = gan.define_generator()
    # discriminator: A -> [real/fake]
    d_model_A = gan.define_discriminator()
    # discriminator: B -> [real/fake]
    d_model_B = gan.define_discriminator()
    # composite: A -> B -> [real/fake, A]
    c_model_AtoB = gan.define_composite_model(g_model_AtoB, d_model_B, g_model_BtoA)
    # composite: B -> A -> [real/fake, B]
    c_model_BtoA = gan.define_composite_model(g_model_BtoA, d_model_A, g_model_AtoB)

    
    
     
    #load the latest weights (if resuming the training)
    if args.resume_train:
        weights_path = '{}/weights/'.format(args.train_dir)
        weights_batch = []
        for file in os.listdir(weights_path):
            number = file.split("_")[-1]
            number = number.split(".h5")[0]
            weights_batch.append(int(number))
        weights_batch.sort()
        try:
            last_epoch = weights_batch[-1]
            most_recent_G_AtoB = '{}g_model_AtoB_weights_{}.h5'.format(weights_path,last_epoch)
            most_recent_G_BtoA = '{}g_model_BtoA_weights_{}.h5'.format(weights_path,last_epoch)
            most_recent_D_B = '{}d_model_B_weights_{}.h5'.format(weights_path,last_epoch)
            most_recent_D_A = '{}d_model_A_weights_{}.h5'.format(weights_path,last_epoch)
            most_recent_comp_AtoB = '{}c_model_AtoB_weights_{}.h5'.format(weights_path,last_epoch)
            most_recent_comp_BtoA = '{}c_model_BtoA_weights_{}.h5'.format(weights_path,last_epoch)
            print('Loading the weights from epoch {}...'.format(last_epoch))
            g_model_AtoB.load_weights(most_recent_G_AtoB)
            g_model_BtoA.load_weights(most_recent_G_BtoA)
            d_model_A.load_weights(most_recent_D_A)
            d_model_B.load_weights(most_recent_D_B)
            # composite: A -> B -> [real/fake, A]
            c_model_AtoB.load_weights(most_recent_comp_AtoB)
            # composite: B -> A -> [real/fake, B]
            c_model_BtoA.load_weights(most_recent_comp_AtoB)
            gan.epoch = last_epoch
            print('Weights loaded.')
        except: 
            print('Could not load or find the weights of epoch {} in given directory, trying for epoch {}...'.format(weights_batch[-1], weights_batch[-2]))
            try:
                temp = weights_batch[-1]
                for ii in range(len(weights_batch)-1, -1, -1):
                    if weights_batch[ii] < temp:
                        last_epoch = weights_batch[ii]
                        break
                most_recent_G_AtoB = '{}g_model_AtoB_weights_{}.h5'.format(weights_path,last_epoch)
                most_recent_G_BtoA = '{}g_model_BtoA_weights_{}.h5'.format(weights_path,last_epoch)
                most_recent_D_B = '{}d_model_B_weights_{}.h5'.format(weights_path,last_epoch)
                most_recent_D_A = '{}d_model_A_weights_{}.h5'.format(weights_path,last_epoch)
                most_recent_comp_AtoB = '{}c_model_AtoB_weights_{}.h5'.format(weights_path,last_epoch)
                most_recent_comp_BtoA = '{}c_model_BtoA_weights_{}.h5'.format(weights_path,last_epoch)
                print('Loading the weights from epoch {}...'.format(last_epoch))
                g_model_AtoB.load_weights(most_recent_G_AtoB)
                g_model_BtoA.load_weights(most_recent_G_BtoA)
                d_model_A.load_weights(most_recent_D_A)
                d_model_B.load_weights(most_recent_D_B)
                # composite: A -> B -> [real/fake, A]
                c_model_AtoB.load_weights(most_recent_comp_AtoB)
                # composite: B -> A -> [real/fake, B]
                c_model_BtoA.load_weights(most_recent_comp_AtoB)
                gan.epoch = last_epoch
                print('Weights loaded.')
            except:
                raise Exception('Could not load epoch {} weights, please retrain. (bummer)'.format(weights_batch[-2]))   
    # print summaries
    g_model_AtoB.summary()
    d_model_B.summary()
    
    # train models
    gan.train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, save_interval = args.save_interval, sample_interval = args.sample_interval)
        




if __name__ == '__main__':
    import argparse
    # import glob
    
    parser = argparse.ArgumentParser(description="Implements a CycleGAN to transfer ISM RIR's to the domain of real RIR's in Keras")
    
    parser.add_argument('mode', type=str, choices=['train', 'generate'])
    parser.add_argument('train_dir', type=str,
        help='Training directory')
    
    data_args = parser.add_argument_group('Data')
    data_args.add_argument('--data_dir', type=str,
        help='Data directory containing *only* the folders with the audio files to load')
    data_args.add_argument('--load_IRs', action='store_true', dest='loader',
        help='Load pregenerated data from .npz files in data directory')
    data_args.add_argument('--sample_interval', type=int,
        help='Sample interval in epochs for which to generate sample IRs')
    data_args.add_argument('--save_interval', type=int,
        help='Sample interval in epochs for which to save model and weights')
    data_args.add_argument('--data_sample_rate', type=int,
        help='Number of audio samples per second')
    data_args.add_argument('--data_slice_len', type=int, choices=[16384, 32768, 65536],
        help='Number of audio samples per slice (maximum generation length)')
    data_args.add_argument('--data_num_channels', type=int,
        help='Number of audio channels to generate (for >2, must match that of data)')
    data_args.add_argument('--no_data_normalise',  action='store_false', dest='data_normalise',
        help='Do not normalise data by max value')

    
    gan_args = parser.add_argument_group('CycleGAN')
    gan_args.add_argument('--total_epochs', type=int,
        help='Number of GAN iterations in epochs')
    gan_args.add_argument('--disc_kernel_len', type=int,
        help='Length of 1D filter kernels')
    gan_args.add_argument("--D_learning_rate", help="Discriminator learning rate.", type=float)
    gan_args.add_argument("--adv_lr", help="Adversarial learning rate.", type=float)


    
    train_args = parser.add_argument_group('Train')
    train_args.add_argument('--train_batch_size', type=int,
        help='Batch size')
    train_args.add_argument('--resume_training', action='store_true', dest='resume_train',
        help = 'Resume training GAN by loading weights and model previously stored')
    
    # incept_args = parser.add_argument_group('Incept')
    # incept_args.add_argument('--incept_metagraph_fp', type=str,
    #     help='Inference model for inception score')
    # incept_args.add_argument('--incept_ckpt_fp', type=str,
    #     help='Checkpoint for inference model')
    # incept_args.add_argument('--incept_n', type=int,
    #     help='Number of generated examples to test')
    # incept_args.add_argument('--incept_k', type=int,
    #     help='Number of groups to test')
    
    parser.set_defaults(
      data_dir=None,
      loader=False,
      sample_interval=5,
      save_interval = 10,
      data_sample_rate=16000,
      data_slice_len=65536,
      data_num_channels=1,
      data_normalise = True,
      total_epochs = 200,
      disc_kernel_len=16,
      D_learning_rate = 0.0002,
      adv_lr = 0.0002,
      train_batch_size=1,
      resume_train=False)


    args = parser.parse_args()
    
    # Make train dir
    if not os.path.isdir(args.train_dir):
        os.makedirs(args.train_dir)
    
    # Save args
    with open(os.path.join(args.train_dir, 'args.txt'), 'w') as f:
      f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))


    if args.mode == 'train':
        if args.loader == False:
            # fps = glob.glob(os.path.join(args.data_dir, '*.wav'))
            # if len(fps) == 0:
            #   raise Exception('Did not find any audio files in specified directory')
            # print('Found {} audio files in specified directory'.format(len(fps)))
            train_cyclegan(args)
        else: 
            # fps = glob.glob(os.path.join(args.data_dir, '*.npz'))
            # if len(fps) == 0:
            #   raise Exception('Did not find .npz files in specified directory')
            # print('Found {} .npz files in specified directory'.format(len(fps)))
            train_cyclegan(args)
        
    # elif args.mode == 'preview':
    #   preview(args)
    # elif args.mode == 'incept':
    #   incept(args)
    # elif args.mode == 'infer':
    #   infer(args)
    else:
      raise NotImplementedError()
    
    