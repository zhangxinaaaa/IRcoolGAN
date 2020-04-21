#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 11:43:26 2020

@author: xen
"""
import scipy.signal as sps
#from scipy.io.wavfile import read as wavread
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from progressbar import ProgressBar
import soundfile as sf



def resample_audio(audio, fs_old, fs_new):
    # Resample data
    if fs_old > fs_new:
        data = sps.decimate(audio, int(fs_old / fs_new))
    elif fs_old == fs_new:
        return audio
    else:
        ValueError("Cannot upsample, please only use files with a higher or equal sampling frequency to fs = {}".format(fs_new))
    return data

def decode_audio(fp, fs=None, num_channels=1, normalize=False):
    """Decodes audio file paths into 32-bit floating point vectors.
    
    Args:
      fp: Audio file path.
      fs: If specified, resamples decoded audio to this rate.
      mono: If true, averages channels to mono.
      fast_wav: Assume fp is a standard WAV file (PCM 16-bit or float 32-bit).
      normalize: normalize audio by maximum value
      slice_len: length of audio to use as data (and consequently generate)
    Returns:
      A np.float32 array containing the audio samples at specified sample rate.
    """
    #Read with scipy wavread (fast).
    _wav, _fs = sf.read(fp, dtype = 'float32')
    _wav = _wav/np.max(np.abs(_wav))
    if _wav.ndim > 1:
        _wav = _wav[:,0]
    # if fs is not 16.000 kHz then resample
    if fs is not None and fs != _fs:
      _wav = resample_audio(_wav, _fs, fs)
    if _wav.dtype == np.int16:
      _wav = _wav.astype(np.float32, order = 'C')
      _wav /= 32768.
    elif _wav.dtype == np.float64:
      _wav = np.copy(_wav).astype(np.float32)
    elif _wav.dtype == np.float32:
      _wav = np.copy(_wav)
    else:
      raise NotImplementedError('Scipy cannot process atypical WAV files.')
  
    assert _wav.dtype == np.float32
    
    # At this point, _wav is np.float32 either [nsamps,] or [nsamps, nch].
    # We want [nsamps, 1, nch] to mimic 2D shape of spectral feats.
    if _wav.ndim == 1:
      nsamps = _wav.shape[0]
      nch = 1
    else:
      nsamps, nch = _wav.shape
      
    # Average (mono) or expand (stereo) channels
    if nch != num_channels:
      if num_channels == 1:
        _wav = np.mean(_wav, 1, keepdims=True)
      elif nch == 1 and num_channels == 2:
        _wav = np.concatenate([_wav, _wav], axis=1)
      else:
        raise ValueError('Number of audio channels not equal to num specified')
    
    if normalize:
      factor = np.max(np.abs(_wav))
      if factor > 0:
        _wav /= factor
    
    return _wav

                
def load_data(direc,
    slice_len,
    IR_fs,
    IR_num_channels = 1,
    IR_normalize=True,
    train_test_ratio = 0.2,
    save_data = True,
    load_array = False,
    tag = ''):
    """Splits and loads data consisting of Imp. Responses into a train and a test set.
    Args:
      slice_len: length of audio to use as data (and consequently generate)
      direc: Directory of .wav files (Impulse Responses).
      IR_fs: If specified, resamples decoded audio to this rate.
      IR_num_channels: Number of channels of audio.
      fast_wav: Assume fp is a standard WAV file (PCM 16-bit or float 32-bit).
      IR_normalize: normalize audio by maximum value
   
    Returns:
      Tuples of the form: X_train, X_test consisting of numpy arrays which 
      contain Room Impulse Responses.
      Xtot = [X_train, X_test] = [ (1- train_test_ratio) x Xtot, train_test_ratio x Xtot]
      X_train/X_test.shape = [n_imp.response, n_samples, n_channels]  
    """
    
    if type(direc) == list:
        direc = direc[0]
    
    if load_array:
        try:
            # Image Source Method or real RIR data of size [n_imp.response, n_samples, n_channels] 
            # train
            x_load = glob.glob(direc + '/*{}_{}_train.npz'.format(slice_len, tag))[0]
            xyt = np.load(x_load)
            X_train = xyt['arr_0']
            # test
            x_load = glob.glob(direc + '/*{}_{}_test.npz'.format(slice_len, tag))[0]
            xyt = np.load(x_load)
            X_test = xyt['arr_0']                        
            print('-' * 80)
            print('Successfully loaded processed data!')
            print('-' * 80)
            return (X_train, X_test)
        except: 
            raise Exception('Cannot find files in given directory, please try again or regenerate the data')

    files = glob.glob(direc + '/*.wav')
    
    X = []
    
    print('-' * 80)
    print('Processing data please wait...')
    print('-' * 80)

    pbar = ProgressBar()


    for f in pbar(files):
        
        IRwav = decode_audio(f, IR_fs, IR_num_channels, IR_normalize)
        IRwav = np.atleast_2d(IRwav).T
        Audio = np.zeros((slice_len, IR_num_channels))
        if np.max(IRwav) < 0.1:
            continue
        # early truncation
        idx = sps.find_peaks(abs(IRwav[:,0]), height=0.1)[0]
        idx = np.min(idx)
        # print(idx)
        # print('n channels ', IR_num_channels )
        # print('idx: ', idx)
        # padding left right (begining of IR - end of IR)
        for jj in range(IR_num_channels):
            chan = IRwav[idx:,jj]
            if chan.shape[0] < slice_len:
                zero_pad = slice_len - chan.shape[0]
                chan = np.hstack( (chan , np.zeros((zero_pad,))))
            else:
                chan = np.hstack((chan[:slice_len-40], np.zeros((40,))))
            Audio[:,jj] = chan   
            
        X.append(Audio)
            
        # print(np.array(X).shape)
        
        
    print('-' * 80)
    print('Successfully processed data!')
    print('Total data of size X_tot: ', np.array(X).shape)
    print('Splitting data into train and test sets...')
    
    X_train, X_test = train_test_split( X, test_size= train_test_ratio, \
                                                        random_state= 58)


    if save_data:
        print('Saving data...')
        import datetime
        x = datetime.datetime.now()
        datetime = x.strftime('%d')+'-'+x.strftime('%m')
        np.savez_compressed(direc + '/X_{}_len_{}_{}_train'.format(datetime, slice_len, tag), X_train)
        np.savez_compressed(direc + '/X_{}_len_{}_{}_test'.format(datetime, slice_len, tag), X_test)
        
    print('Done!')
    print('-' * 80)
    
    
                
    return (np.asarray(X_train), np.asarray(X_test))
