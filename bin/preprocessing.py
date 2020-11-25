import numpy as np
import tensorflow as tf
#from pyts.image import MTF, RecurrencePlots
## Running it in eager mode
##tf.config.experimental_run_functions_eagerly(True)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def rescale(x : np.ndarray) -> np.ndarray:
    """Rescale a NumPy array so its values are in the range [0,1]"""
    amin = tf.math.reduce_min(x)
    amax = tf.math.reduce_max(x)
    return (x - amin) / (amax - amin)

def tf_outer(x : np.ndarray, y : np.ndarray) -> np.ndarray:
    """Compute outer product using TensorFlow.

    Args:
        x: A vector
        y: A vector

    Returns:
        The outer product of x and y
    """
    return tf.tensordot(x, y, axes=0)


## mtf function is a python funciton unlike the rest here

def mtf(x):
    """
    Compute Markov Transition Field

    Args:
        x: A vector of size N
    
    Returns:
        Markov Transition Field of shape (N, N).
    """
    #x = my_func(x)
    #x_proto = tf.make_tensor_proto(x)
    #x = tf.make_ndarray(x)
    encoder1 = MTF(128, n_bins=128//15, quantiles='gaussian') 
    a = encoder1.fit_transform(x)
    r = (a.reshape(-1, 1)).reshape((2, 128, 128))[0, :]
    #r = np.asarray(r)
    return r

def rplot(x):
    """
    Compute the recurrence plot 
    Args:
	x: A vector of size N
    Returns:
       Recurrence Plot of shape (N, N).
    """
    encoder1 = RecurrencePlots()
    a = encoder1.fit_transform(x)
    r = (a.reshape(-1, 1)).reshape((2, 128, 128))[0, :]
    return r

@tf.function(input_signature=[tf.TensorSpec(None, np.double)])
def nmp_rplot(x):
    y = tf.numpy_function(rplot, [x], np.double)
    return y

## Wrap the mtf function

@tf.function(input_signature=[tf.TensorSpec(None, np.double)])
def nmp_mtf(x):
    y = tf.numpy_function(mtf, [x], np.double)
    return y


from numpy import sum,isrealobj,sqrt
from numpy.random import standard_normal

def noise(s):
    """
    AWGN channel
    Add AWGN noise to input signal. The function adds AWGN noise vector to signal 's' to generate a resulting signal vector 'r' of specified SNR in dB. It also
    returns the noise vector 'n' that is added to the signal 's' and the power spectral density N0 of noise added
    Parameters:
        s : input/transmitted signal vector
        SNRdB : desired signal to noise ratio (expressed in dB) for the received signal
        L : oversampling factor (applicable for waveform simulation) default L = 1.
    Returns:
        r : received signal vector (r=s+n)
    """
    SNRdB=5
    L=1
    gamma = 10**(SNRdB/10) #SNR to linear scale
    P=L*np.sum(np.sum(abs(s)**2))/len(s) # if s is a matrix [MxN]
    N0=P/gamma # Find the noise spectral density
    # if isrealobj(s):# check if input is real/complex object type
    #     n = sqrt(N0/2)*standard_normal(s.shape) # computed noise
    # # else:
    # #     n = sqrt(N0/2)*(standard_normal(s.shape)+1j*standard_normal(s.shape))
    # r = s + n # received signal
    return s

@tf.function(input_signature=[tf.TensorSpec(None, np.double)])
def noisy(x):
    y = tf.numpy_function(noise, [x], np.double)
    return y

def gasf(x):
    """Compute Gramian angular summation field

    Args:
        x: A vector of size N

    Returns:
        Gramian angular summation field of shape (N, N).
    """
    y = tf.sqrt(1 - x ** 2)

    return tf_outer(x, x) - tf_outer(y, y)

def gadf(x):
    """Compute Gramian angular difference field

    Args:
        x: A vector of size N

    Returns:
        Gramian angular difference field of shape (N, N).
    """
    y = tf.sqrt(1 - x ** 2)

    return tf_outer(y, x) - tf_outer(x, y)

def preprocess_outer(iq : np.ndarray) -> np.ndarray:
    """Take IQ data of shape (2, N) and compute outer product.

    Args:
        iq: IQ data of shape (2,N)

    Returns:
        Array of shape (3,N) with three channels: the outer product of I and
        I, the outer product of Q and Q, and the outer product of I and Q.
        All channels are jointly scaled to be in the range [0,1].
    """
    result = tf.stack([ tf_outer(iq[0], iq[0])
                      , tf_outer(iq[1], iq[1])
                      , tf_outer(iq[0], iq[1])],
                      axis=2)

    return rescale(result)

def preprocess_gasf(iq : np.ndarray) -> np.ndarray:
    """Take IQ data of shape (2, N) and GASF on i channel.

    Args:
        iq: IQ data of shape (2,N)

    Returns:
        Array of shape (3,N) with three channels: the GASF of I, the outer
        product of Q and Q, and the outer product of I and Q. All channels are
        jointly scaled to be in the range [0,1].
    """
    # Computer full outer product and jointly scale it
    result = tf.stack([ tf_outer(iq[0], iq[0])
                      , tf_outer(iq[1], iq[1])
                      , tf_outer(iq[0], iq[1])],
                      axis=2)

    result = rescale(result)

    # Replace outer product of I and I with GASF
    return tf.stack([ rescale(gasf(iq[0]))
                    , result[:,:,1]
                    , result[:,:,2]],
                    axis=2)

def preprocess_gadf(iq : np.ndarray) -> np.ndarray:
    """Take IQ data of shape (2, N) and GADF on i channel.

    Args:
        iq: IQ data of shape (2,N)

    Returns:
        Array of shape (3,N) with three channels: the GADF of I, the outer
        product of Q and Q, and the outer product of I and Q. All channels are
        jointly scaled to be in the range [0,1].
    """
    # Computer full outer product and jointly scale it
    result = tf.stack([ tf_outer(iq[0], iq[0])
                      , tf_outer(iq[1], iq[1])
                      , tf_outer(iq[0], iq[1])],
                      axis=2)

    result = rescale(result)

    # Replace outer product of I and I with GASF
    return tf.stack([rescale(gadf(iq[0]))
                    , result[:,:,1]
                    , result[:,:,2]],
                    axis=2)

def preprocess_mtf(iq : np.ndarray) ->np.ndarray:
    

    result = tf.stack([ tf_outer(iq[0], iq[0])
                      , tf_outer(iq[1], iq[1])
                      , tf_outer(iq[0], iq[1])],
                      axis=2)
    result = rescale(result)

    # Replace outer product of I and I with MTF
    return tf.stack([tf.cast(nmp_mtf(iq), dtype=tf.float32)
                    , result[:,:,1]
                    , result[:,:,2]],
                    axis=2)

def preprocess_rplot(iq : np.ndarray) ->np.ndarray:
    

    result = tf.stack([tf_outer(iq[0], iq[0])
                      , tf_outer(iq[1], iq[1])
                      , tf_outer(iq[0], iq[1])],
                      axis=2)
    result = rescale(result)

    # Replace outer product of I and I with Recurrence Plot
    return tf.stack([tf.cast(nmp_rplot(iq), dtype=tf.float32)
                    , result[:,:,1]
                    , result[:,:,2]],
                    axis=2)
def preprocess_noisy_outer(iq):
    # iq = noisy(iq)
    SNRdB= np.random.normal(scale = 100, loc = 50)
    L=1
    gamma = 10**(SNRdB/10) #SNR to linear scale
    P=L*tf.math.reduce_sum(tf.math.reduce_sum(abs(iq)**2))/len(iq) # if s is a matrix [MxN]
    N0=P/gamma 
    n = tf.math.sqrt(N0/2)*(tf.random.normal(iq.shape))
    iq= tf.cast(iq + n, tf.float32)
    result = rescale(tf.stack([tf_outer(iq[0], iq[0]), tf_outer(iq[1], iq[1]), tf_outer(iq[0], iq[1])], axis=2))
    return result