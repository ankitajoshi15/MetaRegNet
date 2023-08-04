"""
For some parts of the code below:
Thanks to voxelmorph: Learning-Based Image Registration, https://github.com/voxelmorph/voxelmorph for this code.
If you use this code, please cite the respective papers in their repo.
"""

import keras.layers as KL
from keras.layers import Multiply, Add, Input, concatenate, Activation, Conv2D, LeakyReLU, Lambda, Layer,ReLU, Conv2DTranspose
from keras.models import Model
from keras.initializers import RandomNormal
import keras.initializers
import tensorflow as tf
import stn
from SpectralNormalizationKeras import ConvSN2D
import utils

def unet_core(vol_size, enc_nf, dec_nf, src=None, tgt=None, src_feats=1, tgt_feats=1):
    """
    unet architecture for voxelmorph models presented in the CVPR 2018 paper.
    You may need to modify this code (e.g., number of layers) to suit your project needs.
    :param vol_size: volume size. e.g. (256, 256, 256)
    :param enc_nf: list of encoder filters. right now it needs to be 1x4.
           e.g. [16,32,32,32]
    :param dec_nf: list of decoder filters. right now it must be 1x6 (like voxelmorph-1) or 1x7 (voxelmorph-2)
    :return: the keras model
    """
    ndims = len(vol_size)
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims
    #upsample_layer = getattr(KL, 'UpSampling%dD' % ndims)
    # inputs
    if src is None:
        src = Input(shape=[*vol_size, src_feats])
    if tgt is None:
        tgt = Input(shape=[*vol_size, tgt_feats])

    x_in = concatenate([src, tgt])

    # down-sample path (encoder)
    x_enc = [x_in]

    for i in range(len(enc_nf)):
        x_enc.append(conv_block(x_enc[-1], enc_nf[i], 2))

    # up-sample path (decoder)
    x = conv_block(x_enc[-1], dec_nf[0])
    x = Conv2DTranspose(256, kernel_size=3, strides=2, padding='same')(x)#upsample_layer()(x)
    x = concatenate([x, x_enc[-2]])
    x = conv_block(x, dec_nf[1])
    x = Conv2DTranspose(128, kernel_size=3, strides=2, padding='same')(x)#upsample_layer()(x)
    x = concatenate([x, x_enc[-3]])
    x = conv_block(x, dec_nf[2])
    x = Conv2DTranspose(64, kernel_size=3, strides=2, padding='same')(x)#upsample_layer()(x)
    x = concatenate([x, x_enc[-4]])
    x = conv_block(x, dec_nf[3])
    x = conv_block(x, dec_nf[4])
    x = Conv2DTranspose(32, kernel_size=3, strides=2, padding='same')(x)#upsample_layer()(x)
    x = concatenate([x, x_enc[0]])
    x = conv_block(x, dec_nf[5])
    # optional convolution at output resolution (used in voxelmorph-2)
    if len(dec_nf) == 7:
        x = conv_block(x, dec_nf[6])
    
    return Model(inputs=[src, tgt], outputs=[x])


def unet(vol_size, enc_nf, dec_nf, mask=None):
 
    if mask is None:
        mask = Input(shape=[*vol_size, 1])

    ndims = len(vol_size)
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims

    # get unet
    unet_model = unet_core(vol_size, enc_nf, dec_nf)
    [src, tgt] = unet_model.inputs
    x_out = unet_model.outputs[-1]
    
    # velocity sample
    flow0 = Conv2D(2, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal', use_bias=False)(x_out)
    flow0 = Lambda(lambda x: x / 2.0)(flow0)

    # forward integration
    out1 = ConvSN2D(2, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal', use_bias=False)(
        flow0)
    out1 = LeakyReLU(0.2)(out1)
    out1 = ConvSN2D(2, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal', use_bias=False)(
        out1)
    out1 = Activation(tf.keras.activations.tanh)(out1)
    out1 = Lambda(lambda x: x / 2.0)(out1)
    v1 = Add()([flow0, out1])
    
    out2 = ConvSN2D(2, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal', use_bias=False)(
        v1)
    out2 = LeakyReLU(0.2)(out2)
    out2 = ConvSN2D(2, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal', use_bias=False)(
        out2)
    out2 = Activation(tf.keras.activations.tanh)(out2)
    out2 = Lambda(lambda x: x / 2.0)(out2)
    # out2 = resmodel(v1)
    v2 = Add()([v1, out2])
    
    out3 = ConvSN2D(2, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal', use_bias=False)(
        v2)
    out3 = LeakyReLU(0.2)(out3)
    out3 = ConvSN2D(2, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal', use_bias=False)(
        out3)
    out3 = Activation(tf.keras.activations.tanh)(out3)
    out3 = Lambda(lambda x: x / 2.0)(out3)
    # out3 = resmodel(v2)
    v3 = Add()([v2, out3])

    out4 = ConvSN2D(2, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal', use_bias=False)(
        v3)
    out4 = LeakyReLU(0.2)(out4)
    out4 = ConvSN2D(2, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal', use_bias=False)(
        out4)
    out4 = Activation(tf.keras.activations.tanh)(out4)
    out4 = Lambda(lambda x: x / 2.0)(out4)
    # out4 = resmodel(v3)
    v4 = Add()([v3, out4])

    ###########################
    #INTENSITY PART
    ######################################## 
    q0 = Conv2D(1, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal', use_bias=False)(x_out)
    q0 = Lambda(lambda x: x / 2.0)(q0)
    #q0 = Multiply()([q0,mask])

    # forward integration
    out1 = ConvSN2D(1, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal', use_bias=False)(q0)
    out1 = LeakyReLU(0.2)(out1)
    out1 = ConvSN2D(1, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal', use_bias=False)(out1)
    out1 = Activation(tf.keras.activations.tanh)(out1)
    out1 = Lambda(lambda x: x / 2.0)(out1)
    q1 = Add()([q0, out1])
    #q1 = Multiply()([q1,mask])

    out2 = ConvSN2D(1, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal', use_bias=False)(q1)
    out2 = LeakyReLU(0.2)(out2) 
    out2 = ConvSN2D(1, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal', use_bias=False)(out2)
    out2 = Activation(tf.keras.activations.tanh)(out2)
    out2 = Lambda(lambda x: x / 2.0)(out2)
    q2 = Add()([q1, out2])
    #q2 = Multiply()([q2,mask])   
 
    out3 = ConvSN2D(1, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal', use_bias=False)(q2)
    out3 = LeakyReLU(0.2)(out3) 
    out3 = ConvSN2D(1, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal', use_bias=False)(out3)
    out3 = Activation(tf.keras.activations.tanh)(out3)
    out3 = Lambda(lambda x: x / 2.0)(out3)
    q3 = Add()([q2, out3])
    #q3 = Multiply()([q3,mask])

    out4 = ConvSN2D(1, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal', use_bias=False)(q3)
    out4 = LeakyReLU(0.2)(out4)
    out4 = ConvSN2D(1, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal', use_bias=False)(out4)
    out4 = Activation(tf.keras.activations.tanh)(out4)
    out4 = Lambda(lambda x: x / 2.0)(out4)
    q4 = Add()([q3, out4]) 
    q = Activation(tf.keras.activations.tanh)(q4)
    q = Multiply()([q,mask])
    #q =  Lambda(lambda x: x * 0.025*0.025)(q) #Multiply()([0.000625, q])

    """
    out5 = ConvSN2D(2, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal', use_bias=False)(
        v4)
    out5 = LeakyReLU(0.2)(out5)
    out5 = ConvSN2D(2, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal', use_bias=False)(
        out5)
    out5 = Activation(tf.keras.activations.tanh)(out5)
    out5 = Lambda(lambda x: x / 2.0)(out5)
    # out5 = resmodel(v4)
    v5 = Add()([v4, out5])

    out6 = ConvSN2D(2, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal', use_bias=False)(v5)
    out6 = LeakyReLU(0.2)(out6)
    out6 = ConvSN2D(2, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal', use_bias=False, )(out6)
    out6 = Activation(tf.keras.activations.tanh)(out6)
    out6 = Lambda(lambda x: x / 2.0)(out6)
    # out6 = resmodel(v5)
    v6 = Add()([v5, out6])

    out7 = ConvSN2D(2, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal', use_bias=False)(v6)
    out7 = LeakyReLU(0.2)(out7)
    out7 = ConvSN2D(2, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal', use_bias=False)(out7)
    out7 = Activation(tf.keras.activations.tanh)(out7)
    out7 = Lambda(lambda x: x / 2.0)(out7)
    v7 = Add()([v6, out7])
    """
    phi = Addgrid()(v4)

    # warp the source with the flow
    #updated_src = Add()([src,q])
    deformed_src = stn.SpatialTransformer(interp_method='linear', indexing='ij')([src, phi])
    final_op = Add()([deformed_src,q]) 
    

    return Model(inputs=[src, tgt, mask], outputs=[final_op, q, phi])


def sample(args):
    """
    sample from a normal distribution
    """
    mu = args[0]
    log_sigma = args[1]
    noise = tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)
    z = mu + tf.exp(log_sigma/2.0) * noise
    return z

class Sample(Layer):
    """
    Keras Layer: Gaussian sample from [mu, sigma]
    """

    def __init__(self, **kwargs):
        super(Sample, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Sample, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return sample(x)

    def compute_output_shape(self, input_shape):
        return input_shape[0]


def conv_block(x_in, nf, strides=1):
    """
    specific convolution module including convolution followed by leakyrelu
    """
    ndims = len(x_in.get_shape()) - 2
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims

    Conv = getattr(KL, 'Conv%dD' % ndims)
    x_out = Conv(nf, kernel_size=3, padding='same',
                 kernel_initializer='he_normal', strides=strides)(x_in)
    x_out = LeakyReLU(0.2)(x_out)
    return x_out

class Addgrid(Layer):
    """
   #     Keras Layer: Add grid to velocity field for locations.
   """
    def __init__(self, **kwargs):
        super(Addgrid, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Addgrid, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        volshape = [224,160]
        mesh = utils.volshape_to_meshgrid(volshape)  # volume mesh
        loc = [tf.cast(mesh[d], 'float32') + x[..., d] for d in range(2)]
        loc = tf.stack(loc, -1)
        return loc

    def compute_output_shape(self, input_shape):
        return input_shape


class MM(Layer):
    """
   #     Keras Layer: Add grid to velocity field for locations.
   """
    def __init__(self, **kwargs):
        super(MM, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MM, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inp):
        m = inp[0]
        q = inp[1]
        return m*q

    def compute_output_shape(self, input_shape):
        return input_shape

