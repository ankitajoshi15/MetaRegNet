import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import network
import losses
import numpy as np

from keras.utils import multi_gpu_model

#prepare model folder
model_dir = '../models/'
output_dir = '../outputs/'
data_dir = '/home/ankita/metamorphosis/data/data_metamorphosis_withseg/'

# gpu handling
gpu = '/gpu:%d' % 0 # gpu_id
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
set_session(tf.Session(config=config))


prior_lambda=10
image_sigma=1.0

initial_epoch = 0
steps_per_epoch = 100
gpu_id = "0"
batch_size = 1

lr = 1e-3
vol_size = [224,160]
nf_enc = [16,32,64,128]
nf_dec = [32,32,32,32,16,2]

with tf.device(gpu):
    model = network.unet(vol_size,nf_enc, nf_dec)
    flow_vol_shape = [224,160]
    loss_class = losses.IR(image_sigma, prior_lambda,flow_vol_shape) #losses.NCC()#losses.IR(0.04, 10, flow_vol_shape) 
    loss_class_grad = losses.Grad(penalty='l2')
    loss_class_foldings = losses.Foldings()
    model_losses = [loss_class.recon_loss, loss_class_grad.loss, loss_class_foldings.loss]
    loss_weights = [1.0, 1.0, 0.0001]

#Load downsampled data
tgt = np.load(data_dir+'unhealthy_trainSource.npy')
src = np.load(data_dir+'healthy_trainTarget.npy')
z = np.zeros((200,224,160,2))
mask = np.load(data_dir+'trainmask.npy')

ttgt = np.load(data_dir+"unhealthy_testSource.npy")
tsrc = np.load(data_dir+"healthy_testTarget.npy")
tmask = np.load(data_dir+"testmask.npy") 

with tf.device(gpu):
    model.compile(optimizer=Adam(lr=lr), loss=model_losses, loss_weights=loss_weights)
    mg_model = multi_gpu_model(model,gpus=4)
    mg_model.compile(optimizer=Adam(lr=lr), loss=model_losses, loss_weights=loss_weights)
    model.load_weights('../models/model2000.h5')
    for i in range(0,20):
        mg_model.evaluate([np.expand_dims(tsrc[i,:,:,:],axis=0), np.expand_dims(ttgt[i,:,:,:],axis=0),  np.expand_dims(tmask[i,:,:,:],axis=0)],[np.expand_dims(ttgt[i,:,:,:],axis=0),np.expand_dims(z[i,:,:,:],axis=0), np.expand_dims(z[i,:,:,:],axis=0)], batch_size=1)

     
    #mg_model.fit([src, tgt, mask], [tgt, z, z],
    #             epochs=2000,
    #             batch_size=32,
    #             verbose=1)
    #model.save('../models/model2000.h5')
    #predictions = model.predict([tsrc, ttgt, tmask], batch_size=1)
    #np.save(output_dir + 'target2000', np.asarray(predictions[0]))
    #np.save(output_dir + 'mask2000', np.asarray(predictions[1]))
    #np.save(output_dir + 'phi2000', np.asarray(predictions[2]))

    #predictions = model.predict([src, tgt, mask], batch_size=1)
    #np.save(output_dir + 'traintarget2000', np.asarray(predictions[0]))
    #np.save(output_dir + 'trainmask2000', np.asarray(predictions[1]))
    #np.save(output_dir + 'trainphi2000', np.asarray(predictions[2]))






