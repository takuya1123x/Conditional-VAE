import warnings
import numpy as np
from keras.layers import Input, Dense, Lambda
from keras.layers.merge import concatenate as concat
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
from keras.utils import to_categorical, plot_model, np_utils
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from scipy.misc import imsave
import matplotlib.pyplot as plt
from keras.losses import mse, binary_crossentropy
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import re

warnings.filterwarnings('ignore')
#pylab inline


############################
def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm'):
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if re.match(r'([\w]+\.(?:' + ext + '))', f.lower())]
#######################

x = []
y = []

for picture in list_pictures('./snap/1'):
    img = img_to_array(load_img(picture, target_size=(120,120,3)))
    x.append(img)
    y.append(0)

for picture in list_pictures('./snap/2'):
    img = img_to_array(load_img(picture, target_size=(120,120,3)))
    x.append(img)
    y.append(1)

for picture in list_pictures('./snap/3'):
    img = img_to_array(load_img(picture, target_size=(120,120,3)))
    x.append(img)
    y.append(2)


x = np.asarray(x)
y = np.asarray(y)

#x = x.astype('float32')
#x = x/ 255.0
#print(x)
#x = x.reshape(6015, 43200)
#print(x)
#y = np_utils.to_categorical(y, 15)
#print(y)

X_train, X_test, Y_train, Y_test = train_test_split(x, y, random_state=99)
#original_dim = 43200

###########################

#(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

n_pixels = np.prod(X_train.shape[1:])
#print(n_pixels)
X_train = X_train.reshape((len(X_train), n_pixels))
X_test = X_test.reshape((len(X_test), n_pixels))
#print(X_train)
#print(X_test)
#1,0の行列
#print(Y_train)
#print(Y_test)
#categorical number

y_train = to_categorical(Y_train)
y_test = to_categorical(Y_test)
#print(y_train)
#print(y_test)
#1,0の行列


#m = 250 # batch size
m = 30 # batch size
n_z = 2 # latent space size
encoder_dim1 = 512 # dim of encoder hidden layer
decoder_dim = 512 # dim of decoder hidden layer
decoder_out_dim = n_pixels # dim of decoder output layer
activ = 'relu'
optim = Adam(lr=0.001)


n_x = X_train.shape[1]
n_y = y_train.shape[1]


n_epoch = 100
#n_epoch = 1

X = Input(shape=(n_x,))
label = Input(shape=(n_y,))

inputs = concat([X, label])

encoder_h = Dense(encoder_dim1, activation=activ)(inputs)
mu = Dense(n_z, activation='linear')(encoder_h)
l_sigma = Dense(n_z, activation='linear')(encoder_h)

def sample_z(args):
    mu, l_sigma = args
    eps = K.random_normal(shape=(m, n_z), mean=0., stddev=1.)
    return mu + K.exp(l_sigma / 2) * eps


# Sampling latent space
z = Lambda(sample_z, output_shape = (n_z, ))([mu, l_sigma])

# merge latent space with label
zc = concat([z, label])

decoder_hidden = Dense(decoder_dim, activation=activ)
decoder_out = Dense(decoder_out_dim, activation='sigmoid')
h_p = decoder_hidden(zc)
outputs = decoder_out(h_p)

def vae_loss(y_true, y_pred):
    recon = K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)
    kl = 0.5 * K.sum(K.exp(l_sigma) + K.square(mu) - 1. - l_sigma, axis=-1)
    return recon + kl

def KL_loss(y_true, y_pred):
	return(0.5 * K.sum(K.exp(l_sigma) + K.square(mu) - 1. - l_sigma, axis=1))

def recon_loss(y_true, y_pred):
	return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)

cvae = Model([X, label], outputs)
encoder = Model([X, label], mu)

d_in = Input(shape=(n_z+n_y,))
d_h = decoder_hidden(d_in)
d_out = decoder_out(d_h)
decoder = Model(d_in, d_out)

cvae.compile(optimizer=optim, loss=vae_loss, metrics = [KL_loss, recon_loss])

# compile and fit
cvae_hist = cvae.fit([X_train, y_train], X_train, verbose = 1, batch_size=m, epochs=n_epoch,
							validation_data = ([X_test, y_test], X_test),
							callbacks = [EarlyStopping(patience = 5)])


plt.imshow(X_train[0].reshape(120, 120, 3), cmap = plt.cm.gray), plt.axis('off')
plt.show()
#print(Y_train[0])


#encoded_X0 = encoder.predict([X_train[0].reshape((1, n_pixels)), y_train[0].reshape((1, 5))])
encoded_X0 = encoder.predict([X_train[0].reshape((1, n_pixels)), y_train[0].reshape((1, 15))])
z_train = encoder.predict([X_train, y_train])
encodings= np.asarray(z_train)
encodings = encodings.reshape(X_train.shape[0], n_z)
#plt.figure(figsize=(7, 7))
plt.figure(figsize=(15, 15))
plt.scatter(encodings[:, 0], encodings[:, 1], c=Y_train, cmap=plt.cm.jet)
plt.colorbar()
plt.savefig('result.png')
#plt.show()

np.set_printoptions(threshold=np.inf)
np.save('test.npy', encodings)
ndarr = np.load('test.npy')
f = open('encodings.dat', 'w', encoding='utf-8')
f.writelines("\n")
f.writelines(str(ndarr))
f.writelines("\n")





def construct_numvec(digit, z = None):
    out = np.zeros((1, n_z + n_y))
    out[:, digit + n_z] = 1.
    if z is None:
        return(out)
    else:
        for i in range(len(z)):
            out[:,i] = z[i]
        return(out)

sample_3 = construct_numvec(3)
#print(sample_3)

#plt.figure(figsize=(3, 3))
#plt.imshow(decoder.predict(sample_3).reshape(28,28), cmap = plt.cm.gray), axis('off')
#plt.show()
dig = 0
sides = 8
#sides = 4
max_z = 1.5

img_it = 0
for i in range(0, sides):
    z1 = (((i / (sides-1)) * max_z)*2) - max_z
    for j in range(0, sides):
        z2 = (((j / (sides-1)) * max_z)*2) - max_z
        z_ = [z1, z2]
        vec = construct_numvec(dig, z_)
        decoded = decoder.predict(vec)
        plt.subplot(sides, sides, 1 + img_it)
        img_it +=1
        plt.imshow(decoded.reshape(120,120,3), cmap = plt.cm.gray), plt.axis('off')
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=.2)
plt.savefig('result0.png')



dig = 1
sides = 8
#sides = 4
max_z = 1.5

img_it = 0
for i in range(0, sides):
    z1 = (((i / (sides-1)) * max_z)*2) - max_z
    for j in range(0, sides):
        z2 = (((j / (sides-1)) * max_z)*2) - max_z
        z_ = [z1, z2]
        vec = construct_numvec(dig, z_)
        decoded = decoder.predict(vec)
        plt.subplot(sides, sides, 1 + img_it)
        img_it +=1
        plt.imshow(decoded.reshape(120,120,3), cmap = plt.cm.gray), plt.axis('off')
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=.2)
#plt.show()
plt.savefig('result1.png')

dig = 2
sides = 8
#sides = 4
max_z = 1.5

img_it = 0
for i in range(0, sides):
    z1 = (((i / (sides-1)) * max_z)*2) - max_z
    for j in range(0, sides):
        z2 = (((j / (sides-1)) * max_z)*2) - max_z
        z_ = [z1, z2]
        vec = construct_numvec(dig, z_)
        decoded = decoder.predict(vec)
        plt.subplot(sides, sides, 1 + img_it)
        img_it +=1
        plt.imshow(decoded.reshape(120,120,3), cmap = plt.cm.gray), plt.axis('off')
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=.2)
plt.savefig('result2.png')
