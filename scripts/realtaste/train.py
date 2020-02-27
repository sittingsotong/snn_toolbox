

import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scripts.realtaste.utils import LastRowLayer

pd.set_option('display.max_columns', 500)

import warnings
warnings.filterwarnings('ignore')

from keras import backend
from keras import models
from keras import layers
from keras import regularizers
from keras import callbacks


from sklearn import metrics

backend.tensorflow_backend._get_available_gpus()
print(backend.tensorflow_backend._get_available_gpus())

to20s = 4
to3min = 36
to5min = 60
to20min = 240
toH = 720
toH2 = 900

out_meas = ['LABEL']
in_meas = ['ISF1', 'ISF2', 'ISF3', 'ISF4', 'ISF5', 'ISF6', 'ORP', 'COND', 'TEMP'] #, u'Conductivitat (20 °C)', 'isf1_c', 'isf2_c', 'isf3_c', 'isf4_c', 'isf5_c', 'isf6_c', 'orp', 'temp']

df_all_norm = pd.read_pickle('df_realtaste_lab_filt_std_py3_allmeas.pkl')
df_all_norm['LABEL'] = df_all_norm['LABEL'] - 1 # Labels start at 0 instead of 1

n_test = 2281
n_disc = 787


def createTrainTest(x, y, n_test, n_disc):
	# global n_test
	# global n_disc

	train_x	= x[:-n_test, :]
	test_x	= x[-n_test:-n_disc, :]
	train_y	= y[:-n_test]
	test_y	= y[-n_test:-n_disc]
	return train_x, test_x, train_y, test_y


n_vars = len(in_meas)
n_input = 10

df_all_5s = df_all_norm[out_meas + in_meas].copy()
y_ph_2_5s = df_all_5s[out_meas].dropna()
xl2_ph_5s = []

for i in reversed(range(n_input)):
	xl2_ph_5s.append(pd.DataFrame(df_all_norm.reindex(y_ph_2_5s.index - datetime.timedelta(seconds=i))[in_meas].values, columns=[m + '-{}sec'.format(i) for m in in_meas]))  # u'Conductivitat (20 °C)-'+str(i*5)+'sec', 'isf1_c-'+str(i*5)+'sec', 'temp-'+str(i*5)+'sec'])) #[u'Conductivitat (20 °C)-'+str(i*5)+'sec', 'isf1_c-'+str(i*5)+'sec', 'isf2_c-'+str(i*5)+'sec', 'isf3_c-'+str(i*5)+'sec', 'isf4_c-'+str(i*5)+'sec', 'isf5_c-'+str(i*5)+'sec', 'isf6_c-'+str(i*5)+'sec', 'orp-'+str(i*5)+'sec', 'temp-'+str(i*5)+'sec']))

x_ph_5s = pd.concat(xl2_ph_5s, axis=1)
x_ph_5s.index = y_ph_2_5s.index
x_ph_5s = x_ph_5s.dropna()
y_ph_5s = y_ph_2_5s
y_ph_5s = y_ph_5s.reindex(x_ph_5s.index)


y_ph_5s_s = y_ph_5s.values.astype('float32')  # scaler.fit_transform(pylab.expand_dims(y_ph_5s.values,axis=1))
x_ph_5s_s = x_ph_5s.values.astype('float32')  # scaler.fit_transform(x_ph_5s.values)

x_ph_5s_r = np.empty([len(x_ph_5s_s), n_input, n_vars]).astype('float32')
for i in range(n_vars):
	x_ph_5s_r[:, :, i] = x_ph_5s_s[:, i:(n_input) * n_vars:n_vars]

batch_size = 20000

# train the model 2.2stateful
def build_model(train_x, train_y):
	noise_std = 0.8
	droprate = 0.0
	# define parameters
	verbose, epochs, batch_size = 1, 2, None
	n_features = len(train_x)
	n_filters = 8
	k_size = 2
	k_init = 'he_uniform'
	pad = 'causal'
	# define model
	in_layers, out_layers = list(), list()
	for n in range(n_features):
		n_timesteps, input_dim, n_outputs = train_x[n].shape[1], train_x[n].shape[2], train_y.shape[1]
		inputs		= layers.Input(batch_shape=(batch_size, n_timesteps, input_dim))#shape=(n_timesteps, 1, 1))
		dcnn0		= layers.Conv1D(filters=n_filters, kernel_size=1, padding=pad, kernel_initializer=k_init, activation='relu')(inputs)
		dcnn1		= layers.Conv1D(filters=n_filters, kernel_size=k_size, dilation_rate=1, kernel_initializer=k_init, padding=pad)(dcnn0)
		dcnn1_bn	= layers.BatchNormalization()(dcnn1)
		dcnn1_relu = layers.ReLU()(dcnn1_bn)
		dcnn2		= layers.Conv1D(filters=n_filters, kernel_size=k_size, dilation_rate=2, kernel_initializer=k_init, padding=pad)(dcnn1_relu)
		dcnn2_bn	= layers.BatchNormalization()(dcnn2)
		dcnn2_relu = layers.ReLU()(dcnn2_bn)
		dcnn3		= layers.Conv1D(filters=n_filters, kernel_size=k_size, dilation_rate=4, kernel_initializer=k_init, padding=pad)(dcnn2_relu)
		dcnn3_bn	= layers.BatchNormalization()(dcnn3)
		dcnn3_relu = layers.ReLU()(dcnn3_bn)
		dcnn4		= layers.Conv1D(filters=n_filters, kernel_size=k_size, dilation_rate=8, kernel_initializer=k_init, padding=pad)(dcnn3_relu)
		dcnn4_bn	= layers.BatchNormalization()(dcnn4)
		dcnn4_relu  = layers.ReLU()(dcnn4_bn)
		# dcnn		= layers.core.Lambda(lambda tt: tt[:, -1, :])(dcnn4_relu)
		dcnn 		= LastRowLayer()(dcnn4_relu)
		in_layers.append(inputs)
		out_layers.append(dcnn)#4_bn)
	# merge heads
	if n_features > 1:
		merged_heads = layers.merge.concatenate(out_layers)
	else:
		merged_heads = out_layers[0]
	noise = layers.GaussianNoise(noise_std)(merged_heads)
	dense = layers.core.Dense(5)(noise)
	outputs = layers.Activation('softmax')(dense)
	model = models.Model(inputs=in_layers, outputs=outputs)
	model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.summary()
	# fit network
	hooks = [callbacks.ReduceLROnPlateau(monitor='val_loss', factor=1/3, patience=2, min_lr=1e-4)]
	return model

train_x_2, test_x_2, train_y_2, test_y_2 = createTrainTest(x_ph_5s_r,y_ph_5s_s, n_test, n_disc)

train_x	= train_x_2#[:int(len(train_x_2)*0.95//batch_size*batch_size)]
val_x	= train_x_2[-int(len(train_x_2)*0.1):]#[-int(len(train_x_2)*0.1//batch_size*batch_size):]
test_x	= test_x_2#[:int(len(test_x_2)//batch_size*batch_size)]
train_y	= train_y_2#[:int(len(train_x_2)*0.95//batch_size*batch_size)]
val_y	= train_y_2[-int(len(train_x_2)*0.1):]#[-int(len(train_x_2)*0.1//batch_size*batch_size):]
test_y	= test_y_2#[:int(len(test_x_2)//batch_size*batch_size)]

train_x_x	= [np.expand_dims(train_x[:,:,i], axis=2) for i in range(train_x.shape[2])]
train_y_y	= np.expand_dims(train_y, axis=1)
val_x_x		= [np.expand_dims(val_x[:,:,i], axis=2) for i in range(val_x.shape[2])]
val_y_y		= np.expand_dims(val_y, axis=1)
test_x_x	= [np.expand_dims(test_x[:,:,i], axis=2) for i in range(test_x.shape[2])]
test_y_y	= np.expand_dims(test_y, axis=1)

print(np.shape(train_x_x))
print(np.shape(train_y_y))

np.savez('x_norm', train_x_x)#[0], train_x_x[1], train_x_x[2])
np.savez('x_test', test_x_x)#[0], test_x_x[1], test_x_x[2])
np.savez('y_test', test_y)

npzfile = np.load('x_test.npz')
print(npzfile.files)
print(npzfile['arr_0'])

npzfile = np.load('x_test.npz')['arr_0']
print(npzfile[7:9])

npzfile_y = np.load('y_test.npz')['arr_0'][:100]
np.shape(npzfile_y)

model = build_model(train_x_x, train_y_y)

alpha = 0.1  # weight decay coefficient

for layer in model.layers:
	if isinstance(layer, layers.Conv1D) or isinstance(layer, layers.core.Dense):
		layer.add_loss(regularizers.l2(alpha)(layer.kernel))
	if hasattr(layer, 'bias_regularizer') and layer.use_bias:
		layer.add_loss(regularizers.l2(alpha)(layer.bias))

folder = ''

history = list()
#for i in range(7):
history.append(model.fit(train_x_x, train_y, validation_split=0.1, epochs=20, batch_size=2000, verbose=1, shuffle=True))#validation_data=(val_x_x,val_y_y), epochs=1, batch_size=1000, verbose=1, shuffle=True))
model.save_weights(folder+"model_weights.h5")
model.save(folder+"model.h5")

model2 = build_model(train_x_x, train_y_y)

alpha = 0.1  # weight decay coefficient

for layer in model2.layers:
	if isinstance(layer, layers.Conv1D) or isinstance(layer, layers.core.Dense):
		layer.add_loss(regularizers.l2(alpha)(layer.kernel))
	if hasattr(layer, 'bias_regularizer') and layer.use_bias:
		layer.add_loss(regularizers.l2(alpha)(layer.bias))

model2.load_weights(folder+"model23_DCNN_Realtaste_s10_k8_d8_noise08_lab_isf123456orpcondTemp_1s_1000_weights.h5")
model2.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model2.save(folder+"model23_DCNN_Realtaste_s10_k8_d8_noise08_lab_isf123456orpcondTemp_1s_1000_Adam.h5")

loss_hist = []
vloss_hist = []

for i in range(len(history)):
	plt.figure(figsize=(16,5))
	plt.plot(history[i].history['loss'])
	plt.plot(history[i].history['val_loss'])

plt.figure(figsize=(16,5))
plt.plot(val_y[-int(len(val_y)):])
plt.plot(model.predict(val_x_x)[-int(len(val_y)):])

val_x_ol = np.concatenate((train_x, test_x), axis=0)
val_y_ol = np.concatenate((train_y, test_y), axis=0)

plt.figure(figsize=(16,5))
yTrue = val_y_ol[-n_test:]
plt.plot(yTrue)
yPred = model.predict([np.expand_dims(val_x_ol[-n_test:,:,j], axis=2) for j in range(val_x_ol.shape[2])])
plt.plot(yPred)


def r_sq(yTrue, yPred):
	y_mean_line = np.mean(yTrue)
	squared_error_regr = np.sum((yPred - yTrue) * (yPred - yTrue))
	squared_error_y_mean = np.sum((y_mean_line - yTrue) * (y_mean_line - yTrue))
	return -1 + (squared_error_regr/squared_error_y_mean)

print('Test R-squared: '+str(r_sq(yTrue,yPred)))
print('Test mse: '+str(metrics.mean_squared_error(yTrue, yPred)))
print('Test mae: '+str(metrics.mean_absolute_error(yTrue, yPred)))

