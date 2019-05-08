'''
build and train a model//make an psuedo algorithm
'''
import tensorflow as tf
import numpy as np
import pygal
import os
celsius_q    = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)

""" for i,c in enumerate(celsius_q):
  print("{} degrees Celsius = {} degrees Fahrenheit".format(c, fahrenheit_a[i]))
"""

l0 = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([l0])

#loss / / / numeric analysis
model.compile(
  loss='mean_squared_error',
  optimizer=tf.keras.optimizers.Adam(.1)
  )

#print(model.summary())
#saving the model
""" chkpt = "tf.ckpt"
checkpointPath = os.path.dirname(os.path.abspath('./tf'))
cp_callback = tf.keras.callbacks.ModelCheckpoint(
  checkpointPath,
  save_weights_only=True,
  verbose=1
) 
"""
#print(checkpointPath)

history = model.fit(celsius_q,fahrenheit_a, epochs=500, verbose=True, callbacks=[cp_callback])
print('done training')
#print (history.history['loss'])
""" 
chart = pygal.Line()
chart.x_labels = map(str , range(0,500))

chart.add('loss', history.history['loss'])
chart.render_to_file('./temp.svg')
"""
print(model.predict([100]))
#Real = 212


#print("These are the layer variables: {}".format(l0.get_weights()))
