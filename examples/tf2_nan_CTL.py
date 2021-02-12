import logging, os
logging.disable(logging.WARNING)

import tensorflow as tf

import horovod.tensorflow as hvd
import argparse
import numpy as np
import time

parser = argparse.ArgumentParser()
parser.add_argument("--use_amp", action="store_true")
args = parser.parse_args()

# Startup Horovod
hvd.init()

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

# Enable AMP
if args.use_amp:
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
    tf.keras.mixed_precision.experimental.set_policy(policy)

# Create optimizer
opt = tf.keras.optimizers.SGD(learning_rate=1e-6)
opt = hvd.DistributedOptimizer(opt)

if args.use_amp:
    opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)

# Create dummy model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(1, use_bias=False))

for step in range(5):

    ## On rank 0, set input to NaN to generate NaN gradient
    if hvd.rank() == 0 and step % 2 == 0:
        x = np.ones((1, 1), dtype=np.float32)
        x[:, :] = np.nan
    else:
        x = np.ones((1, 1), dtype=np.float32)

    with tf.GradientTape() as tape:
      predictions = model(x)
      loss = tf.reduce_sum(predictions)
      print("rank {} loss {}".format(hvd.rank(), loss))

      # Scale the losses
      if args.use_amp:
          loss = opt.get_scaled_loss(loss)

    tape = hvd.DistributedGradientTape(tape)
    grads = tape.gradient(loss, model.trainable_variables)

    # Unscale the grads
    if args.use_amp:
        grads = opt.get_unscaled_gradients(grads)

    opt.apply_gradients(zip(grads, model.trainable_variables))

print("rank {} completed step...".format(hvd.rank()))

# Sleep to keep ranks from killing job prematurely
time.sleep(10)
