# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
import tensorflow as tf
import horovod.tensorflow as hvd

from tensorflow.keras.mixed_precision import experimental as mixed_precision

BATCH_SIZE = 128

# Benchmark settings
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')

parser.add_argument('--num-iters', type=int, default=10,
                    help='number of benchmark iterations')

parser.add_argument('--use-amp', action='store_true', default=False,
                    help='enable AMP computation')

args = parser.parse_args()

# Horovod: initialize Horovod.
hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

if args.use_amp:
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)

(mnist_images, mnist_labels), _ = \
    tf.keras.datasets.mnist.load_data(
    path='mnist-%d.npz' % hvd.rank()
)

dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(mnist_images[..., tf.newaxis] / 255.0, tf.float32),
             tf.cast(mnist_labels, tf.int64))
)
dataset = dataset.repeat().shuffle(10000).batch(BATCH_SIZE)

mnist_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, [3, 3], activation='relu'),
    tf.keras.layers.Conv2D(64, [3, 3], activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])
loss = tf.losses.SparseCategoricalCrossentropy()

# Horovod: adjust learning rate based on number of GPUs.
opt = tf.optimizers.Adam(0.001 * hvd.size())

if args.use_amp:
    opt = mixed_precision.LossScaleOptimizer(opt, loss_scale='dynamic')

# checkpoint_dir = './checkpoints'
# checkpoint = tf.train.Checkpoint(model=mnist_model, optimizer=opt)


@tf.function
def training_step(images, labels, first_batch):
    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

    with tf.GradientTape() as tape:
        probs = mnist_model(images, training=True)

        import sys
        print("labels:", labels, file=sys.stderr)
        print("probs:", probs, file=sys.stderr)
        loss_value = loss(labels, probs)

        if args.use_amp:
            loss_value = opt.get_scaled_loss(loss_value)

    # Horovod: add Horovod Distributed GradientTape.
    tape = hvd.DistributedGradientTape(tape, compression=compression)

    grads = tape.gradient(loss_value, mnist_model.trainable_variables)

    if args.use_amp:
        grads = opt.get_unscaled_gradients(grads)

    opt.apply_gradients(zip(grads, mnist_model.trainable_variables))

    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    #
    # Note: broadcast should be done after the first gradient step to ensure optimizer
    # initialization.
    if first_batch:
        hvd.broadcast_variables(mnist_model.variables, root_rank=0)
        hvd.broadcast_variables(opt.variables(), root_rank=0)

    return loss_value


# Horovod: adjust number of steps based on number of GPUs.
for batch, (images, labels) in enumerate(dataset.take((BATCH_SIZE * 5) // hvd.size())):
    loss_value = training_step(images, labels, batch == 0)

    if batch % 10 == 0 and hvd.local_rank() == 0:
        print('Step #%d\tLoss: %.6f' % (batch, loss_value))
