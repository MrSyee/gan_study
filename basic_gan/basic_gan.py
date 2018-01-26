
# GAN

import tensorflow as tf

layers = tf.contrib.layers
tf.reset_default_graph() # tensorboard graph reset

class GAN(object):
    def __init__(self):
        # hyperparameter
        self.n_noise = 100

        self.batch_size = 100
        self.n_hidden = 256
        self.n_output = 28*28

        self.real_data = tf.placeholder(tf.float32, [None, self.n_output])
        self.real_data_size = [-1, 28, 28, 1]


    def build_noise(self):
        # Setup variable of random vector z
        with tf.variable_scope('random_z'):
            self.random_z = tf.random_uniform([self.batch_size, 1, 1, self.n_noise],
                                              minval=-1,
                                              maxval=1,
                                              dtype=tf.float32)
        return self.random_z


    def setup_global_step(self):
        """Sets up the global step Tensor."""
        self.global_step = tf.Variable(initial_value=0,
                                       name='global_step',
                                       trainable=False,
                                       collections=[tf.GraphKeys.GLOBAL_STEP,
                                                    tf.GraphKeys.GLOBAL_VARIABLES])

        print('complete setup global_step.')

    def generator(self, z, is_training=True, reuse=False):
        # generator
        with tf.variable_scope('G') as scope:
            if reuse:
                scope.reuse_variables()

            self.layer1 = layers.fully_connected(inputs=z,
                                                 num_outputs=self.n_hidden,
                                                 activation_fn=tf.nn.relu,
                                                 trainable=is_training,
                                                 scope='layer1')
            self.layer2 = layers.fully_connected(inputs=self.layer1,
                                                 num_outputs=self.n_output,
                                                 activation_fn=tf.nn.sigmoid,
                                                 trainable=is_training,
                                                 scope='layer2')
            output = self.layer2

            return output

    def discriminator(self, inputs, reuse=False):
        is_training = True
        # discriminator
        with tf.variable_scope('D') as scope:
            if reuse:
                scope.reuse_variables()

            self.layer1 = layers.fully_connected(inputs=inputs,
                                                 num_outputs=self.n_hidden,
                                                 activation_fn=tf.nn.relu,
                                                 trainable=is_training,
                                                 scope='layer1')
            self.layer2 = layers.fully_connected(inputs=self.layer1,
                                                 num_outputs=1,
                                                 activation_fn=tf.nn.sigmoid,
                                                 trainable=is_training,
                                                 scope='layer2')
            output = self.layer2

            return output

    def GANLoss(self, logits, is_real, scope=None):
        if is_real:
            labels = tf.ones_like(logits)

        elif is_real==False:
            labels = tf.zeros_like(logits)

        loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels,
                                               logits=logits,
                                               scope=scope)
        return loss


    def build(self):
        # setup global step
        self.setup_global_step()

        # Generating random noise z
        self.noise_z = self.build_noise()

        # Generating data from Generator using noise z
        self.generated_data = self.generator(self.noise_z)
        self.sample_data = self.generator(self.noise_z, is_training=False, reuse=True)

        # Discriminating real data by Discriminator
        self.real_logits = self.discriminator(self.real_data, reuse=False)

        # Discriminating fake data generated from Generator using Discriminator
        self.fake_logits = self.discriminator(self.generated_data, reuse=True)

        # Loss of Discriminator
        self.loss_real = self.GANLoss(logits=self.real_logits,
                                      is_real=True,
                                      scope='loss_D_real')
        self.loss_fake = self.GANLoss(logits=self.fake_logits,
                                      is_real=False,
                                      scope='loss_D_fake')

        self.loss_D = self.loss_real + self.loss_fake

        # Loss of Generator
        self.loss_G = self.GANLoss(logits=self.fake_logits,
                                           is_real=True,
                                           scope='loss_G')

        # Separate variables for each function
        self.D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D')
        self.G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G')

        for var in self.D_vars:
            print(var.name)
        print('\n')
        for var in self.G_vars:
            print(var.name)
        print('\n')

        tf.summary.scalar('losses/loss_D', self.loss_D)
        tf.summary.scalar('losses/loss_G', self.loss_G)

        tf.summary.image('sample_images', tf.reshape(self.sample_data, self.real_data_size), max_outputs=6)
        tf.summary.image('real_images', tf.reshape(self.real_data, self.real_data_size))

        print('complete model build.')


