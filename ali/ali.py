
# Adversarial Learned Inference : ALI

import tensorflow as tf

# layers = tf.contrib.layers
slim = tf.contrib.slim

class ALI(object):
    def __init__(self):
        # hyperparameter
        self.n_noise = 100

        self.z_dim = 100

        self.batch_size = 128

        self.n_output = 28*28

        self.real_data = tf.placeholder(tf.float32, [None, self.n_output])
        self.real_data_size = [-1, 28, 28, 1]


    def build_noise(self):
        # Setup variable of random vector z
        with tf.variable_scope('random_z'):
            self.noise_z = tf.placeholder(tf.float32, [None, 1, 1, self.z_dim])
        #return self.noise_z

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

            batch_norm_params = {'decay': 0.9,
                                 'epsilon': 0.001,
                                 'is_training': is_training,
                                 'scope': 'batch_norm'}

            with slim.arg_scope([slim.conv2d_transpose],
                                kernel_size=[4, 4],
                                stride=[2, 2],
                                padding='SAME',
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                activation_fn=tf.nn.leaky_relu):

                # noise_z 100 dim -> inputs 1 x 1 x 100 dim
                self.inputs = tf.reshape(z, [-1, 1, 1, self.z_dim])
                # layer1 3 x 3 x 256 dim
                self.layer1 = slim.conv2d_transpose(inputs=self.inputs,
                                                    num_outputs=256,
                                                    kernel_size=[3, 3],
                                                    stride=[1, 1],
                                                    padding='VALID',
                                                    scope='layer1')
                # layer2 7 x 7 x 128 dim
                self.layer2 = slim.conv2d_transpose(inputs=self.layer1,
                                                    num_outputs=128,
                                                    kernel_size=[3, 3],
                                                    padding='VALID',
                                                    scope='layer2')
                # layer3 14 x 14 x 64 dim
                self.layer3 = slim.conv2d_transpose(inputs=self.layer2,
                                                    num_outputs=64,
                                                    scope='layer3')
                # outputs = layer4 28 x 28 x 1 dim
                self.layer4 = slim.conv2d_transpose(inputs=self.layer3,
                                                    num_outputs=1,
                                                    normalizer_fn=None,
                                                    activation_fn=tf.sigmoid,
                                                    scope='layer4')
            outputs = self.layer4
            return outputs

    def encoder(self, data, is_training=True, reuse=False):
        # generator
        with tf.variable_scope('E') as scope:
            if reuse:
                scope.reuse_variables()

            batch_norm_params = {'decay': 0.9,
                                 'epsilon': 0.001,
                                 'is_training': is_training,
                                 'scope': 'batch_norm'}

            with slim.arg_scope([slim.conv2d],
                                kernel_size=[4, 4],
                                stride=[2, 2],
                                padding='SAME',
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                activation_fn=tf.nn.leaky_relu):

                # inputs 28 x 28 x 1 dim
                # layer1 14 x 14 x 64 dim
                self.layer1 = slim.conv2d(inputs=data,
                                          num_outputs=64,
                                          scope='layer1')
                # layer2 7 x 7 x 128 dim
                self.layer2 = slim.conv2d(inputs=self.layer1,
                                          num_outputs=128,
                                          kernel_size=[3, 3],
                                          padding='VALID',
                                          scope='layer2')
                # layer3 3 x 3 x 256 dim
                self.layer3 = slim.conv2d(inputs=self.layer2,
                                          num_outputs=256,
                                          kernel_size=[3, 3],
                                          stride=[1, 1],
                                          padding='VALID',
                                          scope='layer1')
                # layer4 1 x 1 x 100 dim
                self.layer4 = slim.conv2d(inputs=self.layer3,
                                          num_outputs=100,
                                          normalizer_fn=None,
                                          activation_fn=None,
                                          scope='layer4')
            # outputs 100 dim
            outputs = tf.squeeze(self.layer4, axis=[1,2])
            return outputs

    def discriminator(self, data, z, is_training = True, reuse=False):

        # discriminator
        with tf.variable_scope('D') as scope:
            if reuse:
                scope.reuse_variables()

            batch_norm_params = {'decay': 0.9,
                                 'epsilon': 0.001,
                                 'is_training': is_training,
                                 'scope': 'batch_norm'}

            with slim.arg_scope([slim.conv2d],
                                kernel_size=[4, 4],
                                stride=[2, 2],
                                padding='SAME',
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                activation_fn=tf.nn.leaky_relu):

                # D(x)
                # inputs 28 x 28 x 1 dim
                # layer1 14 x 14 x 64 dim
                self.d_x_layer1 = slim.conv2d(inputs=data,
                                          num_outputs=64,
                                          scope='d_x_layer1')
                # layer2 7 x 7 x 128 dim
                self.d_x_layer2 = slim.conv2d(inputs=self.d_x_layer1,
                                          num_outputs=128,
                                          kernel_size=[3, 3],
                                          padding='VALID',
                                          scope='d_x_layer2')
                # layer3 3 x 3 x 256 dim
                self.d_x_layer3 = slim.conv2d(inputs=self.d_x_layer2,
                                          num_outputs=256,
                                          kernel_size=[3, 3],
                                          stride=[1, 1],
                                          padding='VALID',
                                          scope='d_x_layer3')
                # layer4 1 x 1 x 100 dim
                self.d_x_layer4 = slim.conv2d(inputs=self.d_x_layer3,
                                          num_outputs=100,
                                          normalizer_fn=None,
                                          activation_fn=None,
                                          scope='d_x_layer4')

                self.D_x = self.d_x_layer4

            # D(z)
            # noise_z 100 dim -> inputs 1 x 1 x 100 dim
            self.D_z = tf.reshape(z, [-1, 1, 1, self.z_dim])

            # D(x,z)
            # D_x + D_z = 1 x 1 x 200 dim
            self.D_input = tf.concat(self.D_x, self.D_z, axis=3)
            # layer1 1 x 1 x 200 dim, drop out = 0.2
            self.d_layer1 = slim.conv2d(inputs=self.D_input,
                                        num_outputs=200,
                                        kernel_size=[1, 1],
                                        stride=[1, 1],
                                        padding='VALID',
                                        activation_fn=tf.nn.leaky_relu,
                                        scope='d_layer1')
            self.d_layer1 = slim.dropout(self.d_layer1, 0.2, scope='d_layer_dropout1')

            # layer2 1 x 1 x 200 dim, drop out = 0.2
            self.d_layer2 = slim.conv2d(inputs=self.d_layer1,
                                        num_outputs=200,
                                        kernel_size=[1, 1],
                                        stride=[1, 1],
                                        padding='VALID',
                                        activation_fn=tf.nn.leaky_relu,
                                        scope='d_layer2')
            self.d_layer2 = slim.dropout(self.d_layer2, 0.2, scope='d_layer_dropout2')

            # layer3 1 x 1 x 1 dim
            self.d_layer3 = slim.conv2d(inputs=self.d_layer3,
                                        num_outputs=1,
                                        kernel_size=[1, 1],
                                        stride=[1, 1],
                                        padding='VALID',
                                        activation_fn=tf.sigmoid,
                                        scope='d_layer3')
            # outputs 1 dim
            outputs = tf.squeeze(self.d_layer3, axis=[2,3])

            return outputs

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

        # setup noise
        self.build_noise()

        # Generating data from Generator using noise z
        self.generated_data = self.generator(self.noise_z)
        self.inference_code = self.encoder(self.real_data)

        # Generating sample using trained G
        self.sample_data = self.generator(self.noise_z, is_training=False, reuse=True)
        self.sample_inference_code = self.encoder(self.real_data, is_training=False, reuse=True)
        self.reconstruction_data = self.generator(self.sample_inference_code, is_training=False, reuse=True)

        # Discriminating real data by Discriminator
        self.real_logits = self.discriminator(self.real_data, self.inference_code)

        # Discriminating fake data generated from Generator using Discriminator
        self.fake_logits = self.discriminator(self.generated_data, self.noise_z, reuse=True)

        # Loss of Discriminator
        self.loss_D_real = self.GANLoss(logits=self.real_logits,
                                      is_real=True,
                                      scope='loss_D_real')
        self.loss_D_fake = self.GANLoss(logits=self.fake_logits,
                                      is_real=False,
                                      scope='loss_D_fake')

        self.loss_D = self.loss_D_real + self.loss_D_fake

        # Loss of Generator
        self.loss_G_real = self.GANLoss(logits=self.fake_logits,
                                           is_real=False,
                                           scope='loss_G_real')
        self.loss_G_fake = self.GANLoss(logits=self.fake_logits,
                                           is_real=True,
                                           scope='loss_G_fake')

        self.loss_G = self.loss_G_real + self.loss_G_fake

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


