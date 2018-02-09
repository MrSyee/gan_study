
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np
import basic_gan as gan

import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
# download mnist data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

version = 1

learning_rate = 0.0001

total_epoch = 1000
print_steps = 100
save_steps = 200
summary_steps = 10

logdir = "train/{}".format(str(version))

def save_sample_data(sample_data, reconstruct_data, global_step, max_print=8):
    if not os.path.isdir("./samples/".format(str(version))):
        os.mkdir("samples")
    if not os.path.isdir("./samples/{}/".format(str(version))):
        os.mkdir("samples/{}/".format(str(version)))

    sample_data = sample_data[:max_print, :]
    sample_img = np.reshape(sample_data, [max_print, 28, 28])
    sample_img = sample_img.swapaxes(0, 1)
    sample_img = sample_img.reshape([28, max_print * 28])

    reconstruct_img = reconstruct_data[:max_print, :]
    reconstruct_img = np.reshape(reconstruct_img, [max_print, 28, 28])
    reconstruct_img = reconstruct_img.swapaxes(0, 1)
    reconstruct_img = reconstruct_img.reshape([28, max_print * 28])

    sum_img = np.concatenate((sample_img, reconstruct_img), axis=0)

    plt.figure(figsize=(max_print, 1))
    plt.axis('off')
    plt.imshow(sum_img)
    plt.savefig('samples/{}/global_step_{}.png'.format(str(version), str(global_step).zfill(3)), bbox_inches='tight')


def main(_):
    model = gan.GAN()
    model.build()

    opt_D = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5)
    opt_G = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5)

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_opt_D = opt_D.minimize(model.loss_D,
                                     var_list=model.D_vars)
        train_opt_G = opt_G.minimize(model.loss_G,
                                     global_step=model.global_step,
                                     var_list=model.G_vars)

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000)
    summary_op = tf.summary.merge_all()

    sv = tf.train.Supervisor(logdir=logdir,
                             summary_op=None,
                             saver=saver,
                             save_model_secs=0,
                             init_fn=None)

    sess_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    with sv.managed_session(config=sess_config) as sess:
        tf.logging.info('Start Session.')

        total_batch = int(mnist.train.num_examples/model.batch_size)

        for e in range(total_epoch):
            for _ in range(total_batch):
                trainX = mnist.train.next_batch(model.batch_size)[0]
                _, loss_D = sess.run([train_opt_D, model.loss_D], feed_dict={model.real_data:trainX})
                _, _global_step, loss_G = sess.run([train_opt_G,
                                                    sv.global_step,
                                                    model.loss_G])

            if e == 0 or e % print_steps == 0:
                # print ("loss D : %6f /t loss G : %6f" % [loss_D, loss_G])
                print ("epoch : {:>5}  global_step : {:>5}  lossD : {:>5}  lossG : {:>5}".format(e, _global_step, loss_D, loss_G))

                sample_img = sess.run(model.sample_data, feed_dict={})
                save_sample_data(sample_img, _global_step)

            if e == 0 or e % summary_steps == 0:
                summary_str = sess.run(summary_op, feed_dict={model.real_data: trainX})
                sv.summary_computed(sess, summary_str)

            if e == 0 or e % save_steps == 0:
                tf.logging.info('Saving model with global step %d to disk.' % _global_step)
                sv.saver.save(sess, sv.save_path, global_step=sv.global_step)

        tf.logging.info('complete training...')

        #writer = tf.summary.FileWriter("./graphs/basic_gan", sess.graph)
    #writer.close()

if __name__ == '__main__':
    tf.app.run()