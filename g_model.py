import tensorflow as tf
import numpy as np
from scipy.misc import imsave
from skimage.transform import resize
import os

import constants as c
from loss_functions import temporal_combined_loss
from utils import psnr_error, sharp_diff_error, ssim_error
from tfutils import w, b, unpool


# noinspection PyShadowingNames
class GeneratorModel:
    def __init__(self, session, summary_writer, height_train, width_train, height_test,
                 width_test, scale_layer_fms, scale_kernel_sizes, scale_is_unpooling, scale_is_batch_norm,
                 scale_gt_inverse_scale_factor):
        """
        Initializes a GeneratorModel.

        @param session: The TensorFlow Session.
        @param summary_writer: The writer object to record TensorBoard summaries
        @param height_train: The height of the input images for training.
        @param width_train: The width of the input images for training.
        @param height_test: The height of the input images for testing.
        @param width_test: The width of the input images for testing.
        @param scale_layer_fms: The number of feature maps in each layer of each scale network.
        @param scale_kernel_sizes: The size of the kernel for each layer of each scale network.

        @type session: tf.Session
        @type summary_writer: tf.train.SummaryWriter
        @type height_train: int
        @type width_train: int
        @type height_test: int
        @type width_test: int
        @type scale_layer_fms: list<list<int>>
        @type scale_kernel_sizes: list<list<int>>
        """
        self.sess = session
        self.summary_writer = summary_writer
        self.height_train = height_train
        self.width_train = width_train
        self.height_test = height_test
        self.width_test = width_test
        self.scale_layer_fms = scale_layer_fms
        self.scale_kernel_sizes = scale_kernel_sizes
        self.scale_is_unpooling = scale_is_unpooling
        self.scale_is_batch_norm = scale_is_batch_norm
        self.scale_gt_inverse_scale_factor = scale_gt_inverse_scale_factor
        self.num_scale_nets = len(scale_layer_fms)

    # noinspection PyAttributeOutsideInit
    def define_graph(self, discriminator):
        """
        Sets up the model graph in TensorFlow.

        @param discriminator: The discriminator model that discriminates frames generated by this
                              model.
        """
        with tf.name_scope('generator'):
            ##
            # Data
            ##

            with tf.name_scope('input'):
                self.input_frames_train = tf.placeholder(
                    tf.float32, shape=[None, self.height_train, self.width_train, 3 * c.HIST_LEN],
                    name='input_frames_train')
                self.gt_frames_train = tf.placeholder(
                    tf.float32, shape=[None, self.height_train, self.width_train, 3 * c.GT_LEN], name='gt_frames_train')

                self.input_frames_test = tf.placeholder(
                    tf.float32, shape=[None, self.height_test, self.width_test, 3 * c.HIST_LEN],
                    name='input_frames_test')
                self.gt_frames_test = tf.placeholder(
                    tf.float32, shape=[None, self.height_test, self.width_test, 3 * c.GT_LEN], name='gt_frames_test')

                # use variable batch_size for more flexibility
                with tf.name_scope('batch_size_train'):
                    self.batch_size_train = tf.shape(self.input_frames_train, name='input_frames_train_shape')[0]
                with tf.name_scope('batch_size_test'):
                    self.batch_size_test = tf.shape(self.input_frames_test, name='input_frames_test_shape')[0]

            ##
            # Scale network setup and calculation
            ##

            self.train_vars = []  # the variables to train in the optimization step

            self.summaries_train = []
            self.scale_preds_train = []  # the generated images at each scale
            self.scale_gts_train = []  # the ground truth images at each scale
            self.d_scale_preds = []  # the predictions from the discriminator model

            self.summaries_test = []
            self.scale_preds_test = []  # the generated images at each scale
            self.scale_gts_test = []  # the ground truth images at each scale

            self.ws = []
            self.bs = []
            for scale_num in xrange(self.num_scale_nets):
                with tf.name_scope('scale_net_' + str(scale_num)):
                    with tf.name_scope('setup'):
                        scale_ws = []
                        scale_bs = []

                        # create weights for kernels
                        with tf.name_scope('weights'):
                            for i in xrange(len(self.scale_kernel_sizes[scale_num])):
                                scale_ws.append(w([self.scale_kernel_sizes[scale_num][i],
                                                   self.scale_kernel_sizes[scale_num][i],
                                                   self.scale_layer_fms[scale_num][i],
                                                   self.scale_layer_fms[scale_num][i + 1]],
                                                  'gen_' + str(scale_num) + '_' + str(i)))

                        with tf.name_scope('biases'):
                            for i in xrange(len(self.scale_kernel_sizes[scale_num])):
                                scale_bs.append(b([self.scale_layer_fms[scale_num][i + 1]]))

                        # add to trainable parameters
                        self.train_vars += scale_ws
                        self.train_vars += scale_bs

                        self.ws.append(scale_ws)
                        self.bs.append(scale_bs)

                    with tf.name_scope('calculation'):
                        with tf.name_scope('calculation_train'):
                            ##
                            # Perform train calculation
                            ##
                            if scale_num > 0:
                                last_scale_pred_train = self.scale_preds_train[scale_num - 1]
                            else:
                                last_scale_pred_train = None

                            train_preds, train_gts = self.generate_predictions(scale_num,
                                                                               self.height_train,
                                                                               self.width_train,
                                                                               self.input_frames_train,
                                                                               self.gt_frames_train,
                                                                               last_scale_pred_train)

                        with tf.name_scope('calculation_test'):
                            ##
                            # Perform test calculation
                            if scale_num > 0:
                                last_scale_pred_test = self.scale_preds_test[scale_num - 1]
                            else:
                                last_scale_pred_test = None

                            test_preds, test_gts = self.generate_predictions(scale_num,
                                                                             self.height_test,
                                                                             self.width_test,
                                                                             self.input_frames_test,
                                                                             self.gt_frames_test,
                                                                             last_scale_pred_test,
                                                                             'test')

                        self.scale_preds_train.append(train_preds)
                        self.scale_gts_train.append(train_gts)

                        self.scale_preds_test.append(test_preds)
                        self.scale_gts_test.append(test_gts)

            ##
            # Get Discriminator Predictions
            ##

            if c.ADVERSARIAL:
                with tf.name_scope('d_preds'):
                    # A list of the prediction tensors for each scale network
                    self.d_scale_preds = []

                    for scale_num in xrange(self.num_scale_nets):
                        with tf.name_scope('scale_' + str(scale_num)):
                            with tf.name_scope('calculation'):
                                input_scale_factor = 1. / self.scale_gt_inverse_scale_factor[scale_num]
                                input_scale_height = int(self.height_train * input_scale_factor)
                                input_scale_width = int(self.width_train * input_scale_factor)

                                scale_inputs_train = tf.image.resize_images(self.input_frames_train,
                                                                            [input_scale_height, input_scale_width])

                                # get predictions from the d scale networks
                                self.d_scale_preds.append(
                                    discriminator.scale_nets[scale_num].generate_all_predictions(
                                        scale_inputs_train, self.scale_preds_train[scale_num]))

            ##
            # Training
            ##

            with tf.name_scope('training'):
                # global loss is the combined loss from every scale network
                self.global_loss = temporal_combined_loss(self.scale_preds_train,
                                                          self.scale_gts_train,
                                                          self.d_scale_preds)

                with tf.name_scope('train_step'):
                    self.global_step = tf.Variable(0, trainable=False, name='global_step')
                    self.optimizer = tf.train.AdamOptimizer(learning_rate=c.LRATE_G, name='optimizer')

                    self.train_op = self.optimizer.minimize(self.global_loss,
                                                            global_step=self.global_step,
                                                            var_list=self.train_vars,
                                                            name='train_op')

                    # train loss summary
                    loss_summary = tf.summary.scalar('train_loss_G', self.global_loss)
                    self.summaries_train.append(loss_summary)

            ##
            # Error
            ##

            with tf.name_scope('error'):
                # error computation
                # get error at largest scale
                with tf.name_scope('psnr_train'):
                    self.psnr_error_train = []
                    for gt_num in xrange(c.GT_LEN):
                        self.psnr_error_train.append(psnr_error(self.scale_preds_train[-1][:, :, :,
                                                                gt_num * 3: (gt_num + 1) * 3],
                                                                self.gt_frames_train[:, :, :,
                                                                gt_num * 3: (gt_num + 1) * 3]))
                with tf.name_scope('sharpdiff_train'):
                    self.sharpdiff_error_train = []
                    for gt_num in xrange(c.GT_LEN):
                        self.sharpdiff_error_train.append(sharp_diff_error(self.scale_preds_train[-1][:, :, :,
                                                                           gt_num * 3: (gt_num + 1) * 3],
                                                                           self.gt_frames_train[:, :, :,
                                                                           gt_num * 3: (gt_num + 1) * 3]))
                with tf.name_scope('ssim_train'):
                    self.ssim_error_train = []
                    for gt_num in xrange(c.GT_LEN):
                        self.ssim_error_train.append(ssim_error(self.scale_preds_train[-1][:, :, :,
                                                                gt_num * 3: (gt_num + 1) * 3],
                                                                self.gt_frames_train[:, :, :,
                                                                gt_num * 3: (gt_num + 1) * 3]))
                with tf.name_scope('psnr_test'):
                    self.psnr_error_test = []
                    for gt_num in xrange(c.GT_LEN):
                        self.psnr_error_test.append(psnr_error(self.scale_preds_test[-1][:, :, :,
                                                               gt_num * 3: (gt_num + 1) * 3],
                                                               self.gt_frames_test[:, :, :,
                                                               gt_num * 3: (gt_num + 1) * 3]))
                with tf.name_scope('sharpdiff_test'):
                    self.sharpdiff_error_test = []
                    for gt_num in xrange(c.GT_LEN):
                        self.sharpdiff_error_test.append(sharp_diff_error(self.scale_preds_test[-1][:, :, :,
                                                                          gt_num * 3: (gt_num + 1) * 3],
                                                                          self.gt_frames_test[:, :, :,
                                                                          gt_num * 3: (gt_num + 1) * 3]))
                with tf.name_scope('ssim_test'):
                    self.ssim_error_test = []
                    for gt_num in xrange(c.GT_LEN):
                        self.ssim_error_test.append(ssim_error(self.scale_preds_test[-1][:, :, :,
                                                               gt_num * 3: (gt_num + 1) * 3],
                                                               self.gt_frames_test[:, :, :,
                                                               gt_num * 3: (gt_num + 1) * 3]))
                for gt_num in xrange(c.GT_LEN):
                    # train error summaries
                    summary_psnr_train = tf.summary.scalar('train_PSNR_' + str(gt_num),
                                                           self.psnr_error_train[gt_num])
                    summary_sharpdiff_train = tf.summary.scalar('train_SharpDiff_' + str(gt_num),
                                                                self.sharpdiff_error_train[gt_num])
                    summary_ssim_train = tf.summary.scalar('train_SSIM_' + str(gt_num), self.ssim_error_train[gt_num])
                    self.summaries_train += [summary_psnr_train, summary_sharpdiff_train, summary_ssim_train]

                    # test error summaries
                    summary_psnr_test = tf.summary.scalar('test_PSNR_' + str(gt_num),
                                                          self.psnr_error_test[gt_num])
                    summary_sharpdiff_test = tf.summary.scalar('test_SharpDiff_' + str(gt_num),
                                                               self.sharpdiff_error_test[gt_num])
                    summary_ssim_test = tf.summary.scalar('test_SSIM_' + str(gt_num), self.ssim_error_test[gt_num])
                    self.summaries_test += [summary_psnr_test, summary_sharpdiff_test, summary_ssim_test]

            # add summaries to visualize in TensorBoard
            self.summaries_train = tf.summary.merge(self.summaries_train)
            self.summaries_test = tf.summary.merge(self.summaries_test)

    def generate_predictions(self, scale_num, height, width, inputs, gts, last_gen_frames, mode='train'):
        """
        Generate predicted frames at a specified scale.
        @param scale_num: The scale network with which to generate the frames.
        @param height: The height of the full-scale frames.
        @param width: The width of the full-scale frames.
        @param inputs: The input frames. A tensor of shape
                       [batch_size x height x width x c.HIST_LEN x 3]
        @param gts: The ground truth output frames. A tensor of shape
                    [batch_size x height x width x 3]
        @param last_gen_frames: The frames generated by the previous scale network. Used as input
                                to this scale. A tensor of shape
                                [batch_size x (scale_height / 2) x (scale_width / 2) x 3]
        @param mode: Whether predictions are to be made in train or test mode
        @return: The generated frames. A tensor of shape
                                       [batch_size x scale_height x scale_width x c.GT_LEN x 3]
        """
        # scale inputs and gts
        scale_factor = 1. / 2 ** ((self.num_scale_nets + 1) - scale_num)
        scale_height = int(height * scale_factor)
        scale_width = int(width * scale_factor)

        gt_scale_factor = 1. / self.scale_gt_inverse_scale_factor[scale_num]
        gt_scale_height = int(height * gt_scale_factor)
        gt_scale_width = int(width * gt_scale_factor)

        with tf.name_scope('rescale_input_' + mode):
            if scale_num == 0:
                scale_inputs = tf.image.resize_images(inputs, [scale_height, scale_width])
            else:
                scale_factor = 1. / self.scale_gt_inverse_scale_factor[scale_num - 1]
                scale_height = int(height * scale_factor)
                scale_width = int(width * scale_factor)
                scale_inputs = tf.image.resize_images(inputs, [scale_height, scale_width])
            scale_gts = tf.image.resize_images(gts, [gt_scale_height, gt_scale_width])

        with tf.name_scope('add_last_scale_pred_' + mode):
            # for all scales but the first, add the frame generated by the last
            # scale to the input
            if scale_num > 0:
                scale_inputs = tf.concat([scale_inputs, last_gen_frames], 3)

        # generated frame predictions
        preds = scale_inputs

        # perform convolutions
        with tf.name_scope('convolutions_' + mode):
            for i in xrange(len(self.scale_kernel_sizes[scale_num])):
                with tf.name_scope('conv2d_layer' if self.scale_is_unpooling[scale_num][i] == 0
                                   else 'conv2d_transpose_layer'):
                    with tf.variable_scope(
                            'conv2d_layer_' + str(scale_num) + str(i) if self.scale_is_unpooling[scale_num][i] == 0
                            else 'conv2d_transpose_layer_' + str(scale_num) + str(i)):
                        # Convolve layer. If layer is a fractionally strided convolution,
                        #  perform unpooling first
                        if self.scale_is_unpooling[scale_num][i] == 1:
                            preds = unpool(preds)
                        preds = tf.nn.conv2d(preds, self.ws[scale_num][i], [1, 1, 1, 1], padding=c.PADDING_G)

                        if self.scale_is_batch_norm[scale_num][i] == 1:
                            preds = tf.contrib.layers.batch_norm(preds,
                                                                 center=True, scale=True,
                                                                 is_training=True if
                                                                 mode == 'train' else False,
                                                                 reuse=None if
                                                                 mode == 'train' else True,
                                                                 updates_collections=None,
                                                                 decay=0.5,
                                                                 scope='bn')
                        # Activate with ReLU (or Tanh for last layer)
                        if i == len(self.scale_kernel_sizes[scale_num]) - 1:
                            preds = tf.nn.tanh(preds + self.bs[scale_num][i])
                        else:
                            preds = tf.nn.leaky_relu(preds + self.bs[scale_num][i])

        return preds, scale_gts

    def train_step(self, batch):
        """
        Runs a training step using the global loss on each of the scale networks.

        @param batch: An array of shape
                      [c.BATCH_SIZE x self.height x self.width x (3 * (c.HIST_LEN + 1))].
                      The input and output frames, concatenated along the channel axis (index 3).

        @return: The global step.
        """
        ##
        # Split into inputs and outputs
        ##

        input_frames = batch[:, :, :, :-3 * c.GT_LEN]
        gt_frames = batch[:, :, :, -3 * c.GT_LEN:]

        ##
        # Train
        ##

        feed_dict = {self.input_frames_train: input_frames, self.gt_frames_train: gt_frames}

        if c.ADVERSARIAL:
            # Run the generator first to get generated frames
            scale_preds = self.sess.run(self.scale_preds_train, feed_dict=feed_dict)

        _, global_loss, global_psnr_error, global_sharpdiff_error, global_ssim_error, global_step, summaries = \
            self.sess.run([self.train_op,
                           self.global_loss,
                           self.psnr_error_train,
                           self.sharpdiff_error_train,
                           self.ssim_error_train,
                           self.global_step,
                           self.summaries_train],
                          feed_dict=feed_dict)

        ##
        # User output
        ##
        if global_step % c.STATS_FREQ == 0:
            print 'GeneratorModel : Step ', global_step
            print '                 Global Loss     : ', global_loss
            print '                 SSIM Errors     : ', [ssim_error for ssim_error in global_ssim_error]
            print '                 PSNR Errors     : ', [psnr_error for psnr_error in global_psnr_error]
            print '                 Sharpdiff Errors: ', [sharpdiff_error for sharpdiff_error in global_sharpdiff_error]
        if global_step % c.SUMMARY_FREQ == 0:
            self.summary_writer.add_summary(summaries, global_step)
            print 'GeneratorModel: saved summaries'
        if global_step % c.IMG_SAVE_FREQ == 0:
            print '-' * 30
            print 'Saving images...'

            # if not adversarial, we didn't get the preds for each scale net before for the
            # discriminator prediction, so do it now
            if not c.ADVERSARIAL:
                scale_preds = self.sess.run(self.scale_preds_train, feed_dict=feed_dict)

            # re-generate scale gt_frames to avoid having to run through TensorFlow.
            scale_gts = []
            for scale_num in xrange(self.num_scale_nets):
                scale_factor = 1. / 2 ** ((self.num_scale_nets - 1) - scale_num)
                scale_height = int(self.height_train * scale_factor)
                scale_width = int(self.width_train * scale_factor)

                # resize gt_output_frames for scale and append to scale_gts_train
                scaled_gt_frames = np.empty([c.BATCH_SIZE, scale_height, scale_width, 3 * c.GT_LEN])
                for i, img in enumerate(gt_frames):
                    # for skimage.transform.resize, images need to be in range [0, 1], so normalize to
                    # [0, 1] before resize and back to [-1, 1] after
                    sknorm_img = (img / 2) + 0.5
                    resized_frame = resize(sknorm_img, [scale_height, scale_width, 3 * c.GT_LEN])
                    scaled_gt_frames[i] = (resized_frame - 0.5) * 2
                scale_gts.append(scaled_gt_frames)

            # for every clip in the batch, save the inputs, scale preds and scale gts
            for pred_num in xrange(len(input_frames)):
                pred_dir = c.get_dir(os.path.join(c.IMG_SAVE_DIR, 'Step_' + str(global_step),
                                                  str(pred_num)))

                # save input images
                for frame_num in xrange(c.HIST_LEN):
                    img = input_frames[pred_num, :, :, (frame_num * 3):((frame_num + 1) * 3)]
                    imsave(os.path.join(pred_dir, 'input_' + str(frame_num) + '.jpg'), img)

                # save preds and gts at each scale
                # noinspection PyUnboundLocalVariable
                for scale_num, scale_pred in enumerate(scale_preds):
                    gen_imgs = scale_pred[pred_num]

                    path = os.path.join(pred_dir, 'scale_' + str(scale_num))
                    gt_imgs = scale_gts[scale_num][pred_num]

                    for frame_num in xrange(c.GT_LEN):
                        gen_img = gen_imgs[:, :, frame_num * 3:(frame_num + 1) * 3]
                        gt_img = gt_imgs[:, :, frame_num * 3:(frame_num + 1) * 3]

                        imsave(path + '_gen_' + str(frame_num) + '.jpg', gen_img)
                        imsave(path + '_gt_' + str(frame_num) + '.jpg', gt_img)

            print 'Saved images!'
            print '-' * 30

        return global_step

    def test_batch(self, batch, global_step, num_rec_out=1, save_imgs=True):
        """
        Runs a training step using the global loss on each of the scale networks.

        @param batch: An array of shape
                      [batch_size x self.height x self.width x (3 * (c.HIST_LEN+ num_rec_out))].
                      A batch of the input and output frames, concatenated along the channel axis
                      (index 3).
        @param global_step: The global step.
        @param num_rec_out: The number of outputs to predict. Outputs > 1 are computed recursively,
                            using previously-generated frames as input. Default = 1.
        @param save_imgs: Whether or not to save the input/output images to file. Default = True.

        @return: A tuple of (psnr error, sharpdiff error) for the batch.
        """
        if num_rec_out < 1:
            raise ValueError('num_rec_out must be >= 1')

        print '-' * 30
        print 'Testing:'

        ##
        # Split into inputs and outputs
        ##

        input_frames = batch[:, :, :, :3 * c.HIST_LEN]
        gt_frames = batch[:, :, :, 3 * c.HIST_LEN:]

        ##
        # Generate num_rec_out recursive predictions
        ##

        feed_dict = {self.input_frames_test: input_frames,
                     self.gt_frames_test: gt_frames}
        preds, psnr, sharpdiff, ssim, summaries = self.sess.run([self.scale_preds_test[-1],
                                                                self.psnr_error_test,
                                                                self.sharpdiff_error_test,
                                                                self.ssim_error_test,
                                                                self.summaries_test],
                                                                feed_dict=feed_dict)

        print 'SSIM Errors     : ', [ssim_error for ssim_error in ssim]
        print 'PSNR Errors     : ', [psnr_error for psnr_error in psnr]
        print 'Sharpdiff Errors: ', [sharpdiff_error for sharpdiff_error in sharpdiff]

        # write summaries
        self.summary_writer.add_summary(summaries, global_step)

        ##
        # Save images
        ##

        if save_imgs:
            for pred_num in xrange(len(input_frames)):
                pred_dir = c.get_dir(os.path.join(
                    c.IMG_SAVE_DIR, 'Tests/Step_' + str(global_step), str(pred_num)))

                # save input images
                for frame_num in xrange(c.HIST_LEN):
                    img = input_frames[pred_num, :, :, (frame_num * 3):((frame_num + 1) * 3)]
                    imsave(os.path.join(pred_dir, 'input_' + str(frame_num) + '.jpg'), img)

                # save recursive outputs
                for rec_num in xrange(num_rec_out):
                    gen_img = preds[pred_num, :, :, 3 * rec_num: 3 * (rec_num + 1)]
                    gt_img = gt_frames[pred_num, :, :, 3 * rec_num: 3 * (rec_num + 1)]
                    imsave(os.path.join(pred_dir, '_gen_' + str(rec_num) + '.jpg'), gen_img)
                    imsave(os.path.join(pred_dir, '_gt_' + str(rec_num) + '.jpg'), gt_img)

        print '-' * 30
