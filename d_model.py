import tensorflow as tf

from d_scale_model import DScaleModel
from loss_functions import adv_loss
import constants as c


# noinspection PyShadowingNames
class DiscriminatorModel:
    def __init__(self, session, summary_writer, height, width, scale_conv_layer_fms,
                 scale_kernel_sizes, scale_fc_layer_sizes, inverse_scale_factor):
        """
        Initializes a DiscriminatorModel.

        @param session: The TensorFlow session.
        @param summary_writer: The writer object to record TensorBoard summaries.
        @param height: The height of the input images.
        @param width: The width of the input images.
        @param scale_conv_layer_fms: The number of feature maps in each convolutional layer of each
                                     scale network.
        @param scale_kernel_sizes: The size of the kernel for each layer of each scale network.
        @param scale_fc_layer_sizes: The number of nodes in each fully-connected layer of each scale
                                     network.

        @type session: tf.Session
        @type summary_writer: tf.train.SummaryWriter
        @type height: int
        @type width: int
        @type scale_conv_layer_fms: list<list<int>>
        @type scale_kernel_sizes: list<list<int>>
        @type scale_fc_layer_sizes: list<list<int>>
        """
        self.sess = session
        self.summary_writer = summary_writer
        self.height = height
        self.width = width
        self.scale_conv_layer_fms = scale_conv_layer_fms
        self.scale_kernel_sizes = scale_kernel_sizes
        self.scale_fc_layer_sizes = scale_fc_layer_sizes
        self.inverse_scale_factor = inverse_scale_factor
        self.num_scale_nets = len(scale_conv_layer_fms)
        self.lrate = c.LRATE_D

        self.train_vars = []  # the variables to train in the optimization step
        self.setup_scale_nets()

    # noinspection PyAttributeOutsideInit
    def setup_scale_nets(self):
        """
        Setup scale networks. Each will make the predictions for images at a given scale. Done
        separately from define_graph() so that the generator can define its graph using the
        discriminator scale nets before this defines its graph using the generator.
        """

        self.scale_nets = []
        for scale_num in xrange(self.num_scale_nets):
            with tf.name_scope('scale_net_' + str(scale_num)):
                scale_factor = 1. / self.inverse_scale_factor[scale_num]
                scale_model = DScaleModel(scale_num,
                                          int(self.height * scale_factor),
                                          int(self.width * scale_factor),
                                          self.scale_conv_layer_fms[scale_num],
                                          self.scale_kernel_sizes[scale_num],
                                          self.scale_fc_layer_sizes[scale_num])
                self.scale_nets.append(scale_model)

                self.train_vars += scale_model.train_vars

    # noinspection PyAttributeOutsideInit
    def define_graph(self, generator):
        """
        Sets up the model graph in TensorFlow.

        @param generator: The generator model that generates frames for this to discriminate.
        """

        with tf.name_scope('discriminator'):
            ##
            # Data
            ##

            self.input_clips = tf.placeholder(
                tf.float32, shape=[None, self.height, self.width, (c.HIST_LEN + c.GT_LEN) * 3])

            self.g_input_frames = self.input_clips[:, :, :, :c.HIST_LEN * 3]
            self.gt_frames = self.input_clips[:, :, :, c.HIST_LEN * 3:]
            input_shape = tf.shape(self.g_input_frames)
            batch_size = input_shape[0]

            ##
            # Get Generator Frames
            ##

            with tf.name_scope('gen_frames'):
                self.g_scale_preds = []  # the generated images at each scale
                self.scale_gts = []  # the ground truth images at each scale
                self.resized_inputs = []  # the resized input images at each scale

                for scale_num in xrange(self.num_scale_nets):
                    with tf.name_scope('scale_' + str(scale_num)):
                        # for all scales but the first, add the frame generated by the last
                        # scale to the input

                        if scale_num > 0:
                            last_scale_pred = self.g_scale_preds[scale_num - 1]
                        else:
                            last_scale_pred = None

                        # calculate
                        train_preds, train_gts = generator.generate_predictions(scale_num,
                                                                                self.height,
                                                                                self.width,
                                                                                self.g_input_frames,
                                                                                self.gt_frames,
                                                                                last_scale_pred,
                                                                                'test')

                        input_scale_factor = 1. / self.inverse_scale_factor[scale_num]
                        input_scale_height = int(self.height * input_scale_factor)
                        input_scale_width = int(self.width * input_scale_factor)
                        resized_inputs = tf.image.resize_images(self.g_input_frames,
                                                                [input_scale_height, input_scale_width])

                        self.g_scale_preds.append(train_preds)
                        self.scale_gts.append(train_gts)
                        self.resized_inputs.append(resized_inputs)

            # concatenate the generated images and ground truths at each scale
            self.scale_inputs = []
            for scale_num in xrange(self.num_scale_nets):
                self.scale_inputs.append(
                    tf.concat([self.g_scale_preds[scale_num], self.scale_gts[scale_num]], 0))

            # create the labels
            self.labels = tf.concat([tf.zeros([batch_size, 1]), tf.ones([batch_size, 1])], 0)

            ##
            # Calculation
            ##

            # A list of the prediction tensors for each scale network
            self.scale_preds = []

            for scale_num in xrange(self.num_scale_nets):
                with tf.name_scope('scale_' + str(scale_num)):
                    with tf.name_scope('calculation'):
                        # get predictions from the scale network
                        self.scale_preds.append(
                            self.scale_nets[scale_num].generate_all_predictions(
                                tf.concat([self.resized_inputs[scale_num], self.resized_inputs[scale_num]], 0),
                                self.scale_inputs[scale_num]))

            ##
            # Training
            ##

            with tf.name_scope('training'):
                # global loss is the combined loss from every scale network
                self.global_loss = adv_loss(self.scale_preds, self.labels)

                with tf.name_scope('train_step'):
                    self.global_step = tf.Variable(0, trainable=False, name='global_step')
                    self.optimizer = tf.train.GradientDescentOptimizer(self.lrate, name='optimizer')
                    self.train_op = self.optimizer.minimize(self.global_loss, var_list=self.train_vars, name='train_op',
                                                            global_step=self.global_step)

                    # Accuracy test
                    all_preds = tf.stack(self.scale_preds)
                    self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(all_preds), self.labels),
                                                           tf.int32))

                    # add summaries to visualize in TensorBoard
                    loss_summary = tf.summary.scalar('loss_D', self.global_loss)
                    accuracy_summary = tf.summary.scalar('accuracy_D', self.accuracy)
                    self.summaries = tf.summary.merge([loss_summary, accuracy_summary])

    def train_step(self, batch):
        """
        Runs a training step using the global loss on each of the scale networks.

        @param batch: An array of shape
                      [BATCH_SIZE x self.height x self.width x (3 * (HIST_LEN + GT_LEN))]. The input
                      and output frames, concatenated along the channel axis (index 3).

        @return: The global step.
        """

        ##
        # Train
        ##

        feed_dict = {self.input_clips: batch}

        _, global_loss, global_step, summaries = self.sess.run(
            [self.train_op, self.global_loss, self.global_step, self.summaries],
            feed_dict=feed_dict)

        ##
        # User output
        ##

        if global_step % c.STATS_FREQ == 0:
            print 'DiscriminatorModel: step %d | global loss: %f' % (global_step, global_loss)
        if global_step % c.SUMMARY_FREQ == 0:
            print 'DiscriminatorModel: saved summaries'
            self.summary_writer.add_summary(summaries, global_step)

        return global_step