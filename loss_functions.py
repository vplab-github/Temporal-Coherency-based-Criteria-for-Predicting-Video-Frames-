import tensorflow as tf
import numpy as np

from tfutils import log10
import constants as c


def temporal_combined_loss(gen_frames, gt_frames, d_preds, lam_adv=0.01, lam_lp=0.025, lam_nccl=0.25, lam_pcdl=0.25,
                           lam_3_pcdl=0.25, l_num=1, patch_size=3, delta=15):
    with tf.name_scope('temporal_combined_loss'):
        with tf.name_scope('batch_size'):
            batch_size = tf.shape(gen_frames[0], name='gen_frames_shape')[0]  # variable batch size as a tensor

        loss = lam_lp * lp_loss(gen_frames, gt_frames, l_num)
        loss += lam_adv * adv_loss(d_preds, tf.ones([batch_size, 1]))
        loss += lam_nccl * nccl_loss(gen_frames, gt_frames, patch_size)
        loss += lam_pcdl * pcdl_loss(gen_frames, d_preds, delta)
        loss += lam_3_pcdl * pcdl_3_loss(gen_frames, d_preds, delta)

        return loss


def nccl_loss(gen_frames, gt_frames, patch_size):
    with tf.name_scope('nccl_loss'):
        score_ncc = 0
        for scale in range(len(gen_frames)):
            gen_frames_s = gen_frames[scale]
            gt_frames_s = gt_frames[scale]
            for batch in range(c.BATCH_SIZE):
                gen_frames_b = gen_frames_s[batch]
                gt_frames_b = gt_frames_s[batch]

                for t in range(gen_frames[0].get_shape()[3] / 3 - 1):
                    gen_frame = gen_frames_b[:, :, (t + 1) * 3: (t + 2) * 3]
                    gt_frame = gt_frames_b[:, :, t * 3: (t + 1) * 3]
                    frame_shape = gen_frame.get_shape()

                    for i in range(2, frame_shape[0], patch_size):
                        if i + patch_size + 2 >= frame_shape[0]:
                            continue
                        for j in range(2, frame_shape[1], patch_size):
                            if j + patch_size + 2 >= frame_shape[1]:
                                continue
                            gen_patch = gen_frame[i:i + patch_size, j:j + patch_size, :]
                            gt_patch = gt_frame[i - 2:i + patch_size + 2, j - 2:j + patch_size + 2, :]

                            gen_norm = tf.image.per_image_standardization(gen_patch)
                            gt_norm = tf.image.per_image_standardization(gt_patch)

                            ncc = tf.nn.conv2d(tf.reshape(gt_norm, [-1, patch_size + 4, patch_size + 4, 3]),
                                               tf.reshape(gen_norm, [patch_size, patch_size, 3, 1]), [1, 1, 1, 1],
                                               'SAME')

                            ncc_sum = tf.reduce_sum(tf.squeeze(ncc))
                            score_ncc += tf.cond(ncc_sum > 0, lambda: ncc_sum, lambda: tf.constant(0, dtype=tf.float32))

                    score_ncc /= tf.floor(tf.cast(frame_shape[0] / patch_size, tf.float32)) ** 2
                score_ncc /= tf.cast(gen_frames[0].get_shape()[3]/3 - 1, tf.float32)
            # score_ncc /= tf.cast(batch_size, tf.float32)
        # score_ncc /= tf.cast(len(gen_frames), tf.float32)

        return -score_ncc


def pcdl_loss(gen_frames, d_preds, delta):
    with tf.name_scope('pdcl_loss'):
        loss = 0
        for scale in range(len(gen_frames)):
            gen_frames_s = gen_frames[scale]
            d_preds_s = d_preds[scale]
            for batch in range(c.BATCH_SIZE):
                gen_frames_b = gen_frames_s[batch]
                d_preds_b = d_preds_s[batch]
                for t in range(gen_frames[0].get_shape()[3] / 3 - 1):
                    gen_frame_t = gen_frames_b[:, :, t * 3: t * 3 + 3]
                    gen_frame_t1 = gen_frames_b[:, :, (t + 1) * 3: (t + 2) * 3]

                    loss += tf.round(d_preds_b[t]) * tf.round(d_preds_b[t + 1]) * \
                        lp_loss_frames(gen_frame_t, gen_frame_t1, 2) + \
                        (1 - tf.round(d_preds_b[t]) * tf.round(d_preds_b[t + 1])) * \
                        tf.maximum(tf.constant(0, dtype=tf.float32),
                                   delta - lp_loss_frames(gen_frame_t, gen_frame_t1, 2))
            # loss /= tf.cast(batch_size, tf.float32)
        # loss /= tf.cast(len(gen_frames), tf.float32)

        return loss


def pcdl_3_loss(gen_frames, d_preds, delta):
    with tf.name_scope('pdcl_3_loss'):
        loss = 0
        for scale in range(len(gen_frames)):
            gen_frames_s = gen_frames[scale]
            d_preds_s = d_preds[scale]
            for batch in range(c.BATCH_SIZE):
                gen_frames_b = gen_frames_s[batch]
                d_preds_b = d_preds_s[batch]
                for t in range(gen_frames[0].get_shape()[3] / 3 - 2):
                    gen_frame_t = gen_frames_b[:, :, t * 3: t * 3 + 3]
                    gen_frame_t1 = gen_frames_b[:, :, (t + 1) * 3: (t + 2) * 3]
                    gen_frame_t2 = gen_frames_b[:, :, (t + 2) * 3: (t + 3) * 3]

                    loss += tf.round(d_preds_b[t]) * tf.round(d_preds_b[t + 1]) * tf.round(d_preds_b[t + 2]) \
                            * lp_loss_frames(tf.abs(tf.subtract(gen_frame_t, gen_frame_t1)),
                                      tf.abs(tf.subtract(gen_frame_t1, gen_frame_t2)), 2) \
                            + (1 - tf.round(d_preds_b[t]) * tf.round(d_preds_b[t + 1]) * tf.round(d_preds_b[t + 2])) \
                            * tf.maximum(tf.constant(0, dtype=tf.float32), delta
                            - lp_loss_frames(tf.abs(tf.subtract(gen_frame_t, gen_frame_t1)),
                                      tf.abs(tf.subtract(gen_frame_t1, gen_frame_t2)), 2))
            # loss /= tf.cast(batch_size, tf.float32)
        # loss /= tf.cast(len(gen_frames), tf.float32)

        return loss


def bce_loss(preds, targets):
    """
    Calculates the sum of binary cross-entropy losses between predictions and ground truths.

    @param preds: A 1xN tensor. The predicted classifications of each frame.
    @param targets: A 1xN tensor The target labels for each frame. (Either 1 or -1). Not "truths"
                    because the generator passes in lies to determine how well it confuses the
                    discriminator.

    @return: The sum of binary cross-entropy losses.
    """
    with tf.name_scope('bce_loss'):
        return tf.squeeze(-1 * (tf.matmul(targets, log10(preds), transpose_a=True) +
                                tf.matmul(1 - targets, log10(1 - preds), transpose_a=True)))


def lp_loss(gen_frames, gt_frames, l_num):
    """
    Calculates the sum of lp losses between the predicted and ground truth frames.

    @param gen_frames: The predicted frames at each scale.
    @param gt_frames: The ground truth frames at each scale
    @param l_num: 1 or 2 for l1 and l2 loss, respectively).

    @return: The lp loss.
    """
    with tf.name_scope('lp_loss'):
        # calculate the loss for each scale
        scale_losses = []
        for i in xrange(len(gen_frames)):
            scale_losses.append(tf.reduce_sum(tf.abs(gen_frames[i] - gt_frames[i])**l_num))

        # condense into one tensor and avg
        return tf.reduce_mean(tf.stack(scale_losses))


def lp_loss_frames(frame_1, frame_2, l_num):
    with tf.name_scope('lp_loss'):
        return tf.reduce_sum(tf.abs(frame_1 - frame_2)**l_num)


def gdl_loss(gen_frames, gt_frames, alpha):
    """
    Calculates the sum of GDL losses between the predicted and ground truth frames.

    @param gen_frames: The predicted frames at each scale.
    @param gt_frames: The ground truth frames at each scale
    @param alpha: The power to which each gradient term is raised.

    @return: The GDL loss.
    """
    with tf.name_scope('gdl_loss'):
        # calculate the loss for each scale
        scale_losses = []
        for i in xrange(len(gen_frames)):
            with tf.name_scope('grad_diff'):
                # create filters [-1, 1] and [[1],[-1]] for diffing to the left and down respectively.
                pos = tf.constant(np.identity(3), dtype=tf.float32)
                neg = -1 * pos
                filter_x = tf.expand_dims(tf.stack([neg, pos]), 0)  # [-1, 1]
                filter_y = tf.stack([tf.expand_dims(pos, 0), tf.expand_dims(neg, 0)])  # [[1],[-1]]
                strides = [1, 1, 1, 1]  # stride of (1, 1)
                padding = 'SAME'

                gen_dx = tf.abs(tf.nn.conv2d(gen_frames[i], filter_x, strides, padding=padding))
                gen_dy = tf.abs(tf.nn.conv2d(gen_frames[i], filter_y, strides, padding=padding))
                gt_dx = tf.abs(tf.nn.conv2d(gt_frames[i], filter_x, strides, padding=padding))
                gt_dy = tf.abs(tf.nn.conv2d(gt_frames[i], filter_y, strides, padding=padding))

                grad_diff_x = tf.abs(gt_dx - gen_dx)
                grad_diff_y = tf.abs(gt_dy - gen_dy)

            scale_losses.append(tf.reduce_sum((grad_diff_x ** alpha + grad_diff_y ** alpha)))

        # condense into one tensor and avg
        return tf.reduce_mean(tf.stack(scale_losses))


def adv_loss(preds, labels):
    """
    Calculates the sum of BCE losses between the predicted classifications and true labels.

    @param preds: The predicted classifications at each scale.
    @param labels: The true labels. (Same for every scale).

    @return: The adversarial loss.
    """
    with tf.name_scope('adv_loss'):
        # calculate the loss for each scale
        scale_losses = []
        for i in xrange(len(preds)):
            loss = bce_loss(preds[i], labels)
            scale_losses.append(loss)

        # condense into one tensor and avg
        return tf.reduce_mean(tf.stack(scale_losses))
