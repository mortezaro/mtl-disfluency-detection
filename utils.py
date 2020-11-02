import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

colors = tf.cast(tf.stack(config.colors[config.working_dataset]), tf.float32)  # / 255
FLAGS = tf.app.flags.FLAGS


def restoring_logs(logfile):
  '''
  Fixed - will now not delete existing log files but add sub-index to path
  :param logfile:
  :return:
  '''
  if tf.gfile.Exists(logfile):
    print('logfile already exist: %s' % logfile)
    # i = 1
    # while os.path.exists(logfile + '_' + str(i)):
    #   i += 1
    # logfile = logfile + '_' + str(i)
    # print('Creating anf writing to: %s' % logfile)
    tf.gfile.DeleteRecursively(logfile)
  tf.gfile.MakeDirs(logfile)



def label_id(logits):
  softmax = tf.nn.softmax(logits)
  argmax = tf.argmax(softmax, 3)
  argmax_expand = tf.expand_dims(argmax, -1)
  return tf.cast(argmax_expand*7, tf.float32)

def disparity(logits):
  return tf.cast(logits, tf.float32)


def accuracy(logits, labels):
  if FLAGS.need_resize:
    labels = tf.image.resize_images(labels, [FLAGS.output_height, FLAGS.output_width],
                                    method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
  softmax = tf.nn.softmax(logits, dim=3)
  argmax = tf.argmax(softmax, 3)

  shape = logits.get_shape().as_list()
  n = shape[3]

  one_hot = tf.one_hot(argmax, n, dtype=tf.float32)
  equal_pixels = tf.reduce_sum(tf.to_float(color_mask(one_hot, labels)))
  total_pixels = reduce(lambda x, y: x * y, [FLAGS.batch] + shape[1:3])
  return equal_pixels / total_pixels


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)




def get_run_list(logits, INF_FLAGS):
    run_list = []
    if INF_FLAGS['use_label_type']:
        label_id_image = rgb(logits[0])
        run_list.append(tf.cast(label_id_image, tf.uint8))
    if INF_FLAGS['use_label_inst']:
        run_list.append(logits[1])
    if INF_FLAGS['use_label_disp']:
        run_list.append(logits[2])
    return run_list

def pred_list2dict(pred_list, INF_FLAGS):
    pred_dict = {}
    if INF_FLAGS['use_label_disp']:
        image = np.expand_dims(pred_list.pop().squeeze().clip(max=1, min=0)*255, 2).astype('uint8')
        image = np.concatenate([image, image, image], axis=2)
        pred_dict['disp'] = image
    if INF_FLAGS['use_label_inst']:
        pred_dict['instance'] = pred_list.pop().squeeze()
    if INF_FLAGS['use_label_type']:
        pred_dict['label'] = pred_list.pop().squeeze()
    return pred_dict


def calc_instance(label_arr, xy_arr):
    mask = make_mask(label_arr)
    raw_image = np.concatenate([xy_arr, np.expand_dims(mask, axis=2)], axis=2)
    instance_image = OPTICS.calc_clusters_img(raw_image)
    return instance_image.clip(max=255, min=0).astype('uint8')


def make_mask(label_image):
    ids = [24, 26]
    for i, id in enumerate(ids):
        color = config.colors[config.working_dataset][id]
        mask = label_image == color
        mask = mask[:, :, 0] * mask[:, :, 1] * mask[:, :, 2]
        if i == 0:
            total_mask = mask
        else:
            total_mask = total_mask + mask
    return total_mask
