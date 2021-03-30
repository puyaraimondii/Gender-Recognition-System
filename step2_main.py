from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
from data import inputs
import numpy as np
import tensorflow as tf
from model import select_model, get_checkpoint
from utils import *
import os
import json
import csv
import cv2
import sys

import datetime as dt
from time import sleep

'''
#parameter for detect face
cascPath = "haarcascade_frontalface_lib.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)
detectface= 0
'''

#parameter for classify the gender
RESIZE_FINAL = 227
GENDER_LIST =['Male','Female']
#why the max batch size is defined as 128 based on trained model???
MAX_BATCH_SZ = 128

#device_id:choose the device you use to run the model
tf.app.flags.DEFINE_string('device_id', '/cpu:0',
                           'What processing unit to execute inference on')
# what is target used for?
tf.app.flags.DEFINE_string('target', '',
                           'CSV file containing the filename processed along with best guess and score')

tf.app.flags.DEFINE_string('checkpoint', 'checkpoint',
                          'Checkpoint basename')

tf.app.flags.DEFINE_string('requested_step', '', 'Within the model directory, a requested step to restore e.g., 9000')

'''
tf.app.flags.DEFINE_string('face_detection_type', 'cascade', 'Face detection model type (yolo_tiny|cascade)')
'''
FLAGS = tf.app.flags.FLAGS
FLAGS.model_dir = 'model_CNN'
FLAGS.class_type = 'gender'
FLAGS.model_type = 'inception'

def resolve_file(fname):
    if os.path.exists(fname): return fname
    for suffix in ('.jpg', '.png', '.JPG', '.PNG', '.jpeg'):
        cand = fname + suffix
        if os.path.exists(cand):
            return cand
    return None


def classify_many_single_crop(sess, label_list, softmax_output, coder, images, image_files, writer):
    try:
        num_batches = math.ceil(len(image_files) / MAX_BATCH_SZ)
        pg = ProgressBar(num_batches)
        for j in range(num_batches):
            
            start_offset = j * MAX_BATCH_SZ
            end_offset = min((j + 1) * MAX_BATCH_SZ, len(image_files))
            
            batch_image_files = image_files[start_offset:end_offset]
            print(start_offset, end_offset, len(batch_image_files))
            image_batch = make_multi_image_batch(batch_image_files, coder)
            batch_results = sess.run(softmax_output, feed_dict={images:image_batch.eval()})
            batch_sz = batch_results.shape[0]
            for i in range(batch_sz):
                output_i = batch_results[i]
                best_i = np.argmax(output_i)
                best_choice = (label_list[best_i], output_i[best_i])
                print('Guess @ 1 %s, prob = %.2f' % best_choice)
                if writer is not None:
                    f = batch_image_files[i]
                    writer.writerow((f, best_choice[0], '%.2f' % best_choice[1]))
            pg.update()
        pg.done()
    except Exception as e:
        print(e)
        print('Failed to run all images')


def list_images(srcfile):
    with open(srcfile, 'r') as csvfile:
        delim = ',' if srcfile.endswith('.csv') else '\t'
        reader = csv.reader(csvfile, delimiter=delim)
        if srcfile.endswith('.csv') or srcfile.endswith('.tsv'):
            print('skipping header')
            _ = next(reader)
        
        return [row[0] for row in reader]


# main

while True:
    FLAGS.filename = '/Users/bruce/Desktop/Gender_Recognition/test2.png'
    FLAGS.model_type = 'inception'
    files = []
    
    tf.reset_default_graph()

    config = tf.ConfigProto(allow_soft_placement=True)

    with tf.Session(config=config) as sess:

        label_list = GENDER_LIST
        nlabels = len(label_list)

        print('Executing on %s' % FLAGS.device_id)
        model_fn = select_model(FLAGS.model_type)

        with tf.device(FLAGS.device_id): 
            images = tf.placeholder(tf.float32, [None, RESIZE_FINAL, RESIZE_FINAL, 3])
            logits = model_fn(nlabels, images, 1, False)
            init = tf.global_variables_initializer()
            
            requested_step = FLAGS.requested_step if FLAGS.requested_step else None
        
            checkpoint_path = '%s' % (FLAGS.model_dir)
            model_checkpoint_path, global_step = get_checkpoint(checkpoint_path, requested_step, FLAGS.checkpoint)
            
            saver = tf.train.Saver()
            saver.restore(sess, model_checkpoint_path)
                
            softmax_output = tf.nn.softmax(logits)
            coder = ImageCoder()

            # Support a batch mode if no face detection model
            if len(files) == 0:
                if (os.path.isdir(FLAGS.filename)):
                    for relpath in os.listdir(FLAGS.filename):
                        abspath = os.path.join(FLAGS.filename, relpath)
                        
                        if os.path.isfile(abspath) and any([abspath.endswith('.' + ty) for ty in ('jpg', 'png', 'JPG', 'PNG', 'jpeg')]):
                            print(abspath)
                            files.append(abspath)
                else:
                    files.append(FLAGS.filename)
                    
            writer = None
            output = None

            if FLAGS.target:
                print('Creating output file %s' % FLAGS.target)
                output = open(FLAGS.target, 'w')
                writer = csv.writer(output)
                writer.writerow(('file', 'label', 'score'))
            image_files = list(filter(lambda x: x is not None, [resolve_file(f) for f in files]))
            print(image_files)

            classify_many_single_crop(sess, label_list, softmax_output, coder, images, image_files, writer)

            if output is not None:
                output.close()