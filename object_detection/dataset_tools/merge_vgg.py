import json
import logging
import os

import numpy as np
import tensorflow as tf

from collections import OrderedDict
from shutil import copyfile
from PIL import Image


tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.app.flags
flags.DEFINE_string('a', '', 'Path to the the first annotations file')
flags.DEFINE_string('b', '', 'Path to the the second annotations file')
flags.DEFINE_string('out', '', 'Output folder where the merged set will be sabed')
FLAGS = flags.FLAGS


def merge_vgg(a, b, out):

    try:
        os.makedirs(os.path.dirname(out))
    except FileExistsError:
        pass

    tf.logging.info('Loading %s.', a)
    with open(a) as f:
        a_annotations = json.load(f, object_pairs_hook=OrderedDict)    
    tf.logging.info('Size of %s: %d', a, len(a_annotations))

    tf.logging.info('Loading %s.', b)
    with open(b) as f:
        b_annotations = json.load(f, object_pairs_hook=OrderedDict)        
    tf.logging.info('Size of %s: %d', b, len(b_annotations))

    tf.logging.info('Merging annotations and copying images')

    merged_annotations = OrderedDict()

    for k, v in a_annotations.items():
        merged_annotations[k] = v
        src = os.path.join(os.path.dirname(a), v["filename"])
        dst = os.path.join(os.path.dirname(out), v["filename"])
        copyfile(src, dst)

    for k, v in b_annotations.items():
        merged_annotations[k] = v
        src = os.path.join(os.path.dirname(b), v["filename"])
        dst = os.path.join(os.path.dirname(out), v["filename"])
        copyfile(src, dst)

    tf.logging.info('Size of merged annotations: %d', len(merged_annotations))

    tf.logging.info('Writing %s.', out)
    with open(out, "w") as f:
        json.dump(merged_annotations, f)

def main(_):
    merge_vgg(FLAGS.a, FLAGS.b, FLAGS.out)

if __name__ == '__main__':
    tf.app.run()
