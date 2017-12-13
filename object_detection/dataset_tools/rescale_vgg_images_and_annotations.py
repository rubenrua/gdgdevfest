import json
import logging
import os

import numpy as np
import tensorflow as tf

from collections import OrderedDict
from PIL import Image


tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.app.flags
flags.DEFINE_string('annotations', None, 'Path to the annotations file')
flags.DEFINE_integer('width', 1080, 'Width of the new images')
flags.DEFINE_integer('height', 720, 'Width of the new images')
FLAGS = flags.FLAGS

def rescale_vgg_images_and_annotations(annotations, width, height):

    tf.logging.info('Loading %s.', annotations)

    with open(annotations) as f:
        old_annotations = json.load(f, object_pairs_hook=OrderedDict)

    new_annotations = OrderedDict()

    for name, annotation in old_annotations.items():

        x = annotation["filename"]

        tf.logging.info('Processing %s.', x)

        full_path = os.path.join(os.path.dirname(annotations), x)

        img = Image.open(full_path)
        img = img.convert('RGB')

        old_size = os.stat(full_path).st_size

        resized = img.resize((width, height), resample=Image.BILINEAR)

        os.remove(full_path)
        resized.save(full_path)

        annotation["size"] = os.stat(full_path).st_size

        regions = annotation["regions"]
        for region in regions.values():
            xmin = region["shape_attributes"]["x"]
            xmax = region["shape_attributes"]["x"] + region["shape_attributes"]["width"]
            ymin = region["shape_attributes"]["y"]
            ymax = region["shape_attributes"]["y"] + region["shape_attributes"]["height"]
            r_xmin = round((xmin / img.size[0]) * resized.size[0])
            r_xmax = round((xmax / img.size[0]) * resized.size[0])
            r_ymin = round((ymin / img.size[1]) * resized.size[1])
            r_ymax = round((ymax / img.size[1]) * resized.size[1])

            region["shape_attributes"]["x"] = r_xmin
            region["shape_attributes"]["width"] = r_xmax - r_xmin
            region["shape_attributes"]["y"] = r_ymin
            region["shape_attributes"]["height"] = r_ymax - r_ymin

        new_name = name.replace(str(old_size), str(annotation["size"]))
        new_annotations[new_name] = annotation

    tf.logging.info("Original first annotation")
    tf.logging.info(next(iter(old_annotations.values())))

    tf.logging.info("New first annotation")
    tf.logging.info(next(iter(new_annotations.values())))

    tf.logging.info('Overwriting %s.', annotations)

    with open(annotations, "w") as f:
        json.dump(new_annotations, f)

def main(_):
    rescale_vgg_images_and_annotations(FLAGS.annotations, FLAGS.width, FLAGS.height)

if __name__ == '__main__':
    tf.app.run()
