import json
import io
import os

import tensorflow as tf

from PIL import Image

from object_detection.utils import dataset_util, label_map_util

flags = tf.app.flags
flags.DEFINE_string('annotations', '', 'Path to the vgg JSON annotation file')
flags.DEFINE_string('label_map', '', 'Path to the label_map.pbtxt file')
flags.DEFINE_string('output', '', 'Path to output TFRecord')
flags.DEFINE_string('class_field', 'Class',
                    'Name of the vgg region_attributes field where class is defined')
FLAGS = flags.FLAGS


def vgg_annotation_to_tf_example(annotation, class_to_idx):
    """
    Parameters
    ----------
    annotation: dict
        json generated by VGG Image Annotator

    Returns
    -------
    tf_example: tf.train.Example
    """
    with tf.gfile.GFile(os.path.join(os.path.dirname(FLAGS.annotations), annotation["filename"]), 'rb') as fid:
        encoded_img = fid.read()

    image = Image.open(io.BytesIO(encoded_img))

    width, height = image.size
    filename = annotation["filename"].encode("utf8")
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    regions = annotation["regions"]
    for region in regions.values():
        c = region["region_attributes"][FLAGS.class_field]
        if c == "other":
            continue
        classes_text.append(c.encode('utf8'))
        classes.append(class_to_idx[c])
        xmins.append(region["shape_attributes"]["x"] / width)
        xmaxs.append((region["shape_attributes"]["x"] +
                      region["shape_attributes"]["width"]) / width)
        ymins.append(region["shape_attributes"]["y"] / height)
        ymaxs.append((region["shape_attributes"]["y"] +
                      region["shape_attributes"]["height"]) / height)

    for i in range(len(xmins)):
        if xmins[i] < 0 or xmaxs[i] > 1 or ymins[i] < 0 or ymaxs[i] > 1:
            print(filename)
            print(xmins)
            print(xmaxs)
            print(ymins)
            print(ymaxs)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_img),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))

    return tf_example


def main(_):
    class_to_idx = label_map_util.get_label_map_dict(FLAGS.label_map)
    print(class_to_idx)
    with tf.python_io.TFRecordWriter(FLAGS.output) as writer:
        with open(FLAGS.annotations) as f:
            data = json.load(f)
        print(len(data))
        for x in data.values():
            tf_example = vgg_annotation_to_tf_example(x, class_to_idx)
            writer.write(tf_example.SerializeToString())

    print('Successfully created TFRecords at: {}'.format(FLAGS.output))


if __name__ == '__main__':
    tf.app.run()
