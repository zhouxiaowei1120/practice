from vgg import vgg_19
import tensorflow as tf

vgg19 = vgg_19(tf.convert_to_tensor(tf.truncated_normal([10,224,224,3])))
vgg19_1 = vgg_19(tf.convert_to_tensor(tf.truncated_normal([10,224,224,3])),reuse=True)

all_var = tf.trainable_variables()
input()