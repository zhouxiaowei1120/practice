import tensorflow as tf
from tensorflow.python.platform import gfile

graph = tf.get_default_graph()
graphdef = graph.as_graph_def()
graphdef.ParseFromString(gfile.FastGFile("tensorflow_inception_graph.pb","rb").read())
_ = tf.import_graph_def(graphdef, name="")

with tf.Session() as sess:
    scope = sess.graph.get_name_scope()
    print(scope)
    #options = sess.graph.get_operations()
    #print(options)

summary_write = tf.summary.FileWriter("./logdir",graph)
