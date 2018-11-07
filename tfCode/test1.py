import tensorflow as tf

with tf.Session() as sess:
     #tf.contrib.predictor.from_saved_model(export_dir='~/Data/code/models/official/mnist/checkpoints/1539227749/',graph = graph)
     #sess.run(tf.global_variables_initializer())
    new_saver = tf.train.import_meta_graph('/home/xiaozhou/Data/code/models/official/mnist/tmp/mnist_model/model.ckpt-24000.meta')
    new_saver.restore(sess, '/home/xiaozhou/Data/code/models/official/mnist/tmp/mnist_model/model.ckpt-24000')
graph = tf.get_default_graph()
summary_write = tf.summary.FileWriter("~/Data/code/models/official/minst/checkpoints/1539227749/mnistlogdir",graph)