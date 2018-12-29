import tensorflow as tf
from tensorflow.python.framework import graph_util

with tf.Session() as sess:
    with open('./expert-graph.pb', 'rb') as graph:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(graph.read())
        for i in graph_def.node:
            print(i.name)
        output = tf.import_graph_def(graph_def, return_elements=['out:0'])



sess = tf.Session()
 
    # 《《《 加载模型结构 》》》
saver = tf.train.import_meta_graph('./interFS-0.meta') 
# 只需要指定目录就可以恢复所有变量信息 
saver.restore(sess, tf.train.latest_checkpoint('./'))  

[i.name for i in sess.graph_def.node]

# The key step
constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def,output_node_names=[i.name for i in sess.graph_def.node]) # 保存图表并保存变量参数

tf.train.write_graph(constant_graph, './', 'expert-graph.pb', as_text=False)