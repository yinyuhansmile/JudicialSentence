# -*- coding:utf-8 -*-
import tensorflow as tf
class Settings(object):
    def __init__(self):
#         self.model_name = 'jieba_512_fasttext'
        self.model_name = 'new1_accu_aug512_fasttext'
#         self.model_name = 'new1_law_aug512_fasttext'
        self.jieba_len = 400
        self.thulac_len = 350
        self.fc_hidden_size = 512
        self.n_class = 202
#         self.n_class = 183
        self.summary_path = '../runs/new/summary/' + self.model_name + '/'
        self.ckpt_path = '../runs/new/ckpt/' + self.model_name + '/'


class Fasttext(object):
    def __init__(self, W_embedding, settings):
        self.model_name = settings.model_name
        self.jieba_len = settings.jieba_len
        self.thulac_len = settings.thulac_len
        self.n_class = settings.n_class
        self.fc_hidden_size = settings.fc_hidden_size
        self._global_step = tf.Variable(0, trainable=False, name='Global_Step')
        self.update_emas = list()
        # placeholders
        self._tst = tf.placeholder(tf.bool, name = 'tst')
        self._keep_prob = tf.placeholder(tf.float32, [], name="dropout_keep_prob")
        self._batch_size = tf.placeholder(tf.int32, [], name='batch_size')
        
        with tf.name_scope('Inputs'):
            self._X_inputs = tf.placeholder(tf.int32, [None, self.jieba_len], name='X_inputs')
#             X2_inputs = tf.placeholder(tf.int32, [None, self.thulac_len], name='X2_input')
            self._y_inputs = tf.placeholder(tf.float32, [None, self.n_class], name='y_input')    
        with tf.device('/cpu:0'),tf.variable_scope('embedding'):
            self.embedding = tf.get_variable(name='embedding', shape=W_embedding.shape,
                                             initializer=tf.constant_initializer(W_embedding), trainable=True)
        self.embedding_size = W_embedding.shape[1]
        
        """ X_inputs->embedding->average->fc+bn+relu+dropout"""
        with tf.variable_scope('fasttext'):
            output_jieba = self.fasttext(self._X_inputs, self.jieba_len)
          
        with tf.name_scope('out_layer'):
#             fast_output = tf.concat([output_jieba, fast_content], axis=1)
            fast_output = tf.concat([output_jieba], axis=1)
#             W_out = weight_variable([self.fc_hidden_size*2, self.n_class], name='Weight_out') 
            W_out = self.weight_variable([self.fc_hidden_size, self.n_class], name='Weight_out') 
            tf.summary.histogram('Weight_out', W_out)
            b_out = self.bias_variable([self.n_class], name='bias_out') 
            tf.summary.histogram('bias_out', b_out)
            self._y_pred = tf.nn.xw_plus_b(fast_output, W_out, b_out, name='y_pred')  #每个类别的分数 scores 
            self.predictions = tf.argmax(self._y_pred, 1, name="predictions") #罪名法条刑期分类
        
        with tf.name_scope('loss'):
            self._loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self._y_pred, labels=self._y_inputs))
            tf.summary.scalar('cost', self._loss) 
            
        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self._y_inputs, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        self.saver = tf.train.Saver(max_to_keep=2)
    
    @property
    def tst(self):
        return self._tst

    @property
    def keep_prob(self):
        return self._keep_prob

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def global_step(self):
        return self._global_step

    @property
    def X_inputs(self):
        return self._X_inputs

#     @property
#     def X2_inputs(self):
#         return self._X2_inputs

    @property
    def y_inputs(self):
        return self._y_inputs

    @property
    def y_pred(self):
        return self._y_pred

    @property
    def loss(self):
        return self._loss
        
    def weight_variable(self, shape, name):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)
    
    def bias_variable(self, shape, name):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)
    
    def batchnorm(self, Ylogits, offset, convolutional=False):
        """batchnormalization.
        Args:
            Ylogits: 1D向量或者是3D的卷积结果。
            num_updates: 迭代的global_step
            offset：表示beta，全局均值；在 RELU 激活中一般初始化为 0.1。
            scale：表示lambda，全局方差；在 sigmoid 激活中需要，这 RELU 激活中作用不大。
            m: 表示batch均值；v:表示batch方差。
            bnepsilon：一个很小的浮点数，防止除以 0.
        Returns:
            Ybn: 和 Ylogits 的维度一样，就是经过 Batch Normalization 处理的结果。
            update_moving_everages：更新mean和variance，主要是给最后的 test 使用。
        """
        exp_moving_avg = tf.train.ExponentialMovingAverage(0.999,
                                                           self._global_step)  # adding the iteration prevents from averaging across non-existing iterations
        bnepsilon = 1e-5
        if convolutional:
            mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
        else:
            mean, variance = tf.nn.moments(Ylogits, [0])
        update_moving_everages = exp_moving_avg.apply([mean, variance])
        m = tf.cond(self.tst, lambda: exp_moving_avg.average(mean), lambda: mean)
        v = tf.cond(self.tst, lambda: exp_moving_avg.average(variance), lambda: variance)
        Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
        return Ybn, update_moving_everages
    
    def fasttext(self, X_inputs, n_step):
        # X_inputs.shape = [batchsize, n_step]  ->  inputs.shape = [batchsize, n_step, embedding_size]
        inputs = tf.nn.embedding_lookup(self.embedding, X_inputs)  
        with tf.name_scope('embedding_average'):
            inputs = tf.reduce_mean(inputs, axis=1)   # [batch_size, embedding_size]
        with tf.name_scope('fc_bn_relu'):
            W_fc = self.weight_variable([self.embedding_size, self.fc_hidden_size], name='Weight_fc')
            tf.summary.histogram('W_fc', W_fc)
            h_fc = tf.matmul(inputs, W_fc, name='h_fc')
            beta_fc = tf.Variable(tf.constant(0.1, tf.float32, shape=[self.fc_hidden_size], name="beta_fc"))
            tf.summary.histogram('beta_fc', beta_fc)
            fc_bn, update_ema_fc = self.batchnorm(h_fc, beta_fc, convolutional=False)
            self.update_emas.append(update_ema_fc)
            fc_bn_relu = tf.nn.relu(fc_bn, name="relu")
            fc_bn_drop = tf.nn.dropout(fc_bn_relu, self.keep_prob)     
        return fc_bn_drop    # shape = [-1, fc_hidden_size]
