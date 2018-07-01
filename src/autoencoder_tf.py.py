
# coding: utf-8

# # Autoencoder tensorflow

# In[1]:


import numpy as np
import tensorflow as tf
from matrix_helpers import load_data, write_submission_file
from sklearn.model_selection import train_test_split

from DAE import DAE
from dataset import _get_training_data, _get_test_data


tf.app.flags.DEFINE_string('f', '', 'kernel')
# See: https://github.com/tensorflow/tensorflow/issues/17702
tf.app.flags.DEFINE_integer('num_epoch', 1000,
                            'Number of training epochs.')

tf.app.flags.DEFINE_integer('batch_size', 16,
                            'Size of the training batch.')

tf.app.flags.DEFINE_float('learning_rate',0.0005,
                          'Learning_Rate')

tf.app.flags.DEFINE_boolean('l2_reg', False,
                            'L2 regularization.'
                            )
tf.app.flags.DEFINE_float('lambda_',0.01,
                          'Wight decay factor.')

tf.app.flags.DEFINE_integer('num_v', 1000,
                            'Number of visible neurons (Number of movies the users rated.)')

tf.app.flags.DEFINE_integer('num_h', 128,
                            'Number of hidden neurons.)')

tf.app.flags.DEFINE_integer('num_samples', 10000,
                            'Number of training samples (Number of users, who gave a rating).')

tf.app.flags.DEFINE_boolean('re_feed', True,
                            'Re-feed the results of one pass back into the network.')

FLAGS = tf.app.flags.FLAGS

'''Building the graph, opening of a session and starting the training od the neural network.'''

num_batches=int(FLAGS.num_samples/FLAGS.batch_size)

default_graph = tf.Graph().as_default()

data_mat = load_data()
train_mat, test_mat = train_test_split(data_mat, train_size=0.8)

train_data, train_data_infer=_get_training_data(FLAGS, train_mat)
test_data=_get_test_data(FLAGS, test_mat)
pred_data = _get_test_data(FLAGS, data_mat)

iter_train = train_data.make_initializable_iterator()
iter_train_infer=train_data_infer.make_initializable_iterator()
iter_test = test_data.make_initializable_iterator()
iter_pred = pred_data.make_initializable_iterator()

x_train = iter_train.get_next()
x_train_infer =iter_train_infer.get_next()
x_test = iter_test.get_next()
x_pred = iter_pred.get_next()

model = DAE(FLAGS)

train_op, train_loss_op=model._optimizer(x_train)
pred_op, test_loss_op=model._validation_loss(x_train_infer, x_test)
infer_op = model._inference(x_pred, train_mode=False)

sess = tf.InteractiveSession()

sess.run(tf.global_variables_initializer())
train_loss=0
test_loss=0

for epoch in range(FLAGS.num_epoch):

    sess.run(iter_train.initializer)

    for batch_nr in range(num_batches):

        _, loss_=sess.run((train_op, train_loss_op))
        train_loss+=loss_

    sess.run(iter_train_infer.initializer)
    sess.run(iter_test.initializer)

    for i in range(FLAGS.num_samples):
        pred, loss_=sess.run((pred_op, test_loss_op))
        test_loss+=loss_

    print('epoch_nr: %i, train_loss: %.3f, test_loss: %.3f'%(epoch,(train_loss/num_batches),(test_loss/FLAGS.num_samples)))
    train_loss=0
    test_loss=0


# In[ ]:


# Build output matrix (aka run one time again)
sess.run(iter_pred.initializer)
preds = []
for batch_nr in range(FLAGS.num_samples):
    pred = sess.run((infer_op))
    preds.append(pred)


# In[ ]:


len(preds)


# In[ ]:


X_pred = np.concatenate(tuple(preds), axis=0)
X_pred.shape


# In[ ]:


write_submission_file(X_pred, 'submission_autoencoder_0.csv')

