import tensorflow as tf
import model_helper


class DAE:
    
    def __init__(self, FLAGS):
        ''' Implementation of deep autoencoder class.'''
        
        self.FLAGS=FLAGS
        self.weight_initializer=model_helper._get_weight_initializer()
        self.bias_initializer=model_helper._get_bias_initializer()
        self.init_parameters()
        

    def init_parameters(self):
        '''Initialize networks weights and biases.'''
        # Encoder weights
        hidden1 = 512
        hidden2 = 512
        hidden3 = 1024
        # Decoder weights are just the transpose
        
        with tf.name_scope('weights'):
            self.W_1=tf.get_variable(name='encoder_weight_1', shape=(self.FLAGS.num_v, hidden1),
                                     initializer=self.weight_initializer)
            self.W_2=tf.get_variable(name='encoder_weight_2', shape=(hidden1, hidden2),
                                     initializer=self.weight_initializer)
            self.W_3=tf.get_variable(name='encoder_weight_3', shape=(hidden2, hidden3),
                                     initializer=self.weight_initializer)
            self.W_4 = tf.transpose(self.W_3)
            self.W_5 = tf.transpose(self.W_2)
            self.W_6 = tf.transpose(self.W_1)

        
        with tf.name_scope('biases'):
            self.b1=tf.get_variable(name='bias_1', shape=(hidden1),
                                    initializer=self.bias_initializer)
            self.b2=tf.get_variable(name='bias_2', shape=(hidden2),
                                    initializer=self.bias_initializer)
            self.b3=tf.get_variable(name='bias_3', shape=(hidden3),
                                    initializer=self.bias_initializer)
            self.b4=tf.get_variable(name='bias_4', shape=(hidden2),
                                    initializer=self.bias_initializer)
            self.b5=tf.get_variable(name='bias_5', shape=(hidden1),
                                    initializer=self.bias_initializer)
            self.b6=tf.get_variable(name='bias_6', shape=(self.FLAGS.num_v),
                                    initializer=self.bias_initializer)


    def _inference(self, x, train_mode=True):
        ''' Making one forward pass. Predicting the networks outputs.
        @param x: input ratings
        
        @return : networks predictions
        '''
        activation_l = tf.nn.sigmoid
        bounded = activation_l == tf.nn.sigmoid or activation_l == tf.nn.tanh

        with tf.name_scope('inference'):
            d1 = activation_l(tf.nn.bias_add(tf.matmul(x, self.W_1),self.b1))
            d2 = activation_l(tf.nn.bias_add(tf.matmul(d1, self.W_2),self.b2))
            d3 = activation_l(tf.nn.bias_add(tf.matmul(d2, self.W_3),self.b3))
            d_ = tf.layers.dropout(inputs=d3, rate=0.8, training=train_mode)
            d4 = activation_l(tf.nn.bias_add(tf.matmul(d_, self.W_4),self.b4))
            d5 = activation_l(tf.nn.bias_add(tf.matmul(d4, self.W_5),self.b5))
            out = tf.nn.bias_add(tf.matmul(d5, self.W_6),self.b6)
            if not bounded:
                out = activation_l(out)

            #a1=tf.nn.selu(tf.nn.bias_add(tf.matmul(x, self.W_1),self.b1))
            #a2=tf.nn.selu(tf.nn.bias_add(tf.matmul(a1, self.W_2),self.b2))
            #a3=tf.nn.selu(tf.nn.bias_add(tf.matmul(a2, self.W_3),self.b3))
            #a4=tf.matmul(a3, self.W_4)


        return out
    
    def _compute_loss(self, predictions, labels,num_labels):
        ''' Computing the Mean Squared Error loss between the input and output of the network.
    		
    	  @param predictions: predictions of the stacked autoencoder
    	  @param labels: input values of the stacked autoencoder which serve as labels at the same time
    	  @param num_labels: number of labels !=0 in the data set to compute the mean
    		
    	  @return mean squared error loss tf-operation
    	  '''
            
        with tf.name_scope('loss'):
            
            loss_op=tf.div(tf.reduce_sum(tf.square(tf.subtract(predictions,labels))),num_labels)
            return loss_op
    	  
    def _mask_outputs(self, x, outputs):
        mask=tf.where(tf.equal(x,0.0), tf.zeros_like(x), x) # indices of 0 values in the training set
        num_train_labels=tf.cast(tf.count_nonzero(mask),dtype=tf.float32) # number of non zero values in the training set
        bool_mask=tf.cast(mask,dtype=tf.bool) # boolean mask
        masked_outputs=tf.where(bool_mask, outputs, tf.zeros_like(outputs)) # set the output values to zero if corresponding input values are zero
        return masked_outputs, num_train_labels


    def _optimizer(self, x):
        '''Optimization of the network parameter through stochastic gradient descent.
            
            @param x: input values for the stacked autoencoder.
            
            @return: tensorflow training operation
            @return: ROOT!! mean squared error
        '''
        
        outputs=self._inference(x)
        masked_outputs, num_train_labels = self._mask_outputs(x, outputs)
        masked_MSE_loss=self._compute_loss(masked_outputs, x, num_train_labels)

        if self.FLAGS.l2_reg==True:
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
            masked_MSE_loss = masked_MSE_loss + self.FLAGS.lambda_ * l2_loss

        optimizer = tf.train.AdamOptimizer(self.FLAGS.learning_rate)
        train_op = optimizer.minimize(masked_MSE_loss)

        if self.FLAGS.re_feed:
            outputs_refeed = self._inference(outputs)
            MSE_loss = tf.losses.mean_squared_error(outputs, outputs_refeed)
            optimizer.minimize(MSE_loss)

        RMSE_loss = tf.sqrt(masked_MSE_loss)

        return train_op, RMSE_loss
    
    def _validation_loss(self, x_train, x_test):
        
        ''' Computing the loss during the validation time.
    		
    	  @param x_train: training data samples
    	  @param x_test: test data samples
    		
    	  @return networks predictions
    	  @return root mean squared error loss between the predicted and actual ratings
    	  '''
        
        outputs=self._inference(x_train) # use training sample to make prediction
        mask=tf.where(tf.equal(x_test,0.0), tf.zeros_like(x_test), x_test) # identify the zero values in the test ste
        num_test_labels=tf.cast(tf.count_nonzero(mask),dtype=tf.float32) # count the number of non zero values
        bool_mask=tf.cast(mask,dtype=tf.bool) 
        outputs=tf.where(bool_mask, outputs, tf.zeros_like(outputs))
    
        MSE_loss=self._compute_loss(outputs, x_test, num_test_labels)
        RMSE_loss=tf.sqrt(MSE_loss)
            
        return outputs, RMSE_loss
    
   
    
    
    
    
    