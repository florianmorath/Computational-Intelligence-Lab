import tensorflow as tf
import os


def _get_training_data(FLAGS, mat):
    ''' Buildind the input pipeline for training and inference using TFRecords files.
    @return data only for the training
    @return data for the inference
    '''
    dataset = tf.data.Dataset.from_tensor_slices(mat)
    dataset = dataset.shuffle(buffer_size=500)
    dataset = dataset.repeat()
    dataset = dataset.batch(FLAGS.batch_size)
    dataset = dataset.prefetch(buffer_size=1)
    
    dataset2 = tf.data.Dataset.from_tensor_slices(mat)
    dataset2 = dataset2.shuffle(buffer_size=1)
    dataset2 = dataset2.repeat()
    dataset2 = dataset2.batch(1)
    dataset2 = dataset2.prefetch(buffer_size=1)
    
    return dataset, dataset2
    

def _get_test_data(FLAGS, mat):
    ''' Buildind the input pipeline for test data.'''

    dataset = tf.data.Dataset.from_tensor_slices(mat)
    dataset = dataset.shuffle(buffer_size=1)
    dataset = dataset.repeat()
    dataset = dataset.batch(1)
    dataset = dataset.prefetch(buffer_size=1)
    
    return dataset
