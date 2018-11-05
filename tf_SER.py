# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 10:19:11 2018

@author: VLSI-AIR
"""

import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import matplotlib.pyplot as plt

from utilities import get_data, class_labels
from keras.utils import np_utils
import pandas as pd



HIDDEN_UNITS=128
BATCH_SIZE=32
LEARNING_RATE=0.002
HIDDEN_UNITS_FC1=64
HIDDEN_UNITS_FC2=16
HIDDEN_UNITS_FC3=len(class_labels)
EPOCH=80

def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]
        
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


#---------Get Data--------------------------------------------------------------------------------------------#
global x_train, y_train, x_test, y_test, y_test_categories

x_train, x_test, y_train, y_test = get_data(flatten=False)


y_test_categories = y_test
y_train1 = np_utils.to_categorical(y_train)
y_test1 = np_utils.to_categorical(y_test)

print('Train sample:',x_train[0].shape)#Train sample
print('Test sample:',x_test.shape[1])#Test sample
print('Timestep:',x_train.shape[1])#TIME STEP
print('Inputsize:',x_train.shape[2])#intput siz for each time step


#tf.reset_default_graph()
#--------------------------------------Define Graph---------------------------------------------------#
graph=tf.Graph()
with graph.as_default():

    #------------------------------------construct LSTM------------------------------------------#
    #place holder
    X_p=tf.placeholder(dtype=tf.float32,shape=(None,x_train[0].shape[0],x_train[0].shape[1]),name="input_placeholder")
    y_p=tf.placeholder(dtype=tf.float32,shape=(None,len(class_labels)),name="pred_placeholder")
    batch_size = tf.placeholder(tf.int32, [], name='batch_size')
    keep_prob = tf.placeholder(tf.float32)

    #lstm instance
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=HIDDEN_UNITS)    
    
    cell_dr = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,  output_keep_prob=keep_prob)

    #initialize to zero
    init_state=lstm_cell.zero_state(batch_size=batch_size,dtype=tf.float32)

    #dynamic rnn
    lstm_1,states=tf.nn.dynamic_rnn(cell=cell_dr,inputs=X_p,initial_state=init_state,dtype=tf.float32)
    
	#FC layer1
    fc_layer1 = tf.layers.dense(inputs=lstm_1[:,-1,:], units=64, activation=tf.nn.relu)

	#FC layer2
    fc_layer2 = tf.layers.dense(inputs=fc_layer1, units=16, activation=tf.nn.tanh)

	#FC layer3
    outputs = tf.layers.dense(inputs=fc_layer2, units=HIDDEN_UNITS_FC3, activation=tf.nn.softmax)

    h=outputs
    print('H shape:',h.shape)
    #--------------------------------------------------------------------------------------------#

    #---------------------------------define loss and optimizer----------------------------------#
    
    
    #loss=tf.losses.softmax_cross_entropy(onehot_labels=y_p,logits=h)
    loss=tf.reduce_mean(tf.keras.backend.binary_crossentropy(y_p,h))
    #print(loss.shape)
    optimizer=tf.train.AdamOptimizer(LEARNING_RATE,epsilon=1e-07).minimize(loss=loss)

    init=tf.global_variables_initializer()

	#accuracy
    correct_pred = tf.equal(tf.argmax(h, 1), tf.argmax(y_p, 1))
    acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


    saver = tf.train.Saver()
    
#-------------------------------------------Define Session---------------------------------------#
#with tf.Session(graph=graph) as sess:
    #
with tf.variable_scope('rnn') as vs:
    sess = tf.Session(graph=graph)

    sess.run(init)
#    print (sess.run(tf.global_variables()))
    accuracy_best = 0
    for epoch in range(1,EPOCH+1):
        results = np.zeros(shape=(x_test.shape[0], len(class_labels)))
        train_losses=[]
        test_losses=[]
        print("epoch:",epoch)
        
        #training
        for x_train0,y_train0  in minibatches(inputs=x_train, targets=y_train1, batch_size=BATCH_SIZE, shuffle=True):
            _,train_loss=sess.run(
                    fetches=(optimizer,loss),
                    feed_dict={
                            X_p:x_train0,
                            y_p:y_train0,
                            batch_size:x_train0.shape[0],
                            keep_prob:0.5
                        }
            )
            train_losses.append(train_loss)
        
        
        print("############################################################")
        print("average training loss:", sum(train_losses) / len(train_losses))

        #--------------------------------validation-----------------------------------------------#

        val_acc=sess.run(acc, feed_dict={X_p:x_test, y_p:y_test1, batch_size:x_test.shape[0], keep_prob:1})
             
        
        print (" Accuracy = ", val_acc)
        if(val_acc>accuracy_best):
            accuracy_best = val_acc
            print(" Best Accuracy update : ", accuracy_best)
            save_path = saver.save(sess, "ckpt/SER1101_twn.ckpt")

    print(" Best Accuracy  : ", accuracy_best)

	
	
	#Best
    saver.restore(sess, "ckpt/SER1101_twn.ckpt")
    
    results = sess.run(fetches=(h), feed_dict={X_p:x_test, y_p:y_test1, batch_size:x_test.shape[0], keep_prob:1})
    
    new_x = [ np.argmax(item) for item in results ]
    print(pd.crosstab(y_test_categories,np.array(new_x[0:x_test.shape[0]]),rownames=['label'],colnames=['predict']))
    plt.imshow(pd.crosstab(y_test_categories,np.array(new_x[0:x_test.shape[0]]),rownames=['label'],colnames=['predict']))
    
    #print("val. Accuracy:", sess.run(acc, feed_dict={X_p:x_test, y_p:y_test1, batch_size:x_test.shape[0], keep_prob:1 }))
    
 

