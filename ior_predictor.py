import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
# We have imported all dependencied
df = pd.read_csv('data.csv') # read data set using pandas
print(df.info()) # Overview of dataset
df = df.drop(['Date'],axis=1) # Drop Date feature
df = df.dropna(inplace=False)  # Remove all nan entries.

df = df.drop(['Adj Close','Volume'],axis=1) # Drop Adj close and volume feature
df_train = df[:1059]    # 60% training data and 40% testing data
df_test = df[1059:]
scaler = MinMaxScaler() # For normalizing dataset
# We want to predict Close value of stock 
X_train = scaler.fit_transform(df_train.drop(['Close'],axis=1).as_matrix())
y_train = scaler.fit_transform(df_train['Close'].as_matrix())
# y is output and x is features.
X_test = scaler.fit_transform(df_test.drop(['Close'],axis=1).as_matrix())
y_test = scaler.fit_transform(df_test['Close'].as_matrix())




def neural_net_model(X_data,input_dim):
    W_1 = tf.Variable(tf.random_uniform([input_dim,10]))
    b_1 = tf.Variable(tf.zeros([10]))
    layer_1 = tf.add(tf.matmul(X_data,W_1), b_1)
    layer_1 = tf.nn.relu(layer_1)

    # layer 1 multiplying and adding bias then activation function
    W_2 = tf.Variable(tf.random_uniform([10,10]))
    b_2 = tf.Variable(tf.zeros([10]))
    layer_2 = tf.add(tf.matmul(layer_1,W_2), b_2)
    layer_2 = tf.nn.relu(layer_2)
    # layer 2 multiplying and adding bias then activation function
    W_O = tf.Variable(tf.random_uniform([10,1]))
    b_O = tf.Variable(tf.zeros([1]))
    output = tf.add(tf.matmul(layer_2,W_O), b_O)
    # O/p layer multiplying and adding bias then activation function
    # notice output layer has one node only since performing #regression
    return output
"""
neural_net_model is function applying 2 hidden layer feed forward neural net.
Weights and biases are abberviated as W_1,W_2 and b_1, b_2 
These are variables with will be updated during training.
"""
xs = tf.placeholder("float")
ys = tf.placeholder("float")
output = neural_net_model(xs,3)
cost = tf.reduce_mean(tf.square(output-ys))
# our mean squared error cost function
train = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
# Gradinent Descent optimiztion just discussed above for updating weights and biases

with tf.Session() as sess:
    # Initiate session and initialize all vaiables
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    #saver.restore(sess,'yahoo_dataset.ckpt')
    for i in range(100):
        for j in range(X_train.shape[0]):
            sess.run([cost,train],feed_dict=    {xs:X_train[j,:].reshape(1,3), ys:y_train[j]})
            # Run cost and train with each sample
        c_t.append(sess.run(cost, feed_dict={xs:X_train,ys:y_train}))
        c_test.append(sess.run(cost, feed_dict={xs:X_test,ys:y_test}))
        print('Epoch :',i,'Cost :',c_t[i])
    pred = sess.run(output, feed_dict={xs:X_test})
    # predict output of test data after training
    print('Cost :',sess.run(cost, feed_dict={xs:X_test,ys:y_test}))
    y_test = denormalize(df_test,y_test)
    pred = denormalize(df_test,pred)
    #Denormalize data     
    plt.plot(range(y_test.shape[0]),y_test,label="Original Data")
    plt.plot(range(y_test.shape[0]),pred,label="Predicted Data")
    plt.legend(loc='best')
    plt.ylabel('Stock Value')
    plt.xlabel('Days')
    plt.title('Stock Market Nifty')
    plt.show()
    if input('Save model ? [Y/N]') == 'Y':
        saver.save(sess,'yahoo_dataset.ckpt')
        print('Model Saved')