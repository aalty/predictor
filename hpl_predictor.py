import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
# We have imported all dependencied
df = pd.read_csv('hpl.csv', names=["Arg1","Arg2","Time","Node","Avg.CPU","Max.CPU","Avg.Mem","Max.Mem"]) # read data set using pandas
#print(df.info()) # Overview of dataset
df = df.drop(["Node","Avg.CPU","Max.CPU","Avg.Mem","Max.Mem"],axis=1) # Drop Date feature
df = df.dropna(inplace=False)  # Remove all nan entries.

#with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
#    print(df)

#df = df.drop(['Adj Close','Volume'],axis=1) # Drop Adj close and volume feature
row_num = int(round(df.shape[0] * 0.8))
print("row_num ", row_num)
df_train = df[:row_num]    # 60% training data and 40% testing data
df_test = df[row_num:]
scaler = MinMaxScaler() # For normalizing dataset
# We want to predict Close value of stock 
X_train = scaler.fit_transform(df_train.drop(["Time"],axis=1).values)
df_train_time = df_train["Time"]
y_train = scaler.fit_transform(df_train_time[:,None])
# y is output and x is features.
X_test = scaler.fit_transform(df_test.drop(["Time"],axis=1).values)
df_test_time = df_test["Time"]
y_test = scaler.fit_transform(df_test_time[:,None])
#print(X_train.shape)
#print(X_train[0].shape)


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

xs = tf.placeholder("float")
ys = tf.placeholder("float")
output = neural_net_model(xs,2)
cost = tf.reduce_mean(tf.square(output-ys))
# our mean squared error cost function
train = tf.train.AdamOptimizer().minimize(cost)
# Gradinent Descent optimiztion just discussed above for updating weights and biases

c_t = []
def denormalize(df,norm_data):
    df = df['Time'].values.reshape(-1,1)
    norm_data = norm_data.reshape(-1,1)
    scl = MinMaxScaler()
    a = scl.fit_transform(df)
    new = scl.inverse_transform(norm_data)


with tf.Session() as sess:
    # Initiate session and initialize all vaiables
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    #saver.restore(sess,'yahoo_dataset.ckpt')

    for i in range(3000):
        #for j in range(X_train.shape[0]):
        sess.run([cost,train],feed_dict={xs:X_train, ys:y_train})
            # Run cost and train with each sample
        c_t.append(sess.run(cost, feed_dict={xs:X_train,ys:y_train}))
        print('Epoch :',i,'Cost :',c_t[i])
    pred = sess.run(output, feed_dict={xs:X_test})
    # predict output of test data after training

    out_test, cost_test = sess.run([output, cost], feed_dict={xs:X_test[0].reshape(1,2),ys:y_test[0]})
    #print('Cost :',sess.run(cost, feed_dict={xs:X_test[0].reshape(1,2),ys:y_test[0]}))
    y_test_ori = scaler.inverse_transform(y_test[3].reshape(-1,1))
    out_test_ori = scaler.inverse_transform(out_test.reshape(-1,1))
    print('X: ', X_test[3], 'y: ', y_test_ori, 'Output: ', out_test_ori, 'Cost: ', cost_test)
    