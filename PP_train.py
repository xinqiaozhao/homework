import numpy as np
import tensorflow as tf
import pandas as pd
import PP_inference


TIMESTEPS = 20
TRAINING_STEPS = 10000
BATCH_SIZE = 100

INPUT_SIZE = 3

LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.9

NUM_EXAMPLES = 50000


f=open('BSE.csv')
df=pd.read_csv(f)
data=df.iloc[:,0:3].values


MODEL_SAVE_PATH = "model_saved/"
MODEL_NAME = "model.ckpt"

LSTM_KEEP_PROB = 0.9


def variable_summaries(var, name):
    with tf.name_scope('summaries'):
        tf.summary.histogram(name, var)
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/'+name, mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev/'+name, stddev)


def get_train_data(time_step,train_begin,train_end):
    # 数据预处理过程，由于为了避免改变温度的分布规律，这里仅减了均值，温度与电力做了归一化处理
    # 其实应该用BN的！真的懒了，大家可以试试，无脑上BN肯定没错的。。。
    data_time = data[train_begin:train_end,0]
    data_temp = data[train_begin:train_end,1]
    data_power = data[train_begin:train_end,2]
    mdt = np.mean(data_time,axis=0)
    print(mdt)
    mdtp = np.mean(data_temp,axis=0)
    print(mdtp)
    stddtp = np.std(data_temp,axis=0)
    print(stddtp)
    mdtpower = np.mean(data_power,axis=0)
    print(mdtpower)
    stddtpower = np.std(data_power,axis=0)
    print(stddtpower)
    normalized_data_time = data_time-mdt
    normalized_data_time = np.array(normalized_data_time).reshape(-1,1)
    print(normalized_data_time.shape)
    normalized_data_temp = (data_temp-mdtp)/stddtp
    normalized_data_temp = np.array(normalized_data_temp).reshape(-1, 1)
    normalized_data_power = (data_power-mdtpower)/stddtpower
    normalized_data_power = np.array(normalized_data_power).reshape(-1, 1)
    normalized_train_data=np.concatenate((normalized_data_time,normalized_data_temp,normalized_data_power),axis=1)
    print(normalized_train_data.shape)
    data_labels = data[train_begin:train_end,1:3]
    train_x=[]
    train_y=[]

    for i in range(len(normalized_train_data)-time_step):

        train_x.append([normalized_train_data[i:i + time_step, :3]])
        train_y.append([data_labels[i+time_step,1]])
    print("get_train_data_finished")

    train_x=np.array(train_x).reshape(-1,TIMESTEPS,INPUT_SIZE)
    train_y=np.array(train_y).squeeze()
    train_y=np.array(train_y).reshape(-1,1)
    print(np.array(train_x, dtype=np.float32).shape)
    print(np.array(train_y, dtype=np.float32).shape)
    return np.array(train_x, dtype=np.float32), np.array(train_y, dtype=np.float32)


def train(train_x, train_y):
    global_step = tf.Variable(0, trainable=False)
    ds = tf.data.Dataset.from_tensor_slices((train_x,train_y))
    ds = ds.repeat().shuffle(100000).batch(BATCH_SIZE)
    X, y = ds.make_one_shot_iterator().get_next()
    predictions = PP_inference.lstm_model(X,LSTM_KEEP_PROB)
    print(predictions)
    loss = tf.losses.mean_squared_error(labels=y,predictions=predictions)
    tf.summary.scalar('loss',loss)
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        NUM_EXAMPLES / BATCH_SIZE,
        LEARNING_RATE_DECAY
    )
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
    print("All paras are setted")
    saver = tf.train.Saver()
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter("log/log", tf.get_default_graph())
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):

            if i % 100 == 0:

                run_options = tf.RunOptions(
                    trace_level = tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _, l, step = sess.run([merged, train_op, loss, global_step],options=run_options,run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata,'step%02d' % i)
                train_writer.add_summary(summary, i)
                print("train step is  %s loss is  %s " % (str(step), str(l)))
                saver.save(sess, "model_saved/model.ckpt")
                print("model has been saved")

            else:

                summary, _, l, step = sess.run([merged, train_op, loss, global_step])
                train_writer.add_summary(summary, i)

    train_writer.close()

def main(argv=None):
    train_x, train_y = get_train_data(TIMESTEPS, 0, NUM_EXAMPLES)
    train(train_x, train_y)

if __name__ == '__main__':
    tf.app.run()