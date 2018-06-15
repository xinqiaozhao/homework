import tensorflow as tf
import matplotlib.pyplot as plt
import PP_inference
import PP_train
import pandas as pd
import numpy as np


TIMESTEPS = 20

INPUT_SIZE = 3

LSTM_KEEP_PROB = 1

f = open('BSE.csv')
df = pd.read_csv(f)
data = df.iloc[:,0:3].values


def get_test_data(time_step,test_begin,test_end):


    data_time = data[test_begin:test_end, 0]
    data_temp = data[test_begin:test_end, 1]
    data_power = data[test_begin:test_end, 2]
    mdt=[12.49888]
    mdt = np.array(mdt,dtype=np.float32)
    mdtp = [50.94878]
    mdtp = np.array(mdtp,dtype=np.float32)
    stddtp = [18.223719612406246]
    stddtp = np.array(stddtp,dtype=np.float32)
    mdtpower = [14916.97936]
    mdtpower = np.array(mdtpower,dtype=np.float32)
    stddtpower = [2938.1257186400294]
    stddtpower = np.array(stddtpower,dtype=np.float32)
    normalized_data_time = data_time - mdt
    normalized_data_time = np.array(normalized_data_time).reshape(-1, 1)
    print(normalized_data_time.shape)
    normalized_data_temp = (data_temp - mdtp) / stddtp
    normalized_data_temp = np.array(normalized_data_temp).reshape(-1, 1)
    normalized_data_power = (data_power - mdtpower) / stddtpower
    normalized_data_power = np.array(normalized_data_power).reshape(-1, 1)
    normalized_test_data = np.concatenate((normalized_data_time, normalized_data_temp, normalized_data_power), axis=1)
    test_x = []
    test_y = []

    data_test=data[test_begin:test_end]

    for i in range(len(normalized_test_data)-time_step):
        test_x.append([normalized_test_data[i:i + time_step, :3]])
        test_y.append([data_test[i + time_step, 1:3]])
    print("get_test_data_finished")
    test_x = np.array(test_x).reshape(-1,TIMESTEPS,INPUT_SIZE)
    print(len(test_y))
    test_y = np.array(test_y,dtype=np.float32).squeeze()
    return np.array(test_x,dtype=np.float32),np.array(test_y,dtype=np.float32)


def run_eval(test_x, test_y):
    ds = tf.data.Dataset.from_tensor_slices((test_x, test_y))
    ds = ds.batch(1)
    X,y = ds.make_one_shot_iterator().get_next()
    pred = PP_inference.lstm_model(X,LSTM_KEEP_PROB)
    predictions = []
    label = []
    realtemp=[]
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(
            PP_train.MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("checkpoint finded")
        else:
            print("No checkpoint finded")
        for i in range(len(test_y)):
            predget,yy =sess.run([pred,y])
            yy=np.array(yy).squeeze()
            print(i)
            print(yy)
            predget=np.array(predget).squeeze()
            predictions.append(predget)
            label.append(yy[1])
            realtemp.append(yy[0])
        predictions = np.array(predictions).squeeze()
        predictionstwo = np.array(label).squeeze()
        realtemp = np.array(realtemp).squeeze()
        print(predictions)
        plt.figure()
        plt.plot(predictions, label='predictions')
        plt.plot(predictionstwo, label='real')
        plt.legend()
        plt.show()


        plt.figure()
        plt.plot(realtemp,label='realtemp')
        plt.legend()
        plt.show()

def main(argv=None):
    test_x, test_y = get_test_data(TIMESTEPS, 50492,51192)
    run_eval(test_x,test_y)

if __name__ == '__main__':
    tf.app.run()