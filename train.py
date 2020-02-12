import tensorflow as tf
from data_loader import DataLoader
from model import ClassificationModel
from config import *

# Create tensorflow session
sess = tf.Session()
# Build model graph
model = ClassificationModel(sess, "DBC")
# Initialize the model graph
sess.run(tf.global_variables_initializer())

# Build dataset pipeline graph
train_dataset = DataLoader(BATCH_SIZE)
# Get end of dataset pipeline
img, labels = train_dataset.get_train_data()
img_val, label_val = train_dataset.get_val_data()


epoch = 0
iter = 0
while True:
    try:
        # Fetch the dataset (tf.Tensor -> numpy array)
        _img, _label = sess.run([img, labels])
        # print(_img.shape)
        # print(_img[0].shape)
        # print(_label)
        # import cv2
        # cv2.imshow("img", _img[0])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # exit()
        # Feed numpy array (data) To model's placeholder
        cost, _ = model.train(_img, _label)
        iter = iter + 1
        if(iter%100 == 0):
            acc = model.get_accuracy(_img, _label)
            print('Iter:', '%02d' % (iter), 'cost =', '{:.9f}'.format(cost), 'acc =', acc)
    except tf.errors.OutOfRangeError:
        val_acc = 0.0
        for i in range(train_dataset._val_array_len):
            _img_val, _label_val = sess.run([img_val, label_val])
            val_acc = val_acc + model.get_accuracy(_img_val, _label_val)
        print('Validation set acc :', val_acc/train_dataset._val_array_len)
        print("Validation End")
        break

