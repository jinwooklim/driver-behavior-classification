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

for epoch in range(TRAINING_EPOCHS):
    iter = 0
    while True:
        try:
            # Fetch the dataset (tf.Tensor -> numpy array)
            _img, _label = sess.run([img, labels])
            # Feed numpy array (data) To model's placeholder
            cost, _ = model.train(_img, _label)
            iter = iter + 1
            if(iter%100 == 0):
                print('Epoch:', '%02d' % (epoch + 1), 'cost =', '{:.9f}'.format(cost))
                i = 0
        except tf.errors.OutOfRangeError:
            _img_val, _label_val = sess.run([img_val, label_val])
            acc = model.get_accuracy(img_val, label_val)
            print(acc)
            break


