import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf

classifier = load_model('./models/cat_dog.h5')

# test_image = image.load_img('/home/becode/Pictures/dog.jpg', target_size=(160,160))
# test_image = image.load_img('/home/becode/Pictures/cat.jpg', target_size=(160,160))
# test_image = image.load_img('/home/becode/Pictures/AdobeStock_190562703.jpg', target_size=(160,160))
test_image = image.load_img('/home/becode/Pictures/images.jpg', target_size=(160, 160))
# test_image = image.load_img('/home/becode/Pictures/_112669106_66030514-b1c2-4533-9230-272b8368e25f.jpg', target_size=(160,160))
# test_image = image.load_img('/home/becode/Pictures/dogcat.jpeg', target_size=(160,160))
# test_image = image.load_img('/home/becode/Pictures/catt.jpeg', target_size=(160,160))
# test_image = image.load_img('/home/becode/Pictures/dog1.jpeg', target_size=(160,160))
# test_image = image.load_img('/home/becode/Pictures/dogg.jpeg', target_size=(160,160))
# test_image = image.load_img('/home/becode/Pictures/baddog.jpeg', target_size=(160,160))


test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
predictions = classifier.predict(test_image).flatten()

# Apply a sigmoid since our model returns logits
predictions = tf.nn.sigmoid(predictions)
predictions = tf.where(predictions < 0.5, 0, 1)

if predictions == 1:
    label = 'dog'
else:
    label = 'cat'

print('Predictions:\n', predictions.numpy())
print('Labels:\n', label)