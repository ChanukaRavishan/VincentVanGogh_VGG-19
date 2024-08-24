import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.vgg19 import preprocess_input

def process_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def deprocess_image(processed_img):
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, axis=0)
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]  # Convert from BGR to RGB
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result / num_locations

def get_model():
    vgg = VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    s_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    c_layers = ['block5_conv2']
    style_layers = [vgg.get_layer(name).output for name in s_layers]
    content_layers = [vgg.get_layer(name).output for name in c_layers]
    model_layers = style_layers + content_layers
    model = tf.keras.models.Model(vgg.input, model_layers)
    return model

def get_content_features(image, model):
    content_outputs = model(image)
    return content_outputs[-1]

def get_style_features(image, model):
    style_outputs = model(image)
    return style_outputs[:-1]

def compute_grad(content_features, style_features, gram_style, image, content_weight, style_weight, model):
    with tf.GradientTape() as tape:
        tape.watch(image)
        model_outputs = model(image)
        style_output_features = model_outputs[:-1]
        content_output_features = model_outputs[-1]
        style_loss = tf.add_n([tf.reduce_mean((gram_matrix(style_output) - gram_target)**2)
                              for style_output, gram_target in zip(style_output_features, gram_style)])
        style_loss *= style_weight / len(style_output_features)
        content_loss = tf.reduce_mean((content_output_features - content_features)**2)
        content_loss *= content_weight
        loss = style_loss + content_loss
    grad = tape.gradient(loss, image)
    return grad, loss

