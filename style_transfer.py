import tensorflow as tf
from tensorflow.keras.preprocessing import image as kp_image
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
import numpy as np
import argparse
from utils import process_image, deprocess_image, compute_grad, get_model, get_content_features, get_style_features, gram_matrix

def vvg(content_path, style_path, output_path, steps=1000, content_weight=1e3, style_weight=1e-2):
    model = get_model()
    for layer in model.layers:
        layer.trainable = False

    content = process_image(content_path)
    style = process_image(style_path)

    image = tf.Variable(process_image(content_path))

    content_features = get_content_features(content, model)
    style_features = get_style_features(style, model)
    gram_style = [gram_matrix(style_feature) for style_feature in style_features]

    optimizer = tf.keras.optimizers.Adam(learning_rate=5)

    best_loss = float('inf')
    best_image = None

    for i in range(steps):
        grad, loss = compute_grad(content_features, style_features, gram_style, image, content_weight, style_weight, model)
        optimizer.apply_gradients([(grad, image)])

        if loss < best_loss:
            best_loss = loss
            best_image = deprocess_image(image.numpy())

        if i % 100 == 0:
            print(f"Step {i}/{steps} - Loss: {loss}")

    if best_image is not None:
        tf.keras.preprocessing.image.save_img(output_path, best_image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Neural Style Transfer')
    parser.add_argument('--content_path', type=str, required=True, help='Path to content image')
    parser.add_argument('--style_path', type=str, required=True, help='Path to style image')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save output image')
    args = parser.parse_args()

    vvg(args.content_path, args.style_path, args.output_path)
