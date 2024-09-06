Python 3.10.4 (tags/v3.10.4:9d38120, Mar 23 2022, 23:13:41) [MSC v.1929 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image as kp_image
from tensorflow.keras import models
from tensorflow.keras.applications import vgg19

# Utility functions
def load_and_process_img(path_to_img):
    img = kp_image.load_img(path_to_img, target_size=(224, 224))
    img = kp_image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img

def deprocess_img(processed_img):
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    assert len(x.shape) == 3
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def load_img(path_to_img):
    img = load_and_process_img(path_to_img)
    return deprocess_img(img)

# Load and display images
def imshow(img, title=None):
    if len(img.shape) == 4:
        img = np.squeeze(img, axis=0)
    img = img.astype('uint8')
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.imshow(img)

# Content and Style layers from VGG19
content_layers = ['block5_conv2'] 
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 
                'block4_conv1', 'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

# Build model using VGG19
def get_model():
    vgg = vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    model_outputs = style_outputs + content_outputs
    return models.Model(vgg.input, model_outputs)

def get_content_loss(base_content, target):
    return tf.reduce_mean(tf.square(base_content - target))

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations

def get_style_loss(base_style, gram_target):
    gram_style = gram_matrix(base_style)
    return tf.reduce_mean(tf.square(gram_style - gram_target))

def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
    style_weight, content_weight = loss_weights
    model_outputs = model(init_image)
    
    style_output_features = model_outputs[:num_style_layers]
    content_output_features = model_outputs[num_style_layers:]
    
    style_score = 0
    content_score = 0

    weight_per_style_layer = 1.0 / float(num_style_layers)
    for target_style, comb_style in zip(gram_style_features, style_output_features):
        style_score += weight_per_style_layer * get_style_loss(comb_style, target_style)
    
    weight_per_content_layer = 1.0 / float(num_content_layers)
    for target_content, comb_content in zip(content_features, content_output_features):
        content_score += weight_per_content_layer * get_content_loss(comb_content, target_content)
    
    style_score *= style_weight
    content_score *= content_weight

    loss = style_score + content_score 
    return loss, style_score, content_score

@tf.function()
def compute_grads(cfg):
    with tf.GradientTape() as tape: 
        all_loss = compute_loss(**cfg)
    total_loss = all_loss[0]
    return tape.gradient(total_loss, cfg['init_image']), all_loss

# Neural Style Transfer Function
def run_style_transfer(content_path, style_path, num_iterations=1000,
                       content_weight=1e3, style_weight=1e-2): 
    model = get_model()
    for layer in model.layers:
        layer.trainable = False
    
    content_image = load_and_process_img(content_path)
    style_image = load_and_process_img(style_path)

    content_features = [model(content_image)[num_style_layers:]]
    style_features = [model(style_image)[:num_style_layers]]

    gram_style_features = [gram_matrix(style_feature) for style_feature in style_features[0]]

    init_image = tf.Variable(content_image, dtype=tf.float32)
    opt = tf.optimizers.Adam(learning_rate=5, beta_1=0.99, epsilon=1e-1)

    best_loss, best_img = float('inf'), None
    loss_weights = (style_weight, content_weight)

    cfg = {
        'model': model,
        'loss_weights': loss_weights,
        'init_image': init_image,
        'gram_style_features': gram_style_features,
        'content_features': content_features
    }

    for i in range(num_iterations):
        grads, all_loss = compute_grads(cfg)
        loss, style_score, content_score = all_loss
        opt.apply_gradients([(grads, init_image)])
        clipped_img = tf.clip_by_value(init_image, -1.0, 1.0)
        
        if loss < best_loss:
            best_loss = loss
            best_img = clipped_img.numpy()

        if i % 100 == 0:
            print('Iteration: {}'.format(i))        
    
    return best_img, best_loss

# Define the content and style images paths
content_image_path = 'path_to_your_content_image.jpg'
style_image_path = 'path_to_your_style_image.jpg'

# Perform Neural Style Transfer
best, best_loss = run_style_transfer(content_image_path, style_image_path, num_iterations=1000)

# Display the final stylized image
final_img = deprocess_img(best)
imshow(final_img)
plt.show()