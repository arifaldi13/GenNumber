import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# --- 1. Define Model Architecture ---
# This MUST be the same Keras architecture as in the training script.

LATENT_DIM = 100
NUM_CLASSES = 10

def build_generator():
    noise_input = layers.Input(shape=(LATENT_DIM,))
    label_input = layers.Input(shape=(1,), dtype='int32')
    
    label_embedding = layers.Embedding(NUM_CLASSES, LATENT_DIM)(label_input)
    label_embedding = layers.Flatten()(label_embedding)
    
    model_input = layers.Concatenate()([noise_input, label_embedding])
    
    x = layers.Dense(256, use_bias=False)(model_input)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dense(512, use_bias=False)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dense(1024, use_bias=False)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dense(28 * 28 * 1, activation='tanh')(x)
    output = layers.Reshape((28, 28, 1))(x)
    
    return keras.Model([noise_input, label_input], output, name="generator")

# --- 2. Load the Trained Model ---
@st.cache_resource
def load_model():
    # Build a new model with the same architecture
    model = build_generator()
    # Load the saved weights
    model.load_weights('cgan_generator.h5')
    return model

generator = load_model()

# --- 3. Create the Streamlit Web App UI ---
st.set_page_config(layout="wide")
st.title("Handwritten Digit Image Generator (TensorFlow/Keras)")
st.write("Generate synthetic MNIST-like images using a Conditional GAN model trained from scratch.")

# --- UI Components ---
st.sidebar.header("Controls")
digit_to_generate = st.sidebar.selectbox("Choose a digit to generate (0-9):", list(range(10)))

if st.sidebar.button("Generate Images"):
    st.subheader(f"Generated images of digit: {digit_to_generate}")

    # Generate 5 images
    num_images = 5
    
    # Prepare noise and labels using TensorFlow
    noise = tf.random.normal([num_images, LATENT_DIM])
    labels = tf.constant([digit_to_generate] * num_images, dtype=tf.int32)
    
    # Generate images
    generated_imgs = generator([noise, labels], training=False)
    
    # Post-process for display: un-normalize from [-1, 1] to [0, 1]
    generated_imgs = (generated_imgs * 127.5 + 127.5) / 255.0

    # Use columns to display images side-by-side
    cols = st.columns(num_images)
    for i in range(num_images):
        with cols[i]:
            # Convert tensor to numpy for display
            img_np = generated_imgs[i].numpy()
            st.image(img_np, caption=f"Sample {i+1}", use_column_width=True)
else:
    st.info("Select a digit and click 'Generate Images' in the sidebar.")