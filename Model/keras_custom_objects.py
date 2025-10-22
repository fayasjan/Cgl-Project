import tensorflow as tf
from tensorflow.keras import backend as K

def contrastive_loss(y_true, y_pred, margin=1.0):
    """
    Contrastive loss function.
    y_true = 1 for positive pairs, 0 for negative pairs.
    y_pred = distance between embeddings.
    """
    # Ensure margin matches the value used during training (defaulted to 1.0 here)
    y_true = K.cast(y_true, 'float32')
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

def euclidean_distance(vectors):
    """Calculates the euclidean distance between two embedding vectors."""
    x, y = vectors
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    # Add epsilon for numerical stability before sqrt
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

# How your friend uses these when loading the model:
# custom_objects = {
#     'contrastive_loss': contrastive_loss,
#     'euclidean_distance': euclidean_distance
# }
# siamese_model = tf.keras.models.load_model('your_model.keras', custom_objects=custom_objects)