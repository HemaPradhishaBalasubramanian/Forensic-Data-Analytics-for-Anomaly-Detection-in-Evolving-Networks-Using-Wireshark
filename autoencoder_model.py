from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_autoencoder(input_dim):
    """
    Build and compile an Autoencoder model for anomaly detection.
    Args:
        input_dim (int): Number of input features
    Returns:
        model (keras.Model): Compiled autoencoder model
    """

    # ----- Encoder -----
    input_layer = Input(shape=(input_dim,))
    x = Dense(128, activation='relu')(input_layer)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    encoded = Dense(32, activation='relu')(x)

    # ----- Decoder -----
    x = Dense(64, activation='relu')(encoded)
    x = Dense(128, activation='relu')(x)
    output_layer = Dense(input_dim, activation='sigmoid')(x)

    # ----- Model -----
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    print(model.summary())
    return model

# Example test
if __name__ == "__main__":
    test_model = build_autoencoder(100)  # 100 features test run

