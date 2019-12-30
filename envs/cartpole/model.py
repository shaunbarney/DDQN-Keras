import tensorflow as tf

def build_model(lr, n_actions, input_shape):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(
        24, input_shape=(*input_shape,), activation='relu'))
    model.add(tf.keras.layers.Dense(24, activation='relu'))
    model.add(tf.keras.layers.Dense(n_actions, activation='linear'))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss='mse')
    return model