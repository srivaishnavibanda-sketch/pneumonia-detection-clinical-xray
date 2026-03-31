from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout, concatenate, BatchNormalization

def build_model(img_shape=(224, 224, 3), clinical_shape=(6,)):
    # Image branch
    img_input = Input(shape=img_shape)
    x = Conv2D(32, (3, 3), activation='relu')(img_input)
    x = MaxPooling2D()(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)

    # Clinical branch
    clin_input = Input(shape=clinical_shape)
    y = Dense(32, activation='relu')(clin_input)
    y = BatchNormalization()(y)

    # Combine
    merged = concatenate([x, y])
    z = Dense(64, activation='relu')(merged)
    z = Dropout(0.5)(z)
    output = Dense(1, activation='sigmoid')(z)

    return Model(inputs=[img_input, clin_input], outputs=output)
