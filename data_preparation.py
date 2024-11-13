from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_and_preprocess_data(dir_path="garbage_dataset", img_size=(224, 224), batch_size=32):
    '''
    Preprocess the data and return train and test generators
    80 % of the data will be used for training and 20 % for testing
    I add data augmentation, so that the model can generalize better
    '''
    # Define ImageDataGenerator for training with data augmentation
    train_datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.2,  
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1
    )

    # Define ImageDataGenerator for testing/validation without augmentation
    test_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2  
    )

    # Create the training data generator
    train_generator = train_datagen.flow_from_directory(
        directory=dir_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training' 
    )

    # Create the test data generator
    test_generator = test_datagen.flow_from_directory(
        directory=dir_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'  
    )

    return train_generator, test_generator
