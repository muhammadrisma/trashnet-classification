import os
import tensorflow as tf
from data_preparation import load_and_preprocess_data
import wandb
# from wandb.integration.keras import WandbCallback
from wandb.keras import WandbCallback
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

load_dotenv()

wandb_api_key = os.getenv("WANDB_API_KEY")
hf_token = os.getenv("HF_TOKEN")

wandb.login(key=wandb_api_key)

# Initialize the wandb run and set up config
wandb.init(
    project="trash-classification", 
    config={
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100
    }
)

# Access the config values
config = wandb.config

if not os.path.exists('evaluation'):
    os.makedirs('evaluation')

train_data, val_data = load_and_preprocess_data()
# Here i create the model from scratch, start with small layer
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(6, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

model = create_model()

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=config.epochs,
    batch_size=config.batch_size,
    callbacks=[WandbCallback(save_model=False)]  # Disable model saving, prevent conflict
)

train_acc = history.history['accuracy'][-1]
print(f"Training Accuracy: {train_acc:.2f}")
wandb.log({"Training Accuracy": train_acc})

test_data = val_data   
test_loss, test_acc = model.evaluate(test_data)
print(f"Test Accuracy: {test_acc:.2f}")
wandb.log({"Test Accuracy": test_acc})

with open('evaluation/evaluation_results.txt', 'w') as f:
    f.write(f"Training Accuracy: {train_acc:.2f}\n")
    f.write(f"Test Accuracy: {test_acc:.2f}\n")

model.save("evaluation/trashnet_model.h5")

test_labels = val_data.classes   
predictions = model.predict(test_data, verbose=1)   
predicted_labels = np.argmax(predictions, axis=1)   

cm = confusion_matrix(test_labels, predicted_labels)
print("Confusion Matrix:")
print(cm)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=val_data.class_indices.keys(), yticklabels=val_data.class_indices.keys())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('evaluation/confusion_matrix.png')

wandb.finish()
