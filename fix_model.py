import tensorflow as tf

model = tf.keras.models.load_model("crop_model.keras", compile=False)

model.save("final_model.keras")
print("Saved successfully")