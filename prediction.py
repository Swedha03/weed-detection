import tensorflow as tf
import numpy as np
import cv2
from preprocessing.image_preprocessing import preprocess_image

class WeedDetectionModel:
    def __init__(self, model_path):
        # Load the TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, image_array):
        # Get the expected input shape of the model
        input_shape = self.input_details[0]['shape']  # (1, height, width, channels)
        _, height, width, channels = input_shape  

        # Properly resize the image using OpenCV
        image_array = cv2.resize(image_array, (width, height))  # Resize to match model input shape

        # Convert image to float32 (ensure correct dtype)
        image_array = np.array(image_array, dtype=np.float32)  

        # Normalize (if required, check your model training process)
        image_array /= 255.0  # Convert to range [0,1] (common in deep learning models)

        # Expand dimensions if the model expects a batch dimension
        if len(image_array.shape) == 3:  
            image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension if needed

        print("✅ Shape of image_array before setting tensor:", image_array.shape)

        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], image_array)

        # Run inference
        self.interpreter.invoke()

        # Get output tensor
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output_data


if __name__ == "__main__":
    # Load and preprocess the image
    image_path = "input_images/image.png"  # Make sure the path is correct
    preprocessed_image = preprocess_image(image_path)

    # Load the TensorFlow Lite model
    model = WeedDetectionModel("models/1.tflite")

    # Make a prediction
    prediction = model.predict(preprocessed_image)

    # Output prediction results
    print("📢 Model Prediction:", prediction)
