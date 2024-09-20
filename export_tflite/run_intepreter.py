import tflite_runtime.interpreter as tflite
import numpy as np
from PIL import Image

# Load the TFLite model and allocate tensors.
interpreter = tflite.Interpreter(model_path='modelthirtysix.tflite')
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load and preprocess the image using PIL.
image = Image.open('cat.jpeg').convert('RGB')  # Ensure image is in RGB format
image = image.resize((256, 256))  # Resize the image to match the model input
image_data = np.asarray(image).astype(np.float32)  # Convert image to numpy array
image_data = image_data / 255

# If the model expects shape (1, 3, 224, 224), transpose and reshape the image data.
image_data = np.transpose(image_data, (2, 0, 1))  # Convert from (224, 224, 3) to (3, 224, 224)
image_data = np.expand_dims(image_data, axis=0)  # Add batch dimension: (1, 3, 224, 224)

# Ensure the data type matches the input type of the model.
input_dtype = input_details[0]['dtype']
image_data = image_data.astype(input_dtype)

# Set the tensor.
interpreter.set_tensor(input_details[0]['index'], image_data)

# Run the model.
interpreter.invoke()

# Get the output.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(f"Output data shape: {output_data.shape}")
print('Output data: ', output_data[0, 0:10])