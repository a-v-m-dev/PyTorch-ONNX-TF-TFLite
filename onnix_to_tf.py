import onnx
import tensorflow as tf
from onnx_tf.backend import prepare

# Load the ONNX model
onnx_model = onnx.load('best.onnx')

# Prepare the ONNX model for TensorFlow
tf_rep = prepare(onnx_model)

# Convert the ONNX model to a TensorFlow SavedModel
tf_saved_model_path = 'exported'
tf_rep.export_graph(tf_saved_model_path)


# OR run the code below using the command line
# onnx-tf convert -i /path/to/input.onnx -o /path/to/output.pb