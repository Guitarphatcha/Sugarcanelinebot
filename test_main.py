print("Step 0")
import tflite_runtime.interpreter as tflite 
print("Step 1")
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
print("Step 2")
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Complete")
