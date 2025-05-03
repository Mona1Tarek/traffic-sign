from ultralytics import YOLO
import onnx
import onnxruntime as ort
import numpy as np

def optimize_onnx():
    # Load the YOLOv8n model
    model = YOLO('best.pt')
    
    # Export to ONNX with optimizations
    model.export(
        format='onnx',
        imgsz=640,
        half=False,  # Disable FP16 since we're using CPU
        simplify=True,  # Simplify the model
        opset=12,  # Use ONNX opset 12
        dynamic=True,  # Enable dynamic shapes
        batch=1  # Set batch size to 1 for inference
    )
    
    # Load the exported ONNX model
    onnx_model = onnx.load('best.onnx')
    
    # Get the input name from the model
    input_name = onnx_model.graph.input[0].name
    
    # Optimize the model using ONNX Runtime
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Create a session with optimization options
    session = ort.InferenceSession('best.onnx', sess_options, providers=['CPUExecutionProvider'])
    
    # Save the optimized model
    onnx.save(onnx_model, 'best_optimized.onnx')
    
    # Test the optimized model
    dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
    outputs = session.run(None, {input_name: dummy_input})
    print("Optimized model test successful!")
    print(f"Output shape: {outputs[0].shape}")
    print(f"Input name used: {input_name}")

if __name__ == '__main__':
    optimize_onnx() 