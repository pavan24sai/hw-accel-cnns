import torch
import torchvision.models as models
import numpy as np
import os
import sys

def create_directory(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def save_conv_layer(layer, name, output_dir):
    """Save convolutional layer weights and biases"""
    weights = layer.weight.detach().cpu().numpy()
    bias = layer.bias.detach().cpu().numpy()
    
    # Reshape weights to match C++ implementation's expected format
    # Our C++ implementation expects weights in format [M][N][K*K]
    M, N, K, _ = weights.shape
    reshaped_weights = weights.reshape(M, N, K*K)
    
    # Save weights and bias
    weights_path = os.path.join(output_dir, f"{name}.bin")
    with open(weights_path, 'wb') as f:
        reshaped_weights.tofile(f)
    
    bias_path = os.path.join(output_dir, f"{name}_bias.bin")
    with open(bias_path, 'wb') as f:
        bias.tofile(f)
    
    print(f"Saved {name}: Weights shape {reshaped_weights.shape}, Bias shape {bias.shape}")
    return weights.shape

def save_fc_layer(layer, name, output_dir):
    """Save fully connected layer weights and biases"""
    weights = layer.weight.detach().cpu().numpy()
    bias = layer.bias.detach().cpu().numpy()
    
    # Save weights and bias
    weights_path = os.path.join(output_dir, f"{name}.bin")
    with open(weights_path, 'wb') as f:
        weights.tofile(f)
    
    bias_path = os.path.join(output_dir, f"{name}_bias.bin")
    with open(bias_path, 'wb') as f:
        bias.tofile(f)
    
    print(f"Saved {name}: Weights shape {weights.shape}, Bias shape {bias.shape}")
    return weights.shape

def create_combined_weight_files(model, model_features, output_dir):
    """Create combined weight files that include both weights and biases in the format expected by the C++ code"""
    
    # Use the same indices as in the main function
    conv_layers = [0, 3, 6, 8, 10]  # Indices of conv layers in model.features
    
    # Process convolutional layers
    for i, idx in enumerate(conv_layers):
        name = f"conv{i+1}"
        layer = model_features[idx]
        
        weights = layer.weight.detach().cpu().numpy()
        bias = layer.bias.detach().cpu().numpy()
        
        # Reshape to match C++ implementation's expected format
        M, N, K, _ = weights.shape
        reshaped_weights = weights.reshape(M, N, K*K)
        
        # Create combined binary file
        combined_path = os.path.join(output_dir, f"{name}_combined.bin")
        with open(combined_path, 'wb') as f:
            # First write all weights
            reshaped_weights.tofile(f)
            # Then write all biases
            bias.tofile(f)
        
        print(f"Created combined file for {name}: {os.path.getsize(combined_path)} bytes")
    
    # Process fully connected layers
    fc_names = ['fc6', 'fc7', 'fc8']
    fc_layers = [model.classifier[1], model.classifier[4], model.classifier[6]]
    
    for i, layer in enumerate(fc_layers):
        name = fc_names[i]
        
        weights = layer.weight.detach().cpu().numpy()
        bias = layer.bias.detach().cpu().numpy()
        
        # Create combined binary file
        combined_path = os.path.join(output_dir, f"{name}_combined.bin")
        with open(combined_path, 'wb') as f:
            # First write all weights
            weights.tofile(f)
            # Then write all biases
            bias.tofile(f)
        
        print(f"Created combined file for {name}: {os.path.getsize(combined_path)} bytes")

def generate_metadata_file(output_dir, shapes):
    """Generate a metadata file with layer shapes to help C++ code load weights correctly"""
    metadata_path = os.path.join(output_dir, "network_metadata.txt")
    with open(metadata_path, 'w') as f:
        for layer_name, shape in shapes.items():
            if 'conv' in layer_name:
                M, N, K, _ = shape
                f.write(f"{layer_name} {M} {N} {K}\n")
            else:  # FC layer
                out_features, in_features = shape
                f.write(f"{layer_name} {out_features} {in_features}\n")
    
    print(f"Generated metadata file: {metadata_path}")

def main():
    # Create output directory
    output_dir = "weights"
    create_directory(output_dir)
    
    # Load pre-trained AlexNet model
    print("Downloading pre-trained AlexNet model...")
    model = models.alexnet(pretrained=True)
    print("Model downloaded successfully!")
    
    # Inspect model architecture
    print("\nModel Architecture:")
    print("Features:", model.features)
    print("Classifier:", model.classifier)
    
    # Extract and save feature layers (convolutional layers)
    print("\nExtracting and saving convolutional layers...")
    layer_shapes = {}
    
    # Extract convolutional layers
    conv_layers = [0, 3, 6, 8, 10]  # Indices of conv layers in model.features
    for i, idx in enumerate(conv_layers):
        layer = model.features[idx]
        name = f"conv{i+1}"
        shape = save_conv_layer(layer, name, output_dir)
        layer_shapes[name] = shape
    
    # Extract fully connected layers
    print("\nExtracting and saving fully connected layers...")
    fc_layers = [(1, 'fc6'), (4, 'fc7'), (6, 'fc8')]  # Indices of FC layers in model.classifier
    for idx, name in fc_layers:
        layer = model.classifier[idx]
        shape = save_fc_layer(layer, name, output_dir)
        layer_shapes[name] = shape
    
    # Create combined weight files for easy loading in C++
    print("\nCreating combined weight files...")
    create_combined_weight_files(model, model.features, output_dir)
    
    # Generate metadata file with layer shapes
    generate_metadata_file(output_dir, layer_shapes)
    
    print("\nWeight extraction complete!")
    print(f"All weights saved to {output_dir}/")
    print("You can now use these weights with your C++ implementation.")

if __name__ == "__main__":
    main()