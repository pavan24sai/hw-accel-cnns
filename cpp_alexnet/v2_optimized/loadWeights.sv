// Convolutional Layer Weight Loader
task load_weights(
  input string filename,
  output bit success
);
  int fd;
  byte file_data[];
  success = 0;
  
  fd = $fopen(filename, "rb");
  if (!fd) begin
    $error("Error opening file: %s", filename);
    return;
  end
  
  if ($fread(file_data, fd) == 0) begin
    $error("Failed to read file");
    $fclose(fd);
    return;
  end
  $fclose(fd);

  // Kria-specific 32-bit alignment
  int expected_size = (outputChannels*inputChannels*kernelSize**2 + 
                      outputChannels) * 4;
                      
  if (file_data.size() != expected_size) begin
    $error("Size mismatch: Expected %0d bytes, Got %0d", 
          expected_size, file_data.size());
    return;
  end

  // Zynq UltraScale+ memory optimization
  foreach(weights[i,j,k]) begin
    int offset = (i*inputChannels*kernelSize**2 + 
                 j*kernelSize**2 + k) * 4;
    weights[i][j][k] = $bitstoshortreal({<<byte{file_data[offset+:4]}});
  end

  // Shared L2 cache aware bias loading
  int bias_offset = outputChannels*inputChannels*kernelSize**2 * 4;
  foreach(bias[i]) begin
    bias[i] = $bitstoshortreal({<<byte{file_data[bias_offset+i*4+:4]}});
  end

  success = 1;
endtask

// CNN Weight Loader
task load_cnn_weights(
  input string base_path,
  output bit success
);
  success = 1;
  
  foreach (layers[i]) begin
    if (layers[i].layer_type inside {POOLING_LAYER, NORMALIZATION_LAYER}) continue;
    
    string filename = {base_path, "/", layers[i].name, "_kria.bin"};
    bit layer_success;
    
    // Kria-specific PS-PL interface optimization
    layers[i].load_weights(filename, layer_success);
    
    if (!layer_success) begin
      $error("Load failed: %s", layers[i].name);
      success = 0;
    end
  end
endtask
