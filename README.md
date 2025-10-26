# Video Classification using Tim
This repository implements a video classification pipeline using the **TimeSFormer architecture**, trained and evaluated on a 5-class sports dataset containing videos of **Long Jump**, **Javelin Throw**, **Skiing**, **Pole Vault**, and **Horse Riding**.

The notebook includes:
1. A **custom Transformer-based model** trained from scratch for benchmarking.  
2. A **pretrained TimeSFormer (on Kinetics-400)** model fine-tuned for the 5-class dataset.  
3. **Pruned TimeSFormer variants** (2, 4, 6, and 8 layers pruned) optimized for faster inference.

---

## Overview

This project explores **temporal modeling** in videos using attention-based architectures.  
The main objective was to compare and analyze:
- A **custom-built Transformer** trained from scratch for video understanding.  
- A **pretrained TimeSFormer** fine-tuned for domain-specific tasks.  
- Multiple **pruned TimeSFormer** variants to analyze the trade-off between model complexity and accuracy.

---

## Key Features

- Custom video transformer implementation featuring:
  - Frame-wise CNN feature extractor  
  - Temporal positional encoding  
  - Multi-head self-attention layers  
  - Classification head  
- Transfer learning using pretrained **TimeSFormer**
- **Progressive pruning** to enhance computational efficiency  
- Comprehensive data augmentations (spatial and temporal)
- Evaluation using:
  - Confusion matrix  
  - Accuracy and loss curves  
  - Testing on real-world YouTube videos for generalization

---

## Dataset

- **Total videos:** 700  
- **Classes:** Long Jump, Javelin Throw, Skiing, Pole Vault, Horse Riding  
- **Data split:** 80% training / 10% validation / 10% testing  
- **Frame size:** 224×224  
- **Frames per video:** 8 (uniformly sampled)

This setup follows the original TimeSFormer design which used short video clips (≤10 seconds, 25 FPS) for temporal context.

---

## Model Details

### Custom Transformer Model
- 4 convolutional layers for spatial feature extraction  
- 4 Transformer Encoder layers for temporal modeling  
- Classification head with softmax output  
- Trained for 80 epochs with a learning rate of 1e-3  
- Achieved approximately **29.58% test accuracy** and **52.3% validation accuracy**

### Pretrained TimeSFormer
- 12 Transformer Encoder layers (hidden dimension = 768, 12 attention heads)  
- Pretrained on **Kinetics-400**  
- Fine-tuned by unfreezing the final encoder and classifier layers  
- Trained for 50 epochs with a learning rate of 1e-4  
- Achieved **96.0% test accuracy** and **100% validation accuracy**

---

## Pruned TimeSFormer Variants

### 2-Layer Pruned
- 10 encoder layers (2 pruned from the original 12)
- Model size reduced by ~18%
- Maintained identical accuracy to the unpruned model  

### 4-Layer Pruned
- 8 encoder layers
- Model size reduced by ~32%
- Slight improvement in inference time with a little accuracy loss 

### 6-Layer Pruned
- 6 encoder layers
- Model size reduced by ~46%
- Noticeable trade-off between speed and performance  

### 8-Layer Pruned
- 4 encoder layers
- Model size reduced by ~60%
- Significant reduction in computational cost, but a little high accuracy drop  

---

## Results

| Model Variant                | Encoder Layers | Youtube video Test Accuracy | Validation Accuracy | Parameters |
|-------------------------------|----------------|----------------|---------------------|-------------|
| Custom Transformer (Scratch)  | 4              | 20.0%         | 52.3%               | ~2M         | 
| Pretrained TimeSFormer        | 12             | 96.0%          | 100%                | ~121M       | 
| Pruned TimeSFormer (2 layers) | 10             | 96.0%          | 100%                | ~101M        |
| Pruned TimeSFormer (4 layers) | 8              | 88.0%          | 100%               | ~81M        |
| Pruned TimeSFormer (6 layers) | 6              | 80.0%          | 100%               | ~61M        |
| Pruned TimeSFormer (8 layers) | 4              | 72.0%          | 97.1%               | ~40M        |

- Pruning resulted in **significant reduction in parameters** and **faster inference**, making it more suitable for deployment on resource-constrained systems.  
- The 2-layer pruned model retained the same accuracy as the full model while reducing size of the model by **18-20%**.

---
