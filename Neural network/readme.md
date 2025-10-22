Neural Network Classifier – Letters A, B, and C
📌 Project Overview

This project demonstrates a simple neural network built from scratch using NumPy that learns to recognize binary image patterns of the letters A, B, and C.
It’s part of the “Neural Network for Image Recognition” assignment in Module 11, and the goal is to understand how forward propagation, backpropagation, and weight updates actually work under the hood.

🧩 Data
No external dataset is used.
Each letter (A, B, C) is represented by a 5×6 binary grid (30 pixels) where
1 = pixel on, 0 = pixel off.
These are manually defined as small NumPy arrays inside the notebook.

⚙️ Model Architecture
Input Layer: 30 neurons (one for each pixel)
Hidden Layer: 10 neurons (sigmoid activation)
Output Layer: 3 neurons (one per class – A, B, C)

Activation: Sigmoid
Loss Function: Mean Squared Error (MSE)
Optimizer: Manual Gradient Descent
Learning Rate: 0.5

🔁 Training
Implemented full forward + backward propagation manually
Updated weights using gradient descent
Tracked loss and accuracy across 2000 epochs
Visualized results using Matplotlib

📊 Results
The network successfully classifies all three letters after training.
Both loss and accuracy curves show clear learning progress.
Even though the dataset is tiny, it gives solid hands-on understanding of the math behind neural networks.
