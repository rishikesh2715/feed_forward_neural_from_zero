# Feedforward Neural Network from Scratch 🧠✨

This project is a from-scratch implementation of a **feedforward neural network** using only NumPy, designed for:

- ✅ Binary classification (Modified XOR Problem)
- ✅ Regression (Sinusoidal dataset from Excel)
- ✅ Flexible architecture (change hidden layers, activations, etc.)


## 🚀 How to Run

1. Make sure you have Python 3.8+ and `pip` installed.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
    ```

3. Run the script:
   ```bash
   python main.py
   ```
This will:

- Train the XOR classifier and plot its loss + 2D & 3D decision surface

- Train the regression model twice (3 and 20 hidden units) and plot the fitted results + training error


## ⚙️ Customization

Want to change the architecture? Modify the `layer_sizes` and `activations` in `train.py`:

```python
layer_sizes = [input_dim, 16, 16, 1]
activations = ['tanh', 'relu']

```
Supported activation functions:
- `sigmoid`
- `tanh`
- `relu`

You can also adjust the learning rate, batch size, and number of epochs in the `train.py`.

## ✨ Features

- ✅ Fully vectorized forward and backward pass  
- ✅ Manual weight and bias updates (no PyTorch!)  
- ✅ Clean codebase with docstrings  
- ✅ Generalizable for any number of layers  
- ✅ Beautiful plots (loss, surface, fit)  

---

## 📦 Dependencies

- `numpy`  
- `pandas`  
- `matplotlib`  
- `seaborn`  
