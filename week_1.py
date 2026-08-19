import math

def sigmoid(z):
    return 1/(1+math.exp(z))

def neuron_forward(inputs, weights, bias):
    z = sum(x*w for x,w in zip(inputs, weights)) + bias
    activation = sigmoid(z=z)

    return activation

inputs = [1.2, 0.5, -1.5]
weights = [0.4, -0.7, 0.2]
bias = 0.1

output = neuron_forward(inputs, weights, bias)
print(f"neuron outputs: {output:.4f}")

#======================================================
import math

def sigmoid(z):
    return 1/(1+math.exp(z))

def layer_forward(inputs, weights, bias):
    outputs=[]

    for neuron_weigths, neuron_bias in zip(weights,bias):
        z = sum(x*w for x,w in zip(inputs, neuron_weigths)) + neuron_bias
        outputs.append(sigmoid(z=z))

    return outputs

inputs = [1.2, 0.5, -1.5]
weights = [
    [0.4, -0.7, 0.2],
    [-0.2, 0.9, 0.5]
]

biases = [0.1, -0.3]
outputs = layer_forward(inputs, weights, biases)

for i, val in enumerate(outputs, 1):
    print(f"neuron {i} output: {val:.4f}")

# ===============================================================================
def mean_squared_error(true,predict):
    squared_errors=[(y-y_p)**2 for y,y_p in zip(true,predict)]

    loss = sum(squared_errors) / len(true)
    return loss

true = [1.0, 0.0, 1.0]
pred = [0.85, 0.15, 0.40]

loss_value = mean_squared_error(true, pred)
print(f"MSE loss val: {loss_value:.4f}")

#============================================================
import matplotlib.pyplot as plt

X = [1.0, 2.0, 3.0, 4.0]
y_true = [3.0, 5.0, 7.0, 9.0]

def forward(x, w, b):
    return w * x + b

def compute_mse(X, y_true, w, b):
    n = len(X)
    squared_errors = [(y - forward(x, w, b)) ** 2 for x, y in zip(X, y_true)]
    return sum(squared_errors) / n

fixed_b = 1.0
weights = [i * 0.1 for i in range(-20, 61)]
losses = [compute_mse(X, y_true, w, fixed_b) for w in weights]

min_loss = min(losses)
optimal_w = weights[losses.index(min_loss)]
print(f"En Düşük Kayıp: {min_loss:.4f} (Optimal w = {optimal_w:.1f})")

plt.figure(figsize=(8, 5))
plt.plot(weights, losses, label="Kayıp Eğrisi (Loss Landscape)", color="#1f77b4", linewidth=2)
plt.scatter([optimal_w], [min_loss], color="#d62728", s=80, zorder=5, label=f"Global Minimum (w={optimal_w:.1f})")

plt.title("Ağırlık (w) Değişimine Göre MSE Kayıp Eğrisi", fontsize=12)
plt.xlabel("Ağırlık Değeri (w)", fontsize=11)
plt.ylabel("MSE Kaybı", fontsize=11)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.show()
#==================================================================== 

X = [1.0, 2.0, 3.0, 4.0]
y_true = [3.0, 5.0, 7.0, 9.0]

def forward(x, w, b):
    return w * x + b

def compute_loss(w, b):
    n = len(X)
    squared_errors = [(y - forward(x, w, b)) ** 2 for x, y in zip(X, y_true)]
    return sum(squared_errors) / n

def compute_gradients(w, b, h=1e-5):
    grad_w = (compute_loss(w + h, b) - compute_loss(w - h, b)) / (2 * h)
    
    grad_b = (compute_loss(w, b + h) - compute_loss(w, b - h)) / (2 * h)
    
    return grad_w, grad_b

w = 0.2
b = 0.2
learning_rate = 0.05
epochs = 100

for epoch in range(1, epochs + 1):
    current_loss = compute_loss(w, b)
    grad_w, grad_b = compute_gradients(w, b)

    w -= learning_rate * grad_w
    b -= learning_rate * grad_b
    
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d} | Loss: {current_loss:8.4f} | w: {w:6.4f} | b: {b:6.4f} | grad_w: {grad_w:7.4f}")