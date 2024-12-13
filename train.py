import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


# Sigmoid, ReLU 和 Leaky ReLU 激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)


def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)


# 初始化权重和偏置
np.random.seed(0)

# 设置参数
input_neurons = 40
hidden_neurons_1 = 40  # 增大隐藏层神经元数量
hidden_neurons_2 = 20
output_neurons = 1
learning_rate = 0.001  # 调整学习率
epochs = 500  # 增加训练轮数
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8


# 权重初始化：使用Xavier初始化来防止梯度消失/爆炸
def xavier_init(shape):
    return np.random.randn(*shape) * np.sqrt(2. / (shape[0] + shape[1]))


# 初始化权重和偏置
w_ih1 = xavier_init((input_neurons, hidden_neurons_1))
b_h1 = np.zeros(hidden_neurons_1)

w_h1h2 = xavier_init((hidden_neurons_1, hidden_neurons_2))
b_h2 = np.zeros(hidden_neurons_2)

w_h2o = xavier_init((hidden_neurons_2, output_neurons))
b_o = np.zeros(output_neurons)

# 初始化Adam算法的变量
m_w_ih1, v_w_ih1 = np.zeros_like(w_ih1), np.zeros_like(w_ih1)
m_b_h1, v_b_h1 = np.zeros_like(b_h1), np.zeros_like(b_h1)

m_w_h1h2, v_w_h1h2 = np.zeros_like(w_h1h2), np.zeros_like(w_h1h2)
m_b_h2, v_b_h2 = np.zeros_like(b_h2), np.zeros_like(b_h2)

m_w_h2o, v_w_h2o = np.zeros_like(w_h2o), np.zeros_like(w_h2o)
m_b_o, v_b_o = np.zeros_like(b_o), np.zeros_like(b_o)


# Adam优化更新
def adam_update(param, grad, m, v, t):
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad ** 2)
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    param += learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    return param, m, v


# 提取图像特征：灰度质心及周围区域的灰度均值
def compute_center_and_extract_features(roi_image, pixel_width=20):
    rows, cols = roi_image.shape
    yc = rows // 2

    # 计算灰度质心
    gray_core = int(np.sum(roi_image[yc, :] * np.arange(cols)) / np.sum(roi_image[yc, :]))

    # 提取灰度质心周围区域的特征
    start_x = max(0, gray_core - pixel_width)
    end_x = min(cols, gray_core + pixel_width)

    features = roi_image[yc, start_x:end_x]
    if len(features) < 2 * pixel_width:
        features = np.pad(features, (0, 2 * pixel_width - len(features)), 'constant')

    return features.flatten()


# 读取真实宽度
def load_widths(width_file):
    widths = {}
    with open(width_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 2:
                filename, width = parts
                widths[filename] = float(width)
    return widths


# 损失函数：Huber损失
def huber_loss(true_width, output, delta=1.0):
    error = np.abs(true_width - output)
    loss = np.where(error <= delta, 0.5 * error ** 2, delta * (error - 0.5 * delta))
    return np.mean(loss)


# 前向传播
def forward(x):
    h1_input = np.dot(x, w_ih1) + b_h1
    h1_output = relu(h1_input)  # 使用ReLU

    h2_input = np.dot(h1_output, w_h1h2) + b_h2
    h2_output = relu(h2_input)  # 使用ReLU

    o_input = np.dot(h2_output, w_h2o) + b_o
    o_output = o_input  # 输出层无激活函数，回归问题直接输出

    return h1_output, h2_output, o_output


# 训练模型
# 在train函数的末尾添加这部分代码
def train(image_folder, width_file):
    global w_ih1, b_h1, w_h1h2, b_h2, w_h2o, b_o
    global m_w_ih1, v_w_ih1, m_b_h1, v_b_h1
    global m_w_h1h2, v_w_h1h2, m_b_h2, v_b_h2
    global m_w_h2o, v_w_h2o, m_b_o, v_b_o

    true_widths = load_widths(width_file)
    filenames = list(true_widths.keys())
    train_files, val_files = train_test_split(filenames, test_size=0.2, random_state=42)

    best_loss = float('inf')
    best_train_losses = []  # 保存每50轮的最佳训练损失

    for epoch in range(1, epochs + 1):
        epoch_loss = 0
        for filename in train_files:
            true_width = true_widths[filename]
            image_path = os.path.join(image_folder, filename)
            if not os.path.exists(image_path):
                continue

            roi_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            features = compute_center_and_extract_features(roi_image)

            h1_output, h2_output, output = forward(features)

            error = true_width - output
            loss = np.mean(error ** 2)

            epoch_loss += loss

            d_output = error
            d_h2 = d_output.dot(w_h2o.T) * relu_derivative(h2_output)
            d_h1 = d_h2.dot(w_h1h2.T) * relu_derivative(h1_output)

            w_h2o, m_w_h2o, v_w_h2o = adam_update(w_h2o, h2_output.reshape(-1, 1) * d_output, m_w_h2o, v_w_h2o, epoch)
            b_o, m_b_o, v_b_o = adam_update(b_o, d_output, m_b_o, v_b_o, epoch)

            w_h1h2, m_w_h1h2, v_w_h1h2 = adam_update(w_h1h2, h1_output.reshape(-1, 1) * d_h2, m_w_h1h2, v_w_h1h2, epoch)
            b_h2, m_b_h2, v_b_h2 = adam_update(b_h2, d_h2, m_b_h2, v_b_h2, epoch)

            w_ih1, m_w_ih1, v_w_ih1 = adam_update(w_ih1, features.reshape(-1, 1) * d_h1, m_w_ih1, v_w_ih1, epoch)
            b_h1, m_b_h1, v_b_h1 = adam_update(b_h1, d_h1, m_b_h1, v_b_h1, epoch)

        epoch_loss /= len(train_files)
        print(f'Epoch {epoch}/{epochs}, Training Loss: {epoch_loss:.4f}')

        if epoch % 50 == 0:
            best_train_losses.append(epoch_loss)
            # 保存到txt文件中
            with open('best_train_losses.txt', 'a') as f:
                f.write(f'Epoch {epoch}: {epoch_loss:.4f}\n')

        val_loss = 0
        for filename in val_files:
            true_width = true_widths[filename]
            image_path = os.path.join(image_folder, filename)
            if not os.path.exists(image_path):
                continue

            roi_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            features = compute_center_and_extract_features(roi_image)

            _, _, output = forward(features)

            loss = np.mean((true_width - output) ** 2)

            val_loss += loss

        val_loss /= len(val_files)
        print(f'Epoch {epoch}/{epochs}, Validation Loss: {val_loss:.4f}')

        if val_loss < best_loss:
            best_loss = val_loss
            print("Best model found, saving weights.")
            np.savez('best5.npz', w_ih1=w_ih1, b_h1=b_h1, w_h1h2=w_h1h2, b_h2=b_h2, w_h2o=w_h2o, b_o=b_o)

    return best_train_losses  # 返回保存的最佳训练损失列表




# 请根据实际情况修改路径
image_folder = r'G:\cuda\test\final\roi'
width_file = 'image_dataset.txt'

train(image_folder, width_file)
