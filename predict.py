import os
import cv2
import numpy as np
import time
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
def load_widths(width_file):
    widths = {}
    with open(width_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 2:
                filename, width = parts
                widths[filename] = float(width)
    return widths
def mse_loss(true_width, output):
    error = true_width - output
    return np.mean(error ** 2)
# 读取模型权重
def load_model(model_path):
    model = np.load(model_path, allow_pickle=True)
    w_ih1 = model['w_ih1']
    b_h1 = model['b_h1']
    w_h1h2 = model['w_h1h2']
    b_h2 = model['b_h2']
    w_h2o = model['w_h2o']
    b_o = model['b_o']
    return w_ih1, b_h1, w_h1h2, b_h2, w_h2o, b_o


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


# 前向传播
def forward(x, w_ih1, b_h1, w_h1h2, b_h2, w_h2o, b_o):
    h1_input = np.dot(x, w_ih1) + b_h1
    h1_output = relu(h1_input)  # 使用ReLU

    h2_input = np.dot(h1_output, w_h1h2) + b_h2
    h2_output = relu(h2_input)  # 使用ReLU

    o_input = np.dot(h2_output, w_h2o) + b_o
    o_output = o_input  # 输出层无激活函数，回归问题直接输出

    return o_output

def huber_loss(true_width, output, delta=1.0):
    error = np.abs(true_width - output)
    loss = np.where(error <= delta, 0.5 * error ** 2, delta * (error - 0.5 * delta))
    return np.mean(loss)

# 测试模型
# 计算预测准确率（预测值与实际值的比）
def prediction_accuracy(true_width, output):
    return output / true_width

# 测试模型
# 计算预测精度：1 - (|预测值 - 真实值| / 真实值)
def prediction_accuracy(true_width, output):
    return 1 - (np.abs(output - true_width) / true_width)

# 测试模型
def test(image_folder, width_file, model_path):
    # 加载真实宽度数据
    true_widths = load_widths(width_file)
    filenames = list(true_widths.keys())

    # 加载最佳权重
    w_ih1, b_h1, w_h1h2, b_h2, w_h2o, b_o = load_model(model_path)

    total_accuracy = 0
    total_time = 0  # 用于记录总推理时间

    for filename in filenames:
        true_width = true_widths[filename]
        image_path = os.path.join(image_folder, filename)
        if not os.path.exists(image_path):
            continue

        roi_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        features = compute_center_and_extract_features(roi_image)

        start_time = time.time()
        # 前向传播进行推理
        output = forward(features, w_ih1, b_h1, w_h1h2, b_h2, w_h2o, b_o)
        end_time = time.time()

        inference_time = end_time - start_time
        total_time += inference_time  # 累加到总推理时间

        # 计算精度（1 - (|预测值 - 真实值| / 真实值)）
        accuracy_value = prediction_accuracy(true_width, output)

        total_accuracy += accuracy_value[0]  # accuracy_value 是一个数组，取第一个元素

        print(f"Filename: {filename}, Predicted: {output[0]:.4f}, True: {true_width}, Accuracy: {accuracy_value[0]:.4f}")

    avg_accuracy = total_accuracy / len(filenames)
    avg_time = total_time / len(filenames)

    print(f"Average Prediction Accuracy: {avg_accuracy:.4f}")
    print(f"Total Inference Time: {total_time:.6f} seconds")
    print(f"Average Inference Time per Image: {avg_time:.6f} seconds")


# 请根据实际情况修改路径
image_folder = r'G:\cuda\test\final\roi'
width_file = 'image_dataset.txt'
model_path = 'best5.npz'

test(image_folder, width_file, model_path)
