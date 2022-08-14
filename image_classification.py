import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

FEATURE_NUM = 48


def get_hist(file_name):
    # https://docs.opencv.org/4.x/d1/db7/tutorial_py_histogram_begins.html
    image = cv2.imread(file_name)
    image = cv2.resize(image, (64, 64))
    hist_set = np.array([], dtype=np.longdouble).reshape((0, 1))
    for channel in range(3):
        hist = cv2.calcHist(image, [channel], None, [int(FEATURE_NUM / 3)], [0, 256])
        hist_set = np.vstack((hist_set, hist))

    return hist_set.reshape((1, len(hist_set)))


def read_data(image_set_path):
    """
    :param image_set_path: 图像文件夹路径，例如：./train_data/
    :return: 返回图像的直方图特征和类别集合
    """
    root = os.listdir(image_set_path)
    features_set = np.array([]).reshape((0, FEATURE_NUM))
    categorys = []
    for category, dir_name in enumerate(root):
        files = os.path.join(image_set_path, dir_name)
        file_list = os.listdir(files)
        for image_path in file_list:
            image_path = os.path.join(files, image_path)  # 图片的具体位置
            features = get_hist(image_path)  # 读取图片数据
            categorys.append(category)
            features_set = np.vstack((features_set, features))

    return features_set, np.array(categorys)


def normalizate(x):
    x_max = np.max(x, axis=0)
    x_min = np.min(x, axis=0)
    x_norm = (x - x_min) / (x_max - x_min)
    return x_norm


def logistic_func(theta_b, x_extend):
    return 1 / (1 + np.exp(-theta_b @ x_extend.T))


def gradient_descent(x_extend, theta_b, y_set):
    # http://www.matrixcalculus.org/
    t0 = np.exp(theta_b @ x_extend.T)
    w_grad = t0 / (1 + t0) @ x_extend - y_set @ x_extend
    return w_grad


def loss_func(x_set, theta_b, y_set):
    loss = 0
    for xi, yi in zip(x_set, y_set):
        loss += -yi * theta_b @ xi + np.log(1 + np.exp(theta_b @ xi))
    return loss


def train(x_set, y_set, rounds=100, learning_rate=0.1):
    theta_b = np.ones((FEATURE_NUM + 1))
    loss_record = []
    acc_record = []
    for _ in range(rounds):
        w_grad = gradient_descent(x_set, theta_b, y_set)
        theta_b -= w_grad * learning_rate
        # 计算损失值
        loss = loss_func(x_set, theta_b, y_set)
        loss_record.append(loss)
        # 计算准确率
        acc_rate = calculate_acc(x_set, y_set, theta_b)
        acc_record.append(acc_rate)
    draw_info(loss_record, acc_record)

    return theta_b


def calculate_acc(features_set, labels, theta_b):
    correct = 0
    for xi, yi in zip(features_set, labels):
        probability = logistic_func(theta_b, xi)
        if np.abs(probability - yi) < 0.5:
            correct += 1

    return correct / len(labels)


def draw_info(loss_record, acc_record):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6))
    rounds = range(len(loss_record))
    ax1.plot(rounds, loss_record)
    ax1.set_title("loss_trend")
    ax2.plot(rounds, acc_record)
    ax2.set_title("accuracy_trend")
    plt.show()


def test_display(image_set_path, theta_b):
    root = os.listdir(image_set_path)
    features_set = np.array([]).reshape((0, FEATURE_NUM))
    labels = []
    images = []
    for category, dir_name in enumerate(root):
        files = os.path.join(image_set_path, dir_name)
        file_list = os.listdir(files)
        for image_path in file_list:
            image_path = os.path.join(files, image_path)  # 图片的具体位置
            # 记录图像集
            image = cv2.imread(image_path)
            image = cv2.resize(image, (64, 64))
            images.append(image)
            # 提取特征
            features = get_hist(image_path)
            features_set = np.vstack((features_set, features))
            # 记录标签
            labels.append(category)
    labels = np.array(labels)
    # 扩充矩阵并进行归一化
    x_norm = normalizate(features_set)
    b = np.ones_like(labels, shape=(len(labels), 1))
    x_extend = np.hstack((x_norm, b))
    # 判断类别
    predict = logistic_func(theta_b, x_extend)
    ncols = int(np.ceil(len(labels) / 5))   # 行数
    # 预测结果展示
    fig, ax = plt.subplots(ncols, 5, figsize=(12, 6))
    for i in range(len(labels)):
        row = int(i / 5)
        col = i % 5
        ax[row][col].imshow(images[i])
        info = f'class:{labels[i]}\nprobability:{predict[i]:.2f}'
        ax[row][col].set_title(info)
    plt.show()


if __name__ == "__main__":
    # 读取图片并提取特征
    train_dir_path, test_dir_path = './train_data/', './test_data/'
    train_features_set, train_labels = read_data(train_dir_path)
    normal_dataset = normalizate(train_features_set)    # 归一化
    # 扩充矩阵
    b = np.ones_like(train_labels, shape=(len(train_labels), 1))
    x_extend = np.hstack((normal_dataset, b))
    # 参数优化
    theta_b = train(x_extend, train_labels, 400, 0.2)
    # 测试集展示
    test_features_set, test_labels = read_data(test_dir_path)
    test_display(test_dir_path, theta_b)
