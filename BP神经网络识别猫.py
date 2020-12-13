import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import tkinter as tk
import tkinter.filedialog

""" 0.数据读取与处理
    1.初始化参数
    2.前向线性传播
    3.计算激活值
    4.计算误差
    5.反向传播
    6.更新参数
    7.预测
    8.额外功能"""


def Read_label(path):  # 读取储存类别的文件
    with open(path, 'r') as file:  # 只读
        # 去除返回字符串中的空格和换行符
        data = list(file.read().replace(' ', '').replace('\n', ''))
    label = list(map(int, data))  # 将列表元素转换为整型

    return label


def Read_data(path):  # 读取储存图片的文件
    img_list = []
    filenames = os.listdir(path)
    filenames.sort(key=lambda x: int(x[:-4]))  # 将filenames排序
    for filename in filenames:
        img_list.append(cv2.resize(cv2.imread(
            path + filename, 1), (64, 64)))  # 以BGR形式读取图片

    return np.array(img_list)


def Init_params(layers):  # 初始化权重矩阵和偏置
    np.random.seed(3)  # 保证每次初始化一样
    parameters = {}  # 该字典用来储存参数
    L = len(layers)  # 神经网络的层数

    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layers[l],
                                                   layers[l - 1]) / np.sqrt(layers[l - 1])  # Xaiver初始化方法
        parameters['b'+str(l)] = np.zeros((layers_dims[l], 1))  # 初始化为0

    return parameters


def TanH(Z):
    return (np.exp(2*Z)-1)/(np.exp(2*Z)+1)


def Sigmoid(Z):
    return 1/(1+np.exp(-Z))


def Linear_forward(A, W, b):  # 正向线性传播
    return np.dot(W, A) + b


def Activation_forward(A_pre, W, b, Type='Hiden'):  # 计算激活值
    """
    Z表示经过线性传播后的矩阵，将输给激活函数
    A_pre表示前一层的激活值，将输给线性传播单元
    b将先广播至与W一样的大小，再进行运算
    """
    Z = Linear_forward(A_pre, W, b)
    cache = (A_pre, W, b)
    if Type == "Output":
        A = Sigmoid(Z)
    elif Type == "Hiden":
        A = TanH(Z)

    return A, cache


def Forward_propagation(X, parameters):  # 向前传播
    """
    caches用于储存cache
    每一层的激活值A将输给下一层并作用于线性传播函数
    输出层的激活值为Yhat，将输给代价函数
    """
    caches = []
    A = X
    L = len(parameters) // 2  # 获得整型
    for l in range(1, L):  # （1,3）
        A, cache = Activation_forward(
            A, parameters['W' + str(l)], parameters['b' + str(l)], "Hiden")
        caches.append(cache)
    Yhat, cache = Activation_forward(
        A, parameters['W' + str(L)], parameters['b' + str(L)], "Output")
    caches.append(cache)

    return Yhat, caches


def Compute_cost(Yhat, Y):
    m = Y.shape[1]  # 图片张数
    cost = -np.sum(np.multiply(np.log(Yhat), Y) +
                   np.multiply(np.log(1 - Yhat), 1 - Y)) / m  # 交叉熵误差计算
    # 计算Yhat的梯度，由此开始反向传播
    dYhat = - (np.divide(Y, Yhat) - np.divide(1 - Y, 1 - Yhat))
    return cost, dYhat


def Linear_backward(dZ, cache):
    A, W, b = cache  # 拆分cache
    m = A.shape[1]  # 获得图片张数
    # 除以m防止样本过大而导致数据过大
    dW = np.dot(dZ, A.T) / m  # dW/dZ=A.T,相乘代表与cost的梯度
    db = np.sum(dZ, axis=1, keepdims=True) / m  # db/dZ=I,保持维度不变
    dA = np.dot(W.T, dZ)

    return dA, dW, db


def Sigmoid_backward(dA, A):
    dZ = dA * A*(1-A)  # 相对误差的梯度
    return dZ


def TanH_backward(dA, A):
    dZ = dA*(1-A**2)  # 相对误差的梯度
    return dZ


def Activation_backward(dA, cache, A_next, activation="Hiden"):
    """
    cache储存A_pre，W，b
    A_next为输给下一层的激活值，即本层的激活值
    """
    if activation == "Hiden":
        dZ = TanH_backward(dA, A_next)
    elif activation == "Output":
        dZ = Sigmoid_backward(dA, A_next)
    dA, dW, db = Linear_backward(dZ, cache)

    return dA, dW, db


def Backward_propagation(dYhat, Yhat, Y, caches):
    grads = {}  # 用于储存梯度矩阵
    L = len(caches)  # 4
    m = Y.shape[1]  # 图片个数
    # 输出层
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)
                                                      ] = Activation_backward(dYhat, caches[L-1], Yhat, "Output")
    for l in reversed(range(L-1)):  # (3,0]
        grads["dA" + str(l + 1)], grads["dW" + str(l + 1)], grads["db" + str(l + 1)
                                                                  ] = Activation_backward(grads["dA" + str(l + 2)], caches[l], caches[l+1][0], "Hiden")

    return grads


def Update_params(parameters, grads, learning_rate):
    # 梯度下降更新参数
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l + 1)] -= learning_rate * \
            grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] -= learning_rate * \
            grads["db" + str(l + 1)]
    return parameters


def Predict(X, parameters):  # 预测
    # 将数据和训练好的参数进行一次预测
    probs, caches = Forward_propagation(X, parameters)
    if X.shape[1] == 1:  # 判断输入的是否是一张图片
        print(f'有{probs[0][0]*100}%概率是猫')
        t = tk.Label(width=15, height=1, text='是(>^ω^<)喵！' if probs[0][0] >= 0.5 else f'不是(>^ω^<)喵!', fg='Magenta',
                     bg='aqua', font=('楷书', 25))
        t.pack(side='bottom')
        t.after(4900, t.destroy)

    return probs


def Accuracy(name, probs, y):  # 准确率
    p = []  # 储存预测值
    for i in range(0, probs.shape[1]):
        p.append(int(1) if probs[0, i] > 0.5 else int(0))  # 分类
    print(f"{name}准确度为: {100*np.mean((p == y))}%")

    return p


def Save_params(parameters, layers, path):
    # 储存神经网络各层的信息
    np.savetxt(path+'layers.csv', layers, delimiter=',')
    n = len(parameters)//2
    # 将每个参数分开储存，方便读取
    for i in range(1, n+1):
        np.savetxt(path+'W'+str(i)+'.csv',
                   parameters['W'+str(i)], delimiter=',')
        np.savetxt(path+'b'+str(i)+'.csv',
                   parameters['b'+str(i)], delimiter=',')


def Load_params(path):
    parameters = {}  # 用于接收参数
    layers = list(np.loadtxt(path+'layers.csv', dtype=int, delimiter=','))
    n = len(layers)
    for i in range(1, n):
        parameters['W'+str(i)] = np.loadtxt(path+'W'+str(i) +
                                            '.csv', delimiter=",").reshape(layers[i], -1)
        parameters['b'+str(i)] = np.loadtxt(path+'b'+str(i) +
                                            '.csv', delimiter=",").reshape(layers[i], 1)
    return layers, parameters


def Create_window(parameters):
    global window  # global便于后续引用
    window = tk.Tk()
    window.title('猫咪识别器')
    window.geometry('650x650')
    window.configure(background='lightpink')
    label = tk.Label(window, text='每次识别后等待5秒即可再次识别!',
                     font=('楷书', 15), fg='Purple', bg='orange')
    label.pack(side='top')
    num = tk.Label(window, text='2000301712', font=(
        'fira_Code'), bg='orange', fg='purple')
    num.pack(side='right')
    choose_button = tk.Button(window, text='打开一张图片', fg='deeppink', bg='violet', activebackground='yellow',
                              font=('宋体', 20), command=lambda: Show_img(parameters))
    choose_button.pack(side='bottom')

    window.mainloop()


def Show_img(parameters):
    global window, img
    file = tk.filedialog.askopenfilename()  # 获取选择的文件路径
    Img = Image.open(file)
    # 使用cv2读取图片，供后续预测使用
    data = cv2.resize(cv2.imread(file, 1), (64, 64)).reshape(1, -1).T/255
    img = ImageTk.PhotoImage(Img)
    Predict_button = tk.Button(window, text='识别！', fg='CornflowerBlue', bg='slateblue', activebackground='red',
                               font=('宋体', 20), command=lambda: Predict(data, parameters))
    Predict_button.pack(side='bottom')
    Predict_button.after(5000, Predict_button.destroy)  # 5000毫秒后销毁按钮
    label_Img = tk.Label(window, image=img)  # 显示图片
    label_Img.pack(side='top')
    label_Img.after(5000, label_Img.destroy)


def Plot(costs, layers):
    """
    costs储存每100次迭代后的误差值
    layers是神经网络的信息
    """
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title("Learning rate =" +
              str(learning_rate) + f",layers={layers}")
    plt.show()


def Find_wrong(p, label, data):
    """
    p为预测的标签，是一个列表
    label为真实的标签，是一个1行n列的矩阵
    data为经过处理后的图片矩阵
    """
    m = len(p)
    diff = []  # 储存预测错误图片的索引值
    for i in range(m):
        if label[0, i] != p[i]:
            diff.append(i)
    # cv2会自动将0到1的浮点数乘以255
    Show_pic((data.T).reshape((m, 64, 64, 3)), diff)


def Show_pic(data, diff):
    for i in diff:
        cv2.imshow(f'The {i+1} picture', data[i])
        k = cv2.waitKey(0)
        # cv2.imshow当中使用字符串格式化，无法直接回车关掉图像
        if k == (32, 27, 13):  # 键值
            cv2.destroyAllWindows()


def Train_model(X, Y, parameters, learning_rate, iterations, threshold):  # 训练用模块
    costs = []  # 储存每100次迭代的损失值，用于绘制折线图
    for i in range(iterations):
        Yhat, caches = Forward_propagation(X, parameters)  # 正向传播
        cost, dYhat = Compute_cost(Yhat, Y)  # 计算误差
        grads = Backward_propagation(dYhat, Yhat, Y, caches)  # 计算梯度
        parameters = Update_params(
            parameters, grads, learning_rate)  # 更新参数
        if i % 100 == 0:
            costs.append(cost)
            print(f"迭代次数：{i}，误差值：{cost}")
        if cost < threshold:
            costs.append(cost)
            print(f"迭代次数：{i}，误差值：{cost}")
            break
    return parameters, costs, i


def BP(X, Y, test_data, test_label, path,  layers, iterations=1600, threshold=0.06,
       find_wrong=False, save_params=False, load_params=False, continue_train=False, plot=True, test=False):
    parameters = Init_params(layers)  # 接收初始化的参数
    if load_params or continue_train:  # 继续训练也需要读取储存的参数
        layers, parameters = Load_params(path)
    # 当不继续训练或不加载数据时
    if continue_train or not load_params:
        parameters, costs, times = Train_model(
            X, Y, parameters, learning_rate, iterations, threshold)
        if save_params:
            Save_params(parameters, layers, path)
    # 如果要测试则不执行画图，显示准确率和找出错误图片
    if test:
        Create_window(parameters)
    else:
        train_probs = Predict(X, parameters)
        train_p = Accuracy('训练集', train_probs, Y)
        test_probs = Predict(test_data, parameters)
        test_p = Accuracy('测试集', test_probs, test_label)
        if find_wrong:
            Find_wrong(train_p, Y, X)
            Find_wrong(test_p, test_label, test_data)
        if plot:
            Plot(costs, layers)


# 文件的路径
Path_train = 'D:/Asthestarsfall/code/train_data/'
Path_train_label = 'D:/Asthestarsfall/code/训练集类别.txt'
Path_test = 'D:/Asthestarsfall/code/data_test/'
Path_test_label = 'D:/Asthestarsfall/code/测试集类别.txt'
Path_params = 'D:/Asthestarsfall/code/parameters/'


# 测试集和训练集和图片矩阵纵向维度保持一致
train_label = np.array(Read_label(Path_train_label)).reshape(1, -1)
test_label = np.array(Read_label(Path_test_label)).reshape(1, -1)


# 转置为（64*64*3, files acount）的矩阵(同一图片的矩阵信息转换到一列)，并进行归一化
train_data = Read_data(Path_train).reshape(train_label.shape[1], -1).T/255
test_data = Read_data(Path_test).reshape(test_label.shape[1], -1).T/255

# 各层的节点数
layers_dims = [train_data.shape[0], 20, 8, 6, 1]
learning_rate = 0.0075
BP(train_data, train_label, test_data, test_label, Path_params, layers_dims,
   iterations=1600, threshold=0.006,  find_wrong=False, save_params=False,
   load_params=False, continue_train=False, plot=True, test=False)
"""
    train_data，train_label为训练集的图片数据和标签，test_data，test_lael为测试集数据和标签
    path为保存和读取参数的绝对路径n
    layers储存了每层神经元个数
    interactions为迭代次数
    threshold为阈值，误差小于该值可结束程序

    find_wrong：是否显示识别错误的图像
    save_params：是否储存训练好的参数
    load_params：是否读取参数
    continue_train：是否读取参数继续训练
    plot：是否显示误差与迭代次数关系的图像
    test：是否自己选取测试用图像
"""
