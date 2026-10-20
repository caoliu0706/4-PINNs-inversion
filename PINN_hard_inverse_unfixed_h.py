# -*- coding: utf-8 -*-
"""
Created on Nov 4 2024

@author: caoliu
"""

#---------------- 使用说明 ----------------#
"""
使用场景：
采用PINN求解一维情形下的瞬态热传导的逆问题，其中，左端为对流换热边界（随温变化的对流换热系数），右端为绝热边界
控制方程及初始、边界条件进行了无量纲化处理
随温变化的对流换热系数通过设置单线性模型实现
反演的参数为：ratio、h_solidus
对初始条件施加了硬约束

使用方法：
1）设置长度、物性参数、初始温度、反演参数、提取点位置等信息
2）运行代码，需要读取data.txt文件，文件中有3列数据，分别有量纲位置、有量纲时间、有量纲温度
3）代码目录下会生成variable_history.txt文件，该文件记录了不同迭代步下反演得到的对流换热系数
4）代码目录下会生成predict.txt文件，该文件记录了最后训练好的神经网络预测的提取点随时间变化的温度
5）输出对应图像（训练残差及整体预测结果）并保存训练过程的残差
6）输出better_model文件夹，该文件夹中存放的是更好的model
7）输出saved_last_model文件，该文件为最终的model
"""

# 导入Backend
from turtle import shape
import deepxde as dde
import numpy as np
import paddle

L = 0.15 #1.0 #0.2  # 长度，单位：m
rho = 8046.0 #1.0 #8046.0 # 密度，单位：kg/m3
cp = 600.0 #1.0 #600.0  # 比热容，单位:J/(kg·K)
k = 136.1 #1.0 #136.1  # 导热系数，单位：W/(m·K)
Ta = 120.0 #1.0 #100.0  # 外界温度，单位：℃
initial_temp = 1100.0 #2.0 #1000.0 # 初始温度，单位：℃
total_time = 530.0 #10.0 #100.0 # 总时长，单位：s

# 设置随温变化的对流换热系数（单线性模型），单位：W/(m2·K)，不同温度下的对流换热系数为：ratio*(T-T_solidus)+h_solidus
T_solidus = 877.0 # 固相线温度，单位：℃
ratio = dde.Variable(3.0) # 需要反求的参数1，h随温变化的斜率，单位：W/(m2·K2)
h_solidus = dde.Variable(2000.0) # 需要反求的参数2，温度为固相线温度时的对流换热系数，单位：W/(m2·K)

alpha_1 = ( k * total_time ) / ( rho * cp * L * L ) # 无量纲化后的参数1

position = 1.0/30.0 # 提取点的相对位置，范围为：0~1，实际位置为position*L
timeN = 1060 # 划分的时间段数，用于控制输出提取点的随时间变化的温度

# 定义PDE损失
def pde(x, y):
    dy_t = dde.grad.jacobian(y, x, i=0, j=1) # y对t的一阶偏导
    dy_xx = dde.grad.hessian(y, x, i=0, j=0) # y对x的二阶偏导
    return (
        dy_t - dy_xx * alpha_1
    )

# 读取data.txt文件并分割数据
def read_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    data = [line.strip().split() for line in lines]
    return data

# 将数据存入numpy数组
def store_data(data):
    observe_x = np.array([[float(num) for num in row[:2]] for row in data])
    observe_y_temp = np.array([float(row[2]) for row in data])
    observe_y_temp = observe_y_temp.reshape(-1, 1) # 转换为列向量
    return observe_x, observe_y_temp

# 左端边界
def boundary_left(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 0.0)

# 右端边界
def boundary_right(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 1.0)

# 定义几何域、时间域、方程类型
geom = dde.geometry.Interval(0.0, 1.0) # 无量纲化后的几何域
timedomain = dde.geometry.TimeDomain(0.0, 1.0) # 无量纲化后的时间域
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# 定义边界条件和初始条件的损失
T_solidus_nondimensional = ( T_solidus - Ta ) / ( initial_temp - Ta )
bc_left = dde.icbc.RobinBC(geomtime, lambda x, y: -(ratio*(initial_temp-Ta)*(y-T_solidus_nondimensional)+h_solidus) * L * y / k, boundary_left) # 对流换热
bc_right = dde.icbc.NeumannBC(geomtime, lambda x: 0.0, boundary_right) # 绝热
ic = dde.icbc.IC(geomtime, lambda x: 1.0, lambda _, on_initial: on_initial) # 初始温度

# 设置观测点的损失，观测点的输入是无量纲位置、无量纲时间，输出为无量纲温度
file_path = 'data.txt'  # 文件路径，文中有3列数据，分别为有量纲位置、有量纲时间、有量纲温度，读取完之后需要进行无量纲化处理
data = read_data(file_path)
observe_x_1, observe_y_1 = store_data(data)
observe_x_2 = np.column_stack((observe_x_1[:, 0] / L, observe_x_1[:, 1] / total_time)) # 获得无量纲化的位置和时间
observe_y_2 = ( observe_y_1 - Ta ) / ( initial_temp - Ta ) # 获得无量纲化的温度
observe_y = dde.icbc.PointSetBC(observe_x_2, observe_y_2, component=0)

# 定义瞬态方程
data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc_left, bc_right, ic, observe_y],
    num_domain=10000,
    num_boundary=800,
    num_initial=400,
    anchors=observe_x_2,
    num_test=10000,
)

# 构建神经网络架构
layer_size = [2] + [32] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)
net.apply_output_transform(lambda x, y: x[:,1:2] * y + 1.0) # 对初始条件施加硬约束

# 建立model
model = dde.Model(data, net)

# 设置model的学习方法、学习率、评判误差、需反演的参数
model.compile("adam", lr=0.001, external_trainable_variables=[ratio, h_solidus])
variable = dde.callbacks.VariableValue([ratio, h_solidus], period=1000, filename="variable_history.txt", precision=9)
checkpoint = dde.callbacks.ModelCheckpoint("better_model/better_model", verbose=1, save_better_only=True ) # 定义回调函数以保存最佳模型
loss_history, train_state = model.train(iterations=5000000, callbacks=[variable, checkpoint])

# 保存选取点的实际时间和实际温度
input_x_1 = np.column_stack((np.full(timeN+1, position), np.linspace(0.0, 1.0, timeN+1))) # 无量纲化的输入
output_y_1 = model.predict(input_x_1) # 无量纲化的输出
input_x_2 = np.column_stack((input_x_1[:, 0] * L, input_x_1[:, 1] * total_time))  # 量纲化的输入
output_y_2 = output_y_1 * ( initial_temp - Ta ) + Ta # 量纲化的输出
np.savetxt("predict.txt", np.hstack((input_x_2, output_y_2)))

# 输出结果
dde.saveplot(loss_history, train_state, issave=True, isplot=True)

# 保存训练好的神经网络
model.save("saved_last_model")
