import os
import SimpleITK as sitk
import numpy as np
from sklearn.cluster import KMeans
import torch
from torch.cuda.amp import autocast
from collections import Counter

def extract_coordinates_and_values(input_folder):
    # 获取文件夹中的所有nrrd文件
    nrrd_files = [os.path.join(input_folder, file) for file in os.listdir(input_folder) if file.endswith('.nrrd')]
    if not nrrd_files:
        print("No nrrd files found in the input folder.")
        return

    # 读取所有nrrd文件并存储在一个列表中
    data_list = []
    for file_path in nrrd_files:
        image = sitk.ReadImage(file_path)
        data = sitk.GetArrayFromImage(image)
        data_list.append(data)

    # 读取第一个nrrd文件以获取图像尺寸和坐标信息
    first_nrrd_file_path = nrrd_files[0]
    image = sitk.ReadImage(first_nrrd_file_path)
    data = sitk.GetArrayFromImage(image)
    image_size = data.shape

    # 获取ROI区域的坐标
    roi_coordinates = np.argwhere(~np.isnan(data))

    # 创建一个列表来存储坐标及其像素值
    result_list = []

    # 遍历ROI区域的坐标
    for coord in roi_coordinates:
        z, y, x = coord
        # 创建一个列表来存储当前坐标及其在每个nrrd文件中的像素值
        coordinate_values = [(x, y, z)]
        # 从已读取的数据中获取像素值
        for data in data_list:
            coordinate_values.append(data[z, y, x])

        # 将当前坐标及其像素值添加到结果列表中
        result_list.append(coordinate_values)

    for i in range(1, len(result_list[0])):
        column = [row[i] for row in result_list]
        min_val = min(column)
        max_val = max(column)
        for j in range(len(result_list)):
            result_list[j][i] = (result_list[j][i] - min_val) / (max_val - min_val)

    return result_list

def compute_silhouette_score_gpu(pixel_vectors, cluster_labels):
    """
    使用 PyTorch 在 GPU 上计算 Silhouette 分数
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pixel_vectors_torch = torch.tensor(pixel_vectors, dtype=torch.float32, device=device)
    cluster_labels_torch = torch.tensor(cluster_labels, dtype=torch.int64, device=device)

    with autocast():
        # 计算每个点到其自身聚类中心的距离
        cluster_centers = torch.stack([pixel_vectors_torch[cluster_labels_torch == i].mean(0) for i in torch.unique(cluster_labels_torch)])
        a_i = torch.cdist(pixel_vectors_torch, cluster_centers)[torch.arange(len(pixel_vectors_torch)), cluster_labels_torch]

        # 计算每个点到其他聚类中心的最小距离
        b_i, _ = torch.min(torch.cdist(pixel_vectors_torch, cluster_centers), dim=1)

        # 计算 Silhouette 系数
        silhouette_coef = (b_i - a_i) / torch.max(a_i, b_i)
        silhouette_avg = silhouette_coef.mean().item()

    return silhouette_avg

def cluster_coordinates(data_list):
    # 提取像素值向量
    pixel_vectors = [row[1:] for row in data_list]

    # 转换为 NumPy 数组以便进行 NaN 和无穷大值处理
    pixel_vectors = np.array(pixel_vectors, dtype=np.float64)

    # 检查并处理 NaN 和无穷大值
    if np.isnan(pixel_vectors).any() or np.isinf(pixel_vectors).any():
        print("Warning: Found NaN or Inf values in pixel_vectors. Cleaning the data.")
        # 可以选择替换 NaN 值为列的均值
        col_means = np.nanmean(pixel_vectors, axis=0)
        inds = np.where(np.isnan(pixel_vectors))
        pixel_vectors[inds] = np.take(col_means, inds[1])

        # 也可以选择移除包含 NaN 或无穷大的行
        # pixel_vectors = pixel_vectors[~np.isnan(pixel_vectors).any(axis=1)]
        # pixel_vectors = pixel_vectors[~np.isinf(pixel_vectors).any(axis=1)]

    best_n_clusters = 0
    best_silhouette_score = -1

    for n_clusters in range(3, 6):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(pixel_vectors)
        silhouette_avg = compute_silhouette_score_gpu(pixel_vectors, cluster_labels)

        if silhouette_avg > best_silhouette_score:
            best_n_clusters = n_clusters
            best_silhouette_score = silhouette_avg

    # 使用最佳聚类数量进行最终聚类
    kmeans = KMeans(n_clusters=best_n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(pixel_vectors)

    # 将结果保存为坐标点和标签的字典
    coordinate_label_dict = {}
    for i, row in enumerate(data_list):
        coordinate = tuple(row[0])
        label = cluster_labels[i]
        coordinate_label_dict[coordinate] = label

    # 统计每个类别中有多少个点
    label_counts = Counter(cluster_labels)

    return coordinate_label_dict, label_counts


def save_cluster_labels(input_folder, coordinate_label_dict, output_folder):
    # 获取文件夹中的第一个 nrrd 文件
    nrrd_files = [os.path.join(input_folder, file) for file in os.listdir(input_folder) if file.endswith('.nrrd')]
    if not nrrd_files:
        print("No nrrd files found in the input folder.")
        return

    first_nrrd_file_path = nrrd_files[0]
    image = sitk.ReadImage(first_nrrd_file_path)
    image_size = image.GetSize()

    # 获取所有簇的标签
    all_labels = set(coordinate_label_dict.values())

    # 为每个簇创建一个 NumPy 数组
    cluster_arrays = {label: np.zeros(image_size[::-1], dtype=np.uint8) for label in all_labels}

    # 遍历坐标点,在对应位置上标记标签
    for coord, label in coordinate_label_dict.items():
        x, y, z = coord[::-1]  # NumPy 数组使用 (z, y, x) 索引顺序
        cluster_arrays[label][x, y, z] = 1

    # 将 NumPy 数组转换为 SimpleITK 图像并保存
    for label, cluster_array in cluster_arrays.items():
        cluster_image_sitk = sitk.GetImageFromArray(cluster_array)
        cluster_image_sitk.CopyInformation(image)
        output_file = os.path.join(output_folder, f"cluster_{label}.nrrd")
        sitk.WriteImage(cluster_image_sitk, output_file)


patient_dir = os.path.abspath(os.path.join('file_name', os.pardir))
for patient_name in os.listdir(patient_dir):
    patient_path = os.path.join(patient_dir, patient_name)
    if os.path.isdir(patient_path) and patient_name not in ['.idea', 'exampleSettings', 'H_radiomics','3D reconstuction', 'logistic regression']:
        input_folder = os.path.join(patient_path, 'feature_map')
        data_list = extract_coordinates_and_values(input_folder)
        coordinate_label_dict, label_counts = cluster_coordinates(data_list)
        # 打印总点数和每个类别的点数
        total_points = sum(label_counts.values())
        print(f"Calculating cluster for case: {patient_name}")
        print(f"Total number of points: {total_points}")
        for label, count in label_counts.items():
            print(f"Number of points in class {label}: {count}")
        output_folder = os.path.join(patient_path, 'cluster_labels')
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        save_cluster_labels(input_folder, coordinate_label_dict, output_folder)


