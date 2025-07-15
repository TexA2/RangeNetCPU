#Сферическая проекция позволяет визуализировать данные, имитируя виртуальную 360°-сканирующую станцию. Процесс включает четыре шага:
    #Загрузка 3D-облака точек.
    #Проецирование каждой точки на сферу.
    #Определение геометрии для извлечения пикселей.
    #"Развертывание" геометрии для создания изображения.

import open3d as o3d
import numpy as np
import cv2


def generate_spherical_image(center_coordinates, point_cloud, colors, resolution_y=500):

    translated_points = point_cloud - center_coordinates
    
    # Преобразование в сферические координаты (θ, φ)
    theta = np.arctan2(translated_points[:, 1], translated_points[:, 0])  # Азимутальный угол
    phi = np.arccos(translated_points[:, 2] / np.linalg.norm(translated_points, axis=1))  # Зенитный угол
    
    # Преобразование в координаты пикселей
    resolution_x = 2 * resolution_y  # Ширина = 2 × высота (эквиректангулярная проекция)

    x = (theta + np.pi) / (2 * np.pi) * resolution_x  # Нормализация θ ∈ [0, 2π] → [0, resolution_x]
    y = phi / np.pi * resolution_y  # Нормализация φ ∈ [0, π] → [0, resolution_y]
    
    # Создание пустого изображения и карты соответствий
    image = np.zeros((resolution_y, resolution_x, 3), dtype=np.float64)
    mapping = np.full((resolution_y, resolution_x), -1, dtype=int)  # Инициализация "пустыми" значениями
    
    # Назначение точек пикселям изображения
    for i in range(len(translated_points)):
        ix = np.clip(int(x[i]), 0, resolution_x - 1)
        iy = np.clip(int(y[i]), 0, resolution_y - 1)
        
        # Сохранение ближайшей точки (чтобы избежать перекрытий)
        if mapping[iy, ix] == -1 or np.linalg.norm(translated_points[i]) < np.linalg.norm(translated_points[mapping[iy, ix]]):
            mapping[iy, ix] = i
            image[iy, ix] = colors[i]
    
    return image, mapping




pcd = o3d.io.read_point_cloud("cloud.pcd")
points_array = np.asarray(pcd.points)



# Параметры проекции
resolution = 500
center_coordinates = [0, 0, 1]  # Точка виртуального сканирования

custom_colors = np.zeros((len(points_array), 3), dtype=np.uint8)

z_min, z_max = np.min(points_array[:, 2]), np.max(points_array[:, 2])
normalized_z = (points_array[:, 2] - z_min) / (z_max - z_min)

custom_colors[:, 0] = normalized_z * 255          # R (от 0 до 255)
custom_colors[:, 1] = (1 - normalized_z) * 255    # G (инвертировано)
custom_colors[:, 2] = 128                        # B (фиксированный)

pcd_image, pcd_mapping = generate_spherical_image(center_coordinates, points_array, custom_colors, resolution)

#pcd_two = o3d.geometry.PointCloud()
#pcd_two.points = o3d.utility.Vector3dVector(pcd_mapping)
#o3d.io.write_point_cloud("cloud2.pcd", pcd_two)


height, width = pcd_image.shape[:2]
xx, yy = np.meshgrid(np.arange(width), np.arange(height))
points = np.column_stack((xx.flatten(), yy.flatten(), np.zeros(height * width)))
colors = pcd_image.reshape(-1, 3) / 255.0  # Нормализация цветов [0,1]

pcd_from_image = o3d.geometry.PointCloud()
pcd_from_image.points = o3d.utility.Vector3dVector(points)
pcd_from_image.colors = o3d.utility.Vector3dVector(colors)
#o3d.io.write_point_cloud("cloud2.pcd", pcd_two)
o3d.io.write_point_cloud("cloud2.pcd", pcd_from_image)




#pcd_2d = o3d.geometry.PointCloud()
#pcd_2d.points = o3d.utility.Vector3dVector(projection)
#o3d.visualization.draw_geometries([pcd_2d])
#o3d.io.write_point_cloud("cloud2.pcd", pcd_2d)


cv2.imshow("Projection", pcd_image)
cv2.waitKey(0)
cv2.destroyAllWindows()