import open3d as o3d
import numpy as np
import cv2


def cloud_to_image(pcd_np, resolution):
    minx = np.min(pcd_np[:, 0])
    maxx = np.max(pcd_np[:, 0])
    miny = np.min(pcd_np[:, 1])
    maxy = np.max(pcd_np[:, 1])
    width = int((maxx - minx) / resolution) + 1
    height = int((maxy - miny) / resolution) + 1
    image = np.zeros((height, width, 3), dtype=np.float64)
    for point in pcd_np:
        x, y, *_ = point
        r, g, b = point[-3:]
        pixel_x = int((x - minx) / resolution)
        pixel_y = int((maxy - y) / resolution)
        image[pixel_y, pixel_x] = [r, g, b]
    return image


def project_to_2d(pcd_np):
    """Проецирует 3D-точки на 2D-плоскость (ортогональная проекция)."""
    minx, miny = np.min(pcd_np[:, :2], axis=0)
    maxx, maxy = np.max(pcd_np[:, :2], axis=0)
    
    # Нормализуем координаты X, Y в диапазон [0, 1]
    normalized_x = (pcd_np[:, 0] - minx) / (maxx - minx)  * 250
    normalized_y = (pcd_np[:, 1] - miny) / (maxy - miny)  * 250
    
    # Создаём 2D-точки (Z=0)
    points_2d = np.column_stack([normalized_x, normalized_y, np.zeros_like(normalized_x)])
    
    return points_2d


pcd = o3d.io.read_point_cloud("cloud.pcd")
#o3d.visualization.draw_geometries([pcd])

points_array = np.asarray(pcd.points)

pcd_load = cloud_to_image(points_array, 0.1)

#pcd_two = o3d.geometry.PointCloud()
#pcd_two.points = o3d.utility.Vector3dVector(pcd_load)
#o3d.visualization.draw_geometries([pcd_two])

projection = project_to_2d(points_array)

pcd_2d = o3d.geometry.PointCloud()
pcd_2d.points = o3d.utility.Vector3dVector(projection)
#o3d.visualization.draw_geometries([pcd_2d])
o3d.io.write_point_cloud("cloud2.pcd", pcd_2d)


#cv2.imshow("Projection", pcd_load)
#cv2.waitKey(0)
#cv2.destroyAllWindows()