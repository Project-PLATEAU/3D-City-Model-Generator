import geopandas as gpd
from shapely.geometry import Polygon, LineString, Point
from shapely.affinity import rotate
from shapely.validation import make_valid
import numpy as np


def calculate_main_direction(polygon):
    """
    计算多边形的主方向。

    :param polygon: Shapely 的多边形对象
    :return: 主方向角度（相对于水平轴的逆时针角度，单位为度）
    """
    min_rect = polygon.minimum_rotated_rectangle
    coords = np.array(min_rect.exterior.coords[:-1])  # 最小外接矩形的顶点

    # 计算最长边的方向
    edge_vectors = coords[1:] - coords[:-1]
    edge_lengths = np.linalg.norm(edge_vectors, axis=1)
    longest_edge_index = np.argmax(edge_lengths)
    main_edge = edge_vectors[longest_edge_index]

    angle = np.arctan2(main_edge[1], main_edge[0]) * 180 / np.pi  # 转换为度数
    return angle


def douglas_peucker(coords, epsilon):
    """
    实现 Douglas-Peucker 折线简化算法。

    :param coords: 输入的坐标列表 [(x1, y1), (x2, y2), ...]
    :param epsilon: 距离阈值
    :return: 简化后的坐标列表
    """
    if len(coords) < 3:
        return coords

    # 找到距离起点和终点连线最远的点
    start, end = coords[0], coords[-1]
    line = LineString([start, end])
    max_distance = 0
    index = 0

    for i in range(1, len(coords) - 1):
        point = coords[i]
        distance = line.distance(Point(point))
        if distance > max_distance:
            max_distance = distance
            index = i

    # 如果最大距离大于阈值，则递归处理
    if max_distance > epsilon:
        left = douglas_peucker(coords[:index + 1], epsilon)
        right = douglas_peucker(coords[index:], epsilon)
        return left[:-1] + right
    else:
        return [start, end]


def enforce_right_angles(coords):
    """
    调整多边形的角点以尽量保证每个角为 90 度。

    :param coords: 输入的坐标列表 [(x1, y1), (x2, y2), ...]
    :return: 调整后的坐标列表
    """
    adjusted_coords = [coords[0]]

    for i in range(1, len(coords) - 1):
        prev_point = adjusted_coords[-1]
        curr_point = coords[i]
        next_point = coords[i + 1]

        # 计算向量
        vector1 = np.array(curr_point) - np.array(prev_point)
        vector2 = np.array(next_point) - np.array(curr_point)

        # 判断是否接近 90 度
        dot_product = np.dot(vector1, vector2)
        magnitude1 = np.linalg.norm(vector1)
        magnitude2 = np.linalg.norm(vector2)
        cos_angle = dot_product / (magnitude1 * magnitude2)

        # 如果角度不是 90 度，调整当前点
        if not np.isclose(cos_angle, 0, atol=0.1):
            if abs(vector1[0]) > abs(vector1[1]):
                adjusted_point = (curr_point[0], prev_point[1])
            else:
                adjusted_point = (prev_point[0], curr_point[1])
            adjusted_coords.append(adjusted_point)
        else:
            adjusted_coords.append(curr_point)

    adjusted_coords.append(coords[-1])  # 闭合多边形
    return adjusted_coords


def simplify_polygon_with_dp_and_angles(polygon, epsilon):
    """
    使用 Douglas-Peucker 算法简化多边形，并尽量保证每个角为 90 度。

    :param polygon: 输入的 Shapely 多边形
    :param epsilon: 距离阈值
    :return: 简化后的多边形
    """
    if polygon.is_empty or not polygon.is_valid:
        return polygon

    coords = list(polygon.exterior.coords)
    simplified_coords = douglas_peucker(coords, epsilon)
    adjusted_coords = enforce_right_angles(simplified_coords)

    if len(adjusted_coords) < 3:
        return polygon  # 无法构成多边形，返回原始多边形

    simplified_polygon = Polygon(adjusted_coords)

    # 检查多边形是否有效，并自动修复
    if not simplified_polygon.is_valid:
        simplified_polygon = make_valid(simplified_polygon)

    return simplified_polygon


def snap_to_main_direction(polygon, grid_size=1, epsilon=1):
    """
    将多边形规则化为与主方向平行的简化形状，并尽量保证每个角为 90 度。

    :param polygon: Shapely 的多边形对象
    :param grid_size: 网格大小，控制对齐精度
    :param epsilon: Douglas-Peucker 算法的距离阈值
    :return: 规则化后的多边形
    """
    if polygon.is_empty or not polygon.is_valid:
        return polygon

    # 计算主方向并旋转到水平
    main_angle = calculate_main_direction(polygon)
    rotated_polygon = rotate(polygon, -main_angle, origin='centroid', use_radians=False)

    # 使用 Douglas-Peucker 算法简化并调整角度
    simplified_polygon = simplify_polygon_with_dp_and_angles(rotated_polygon, epsilon)

    # 将规则化后的多边形旋转回原始方向
    final_polygon = rotate(simplified_polygon, main_angle, origin='centroid', use_radians=False)

    # 检查最终多边形是否有效，并自动修复
    if not final_polygon.is_valid:
        final_polygon = make_valid(final_polygon)

    return final_polygon


def regularize_geodataframe(gdf, grid_size=1, epsilon=1):
    """
    对 GeoDataFrame 中的所有几何对象进行规则化处理。

    :param gdf: 输入的 GeoDataFrame
    :param grid_size: 网格大小，控制对齐精度
    :param epsilon: Douglas-Peucker 算法的距离阈值
    :return: 规则化后的 GeoDataFrame
    """
    # 确保输入 GeoDataFrame 有 'geometry' 列
    if 'geometry' not in gdf.columns:
        raise ValueError("GeoDataFrame must have a 'geometry' column.")

    # 复制输入数据，防止修改原数据
    gdf_regularized = gdf.copy()

    # 应用规则化函数到每个几何对象
    gdf_regularized['geometry'] = gdf_regularized['geometry'].apply(
        lambda geom: snap_to_main_direction(geom, grid_size, epsilon) if geom else geom
    )

    return gdf_regularized
