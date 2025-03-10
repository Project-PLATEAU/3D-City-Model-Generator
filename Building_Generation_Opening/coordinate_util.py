import numpy as np 
from pyproj import Transformer

def convert_WGS84_to_ECEF(lat, lon, hei):
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = f*(2.0 - f)

    b = np.pi * lat / 180.0
    l = np.pi * lon / 180.0

    N = a / np.sqrt(1.0 - e2 * np.power(np.sin(b), 2))

    x = (N + hei) * np.cos(b) * np.cos(l)
    y = (N + hei) * np.cos(b) * np.sin(l)
    z = (N * (1.0 - e2) + hei) * np.sin(b)

    return (x, y, z)


def convert_ECEF_to_WGS84(x, y, z):
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = f * (2.0 - f)

    p = np.sqrt(x * x + y * y)
    r = np.sqrt(p * p + z * z)
    mu = np.arctan(z / p * ((1.0 - f) + e2 * a / r))

    B = np.arctan((z * (1.0 - f) + e2 * a * np.power(np.sin(mu), 3)) / ((1.0 - f) * (p - e2 * a * np.power(np.cos(mu), 3))))

    lat = 180.0 * B / np.pi
    lon = 180.0 * np.arctan2(y, x) / np.pi
    hei = p * np.cos(B) + z * np.sin(B) - a * np.sqrt(1.0 - e2 * np.power(np.sin(B), 2))

    return (lat, lon, hei)

def convert_WGS84_to_Japan_Rect_30169(lat, lon):
    crs_transformer = Transformer.from_crs(4326, 30169, always_xy=True)
    return crs_transformer.transform(lon, lat)

def convert_mercator_to_Japan_Rect_30169(x, y):
    crs_transformer = Transformer.from_crs(3857, 30169, always_xy=True)
    return crs_transformer.transform(x, y)
