# gen3D_SVI

Generate .obj and CityGML (.gml) files for real city 3D scenes, used street view images (SVI) as reference information.

# Usage

Environment: Windows OS, CPU, Python
Lib:
```
numpy
pandas
trimesh
geopandas
earcut
lxml
shapely
pyproj
```

Run:
```
python gen3D_SVI.py
```

# Data and result samples

### SVI
![SVI](./sample_img/sample2.png)

### Point cloud and Semantic segmantation (intermediate data)
![PCL](./sample_img/sample3.png)
![SS](./sample_img/sample4.png)

### Real 3D city model (final results)
![SVI3D](./sample_img/sample1.png)


