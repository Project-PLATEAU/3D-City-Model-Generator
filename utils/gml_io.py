from lxml import etree
import numpy as np

def save_citygml(root, file_name):
    tree = etree.ElementTree(root)
    tree.write(file_name, pretty_print=True, xml_declaration=True, encoding='UTF-8')
    

def bldg_citygml(vertices, faces, vertex_num=0, lod=2, srs_name="http://www.opengis.net/def/crs/EPSG/0/30169", srsDimension="3"):
    assert lod in [1, 2]

    # print(vertices, faces, len(vertices), len(faces))
    # Header
    nsmap = {
        'core': "http://www.opengis.net/citygml/2.0",
        'bldg': "http://www.opengis.net/citygml/building/2.0",
        'gml': "http://www.opengis.net/gml"
    }
    cityModel = etree.Element("{http://www.opengis.net/citygml/2.0}CityModel", nsmap=nsmap)

    # bounding
    # total_vertices = []
    # for building in np.array(vertices):
    #     total_vertices.extend(building)
    total_vertices = np.vstack(vertices)
    x_max, y_max, z_max = np.max(total_vertices, axis=0)
    x_min, y_min, z_min = np.min(total_vertices, axis=0)

    boundedBy = etree.SubElement(cityModel, "{http://www.opengis.net/gml}boundedBy")
    Envelope = etree.SubElement(boundedBy, "{http://www.opengis.net/gml}Envelope", srsName=srs_name,
                                srsDimension=srsDimension)
    lowerCorner = etree.SubElement(Envelope, "{http://www.opengis.net/gml}lowerCorner")
    upperCorner = etree.SubElement(Envelope, "{http://www.opengis.net/gml}upperCorner")
    lowerCorner.text = '{} {} {}'.format(x_min, y_min, z_min)
    upperCorner.text = '{} {} {}'.format(x_max, y_max, z_max)

    # geometry
    if lod == 1:
        for vs, fs in zip(vertices, faces):
            building_member = etree.SubElement(cityModel, "{http://www.opengis.net/citygml/2.0}cityObjectMember")
            building = etree.SubElement(building_member, "{http://www.opengis.net/citygml/building/2.0}Building")

            lod1Solid = etree.SubElement(building, "{http://www.opengis.net/citygml/building/2.0}lod1Solid")
            solid = etree.SubElement(lod1Solid, "{http://www.opengis.net/gml}Solid")
            exterior = etree.SubElement(solid, "{http://www.opengis.net/gml}exterior")
            compositeSurface = etree.SubElement(exterior, "{http://www.opengis.net/gml}CompositeSurface")

            for f in fs:
                surfaceMember = etree.SubElement(compositeSurface, "{http://www.opengis.net/gml}surfaceMember")
                polygon = etree.SubElement(surfaceMember, "{http://www.opengis.net/gml}Polygon")
                exterior = etree.SubElement(polygon, "{http://www.opengis.net/gml}exterior")
                linearRing = etree.SubElement(exterior, "{http://www.opengis.net/gml}LinearRing")
                posList = etree.SubElement(linearRing, "{http://www.opengis.net/gml}posList")

                # print(len(vs), f)
                coords = ' '.join(
                    ['{} {} {}'.format(vs[idx - vertex_num - 1][0], vs[idx - vertex_num - 1][1], vs[idx - vertex_num - 1][2]) for idx in f]
                )
                coords += ' {} {} {}'.format(vs[f[0] - vertex_num - 1][0], vs[f[0] - vertex_num - 1][1], vs[f[0] - vertex_num - 1][2])
                posList.text = coords

            vertex_num = vertex_num + len(vs)
    
    elif lod == 2:
        for vs, fs in zip(vertices, faces):
            vs = np.array(vs)
            z_min, z_max = np.min(vs[:, 2]), np.max(vs[:, 2])
            
            building_member = etree.SubElement(cityModel, "{http://www.opengis.net/citygml/2.0}cityObjectMember")
            building = etree.SubElement(building_member, "{http://www.opengis.net/citygml/building/2.0}Building")
            
            measuredHeight = etree.SubElement(building,
                                                  "{http://www.opengis.net/citygml/building/2.0}measuredHeight")
            measuredHeight.text = str(round(z_max - z_min, 2))

            for f in fs:
                boundedBy = etree.SubElement(building,
                                                 "{http://www.opengis.net/citygml/building/2.0}boundedBy")
                zf = [idx - vertex_num for idx in f]
                print(zf)
                z_face = vs[zf][:, 2]
                if (z_face - z_min < 1.).all():
                    typeSurface = etree.SubElement(boundedBy,
                                                    "{http://www.opengis.net/citygml/building/2.0}GroundSurface")
                elif (z_face - z_min > 1.).all():
                    typeSurface = etree.SubElement(boundedBy,
                                                    "{http://www.opengis.net/citygml/building/2.0}RoofSurface")
                else:
                    typeSurface = etree.SubElement(boundedBy,
                                                    "{http://www.opengis.net/citygml/building/2.0}WallSurface")

                lod2MultiSurface = etree.SubElement(typeSurface,
                                                    "{http://www.opengis.net/citygml/building/2.0}lod2MultiSurface")
                MultiSurface = etree.SubElement(lod2MultiSurface, "{http://www.opengis.net/gml}MultiSurface")
                surfaceMember = etree.SubElement(MultiSurface, "{http://www.opengis.net/gml}surfaceMember")
                polygon = etree.SubElement(surfaceMember, "{http://www.opengis.net/gml}Polygon")
                exterior = etree.SubElement(polygon, "{http://www.opengis.net/gml}exterior")
                linearRing = etree.SubElement(exterior, "{http://www.opengis.net/gml}LinearRing")
                posList = etree.SubElement(linearRing, "{http://www.opengis.net/gml}posList")

                # print(len(vs), f)
                coords = ' '.join(
                    ['{} {} {}'.format(vs[idx - vertex_num - 1][0], vs[idx - vertex_num - 1][1], vs[idx - vertex_num - 1][2]) for idx in f]
                )
                coords += ' {} {} {}'.format(vs[f[0] - vertex_num - 1][0], vs[f[0] - vertex_num - 1][1], vs[f[0] - vertex_num - 1][2])
                posList.text = coords

            # vertex_num = vertex_num + len(vs)

    else:
        raise Exception("Building error: Only lod 1 and 2 is implemented. ")

    return cityModel
