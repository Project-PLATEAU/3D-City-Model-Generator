import json
import os
import numpy as np


class plateau_opening():
    def __init__(self, 
                 eid, 
                 openings 
                 ):
        self.edge_id = eid
        self.openings = openings  # list of dicts
        
class plateau_building():
    def __init__(self, 
                 bldg_id, 
                 height, 
                 norm, 
                 footprint, 
                 openings
                 ):
        self.bldg_id = bldg_id
        self.height = height
        self.footprint_orientation = np.array(norm)  # vector
        self.footprint = np.array(footprint)  # list
        self.opening = [plateau_opening(i['eid'], i['elems']) for i in openings] # list
        
class plateau_route():
    def __init__(self, 
                 json_path):
        with open(json_path, 'r') as f:
            route_data = json.load(f)
        
        self.geo_center = np.array(route_data['geo_center'])
        self.buildings = [plateau_building(building['id'], 
                                           building['height'], 
                                           building['norm'], 
                                           building['footprint'], 
                                           building['faces']) for building in route_data['bldgs']] 



    