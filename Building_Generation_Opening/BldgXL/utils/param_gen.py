import random, os
import pandas as pd


def random_param(bldg_num: int, 
                 sod_min: int = 1, 
                 sod_max: int = 3, 
                 height_min: float = 5.0, 
                 height_max: float = 20.0, 
                 type_selected: list[int] = [1, 2, 3, 4, 5], 
                 output_dir: str = '../', 
                 output_csv = True):
    type_check = any(x < 1 or x > 5 for x in type_selected)
    assert type_check == False, "Invalid type identifiers included. "
    
    id_list = [i for i in range(bldg_num)]
    sod_list = [random.randint(sod_min, sod_max) for _ in range(bldg_num)]
    height_list = [round(random.uniform(height_min, height_max), 2) for _ in range(bldg_num)]
    type_list = random.choices(type_selected, k=bldg_num)
    
    params = pd.DataFrame({
        'Bldg_ID': id_list, 
        'SoD': sod_list, 
        'Height': height_list, 
        'Roof_type': type_list
    })
    
    if output_csv:
        params.to_csv(output_dir + '/params_group2.csv', index=False)
    else:
        return params
    
    
if __name__ == '__main__':
    random_param(63, 'temp_param')
    