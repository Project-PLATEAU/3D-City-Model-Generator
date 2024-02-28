import cv2
import glob
import numpy as np

from tqdm import tqdm
from pleatau_inference import read_tif,write_img
from multiprocessing import Pool

def make_pairs(image,img_id):
    # 读入图片
    # image = cv2.imread(src_path,cv2.IMREAD_GRAYSCALE)

    # 确保图像是二值化的
    _, binary = cv2.threshold(image, 2, 255, cv2.THRESH_BINARY)

    # 寻找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建一个全黑的背景
    outline_mask = np.zeros_like(image)
    outline = np.zeros_like(image)

    dst_in='/fast/zcb/data/PLATEAU_obj/obj_tiff/in/'+img_id+'.png'
    dst_out='/fast/zcb/data/PLATEAU_obj/obj_tiff/out/'+img_id+'.png'
    dst_mask = '/fast/zcb/data/PLATEAU_obj/obj_tiff/mask/'+img_id+'.png'

    cv2.drawContours(outline, contours, -1, (255), 1)
    cv2.imwrite(dst_in, outline)

    cv2.drawContours(outline_mask, contours, -1, (255), -1)
    cv2.imwrite(dst_mask, outline_mask)

    mask = np.zeros(image.shape, dtype=np.uint8)
    mask[(image == 0) & (outline_mask != 0)] = 255

    inpainted_image = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    cv2.imwrite(dst_out, inpainted_image)


def singleProc(path):
    for i in tqdm(range(len(path))):
        _, _, data = read_tif(path[i])

        zmin, zmax = np.min(data), np.max(data)
        normalized_res = 255 * (data - zmin) / (zmax - zmin)

        normalized_res = normalized_res.astype(int)
        normalized_res[normalized_res > 255] = 255
        normalized_res[normalized_res < 0] = 0

        img_id = path[i].split('/')[-1][:-5]
        make_pairs(normalized_res.astype(np.uint8), img_id)

        # dst_path='/fast/zcb/data/pleatau/data/obj_sample/image/'+path[i].split('/')[-1][:-5]+'.png'
        # cv2.imwrite(dst_path,normalized_res)

paths=glob.glob('/fast/zcb/data/PLATEAU_obj/obj_tiff/obj_converted_tiff/**/*.tiff',recursive=True)
l = len(paths)
n_cpu = 50
pos = np.linspace(0,l,n_cpu+1).astype(int)
pos[-1] = l

pool = Pool()
proc = []
for i in range(n_cpu):
    p = paths[pos[i]:pos[i + 1]]
    proc.append(pool.apply_async(singleProc, args=(p,)))

for j in tqdm(proc):
    j.get()
