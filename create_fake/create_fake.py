import os 
import numpy as np
import pandas as pd 
import pickle
from tqdm import tqdm
import pydicom
import shutil
import copy

metadata_path = "/media/tuan/Data1/DATA_PET/norm_in_img_full_petct_npz/metadata/pet_case_data.pkl"
# predict_dir = "/media/tuan/Data1/img2img_miccai2022/research_checkpoints/pix2pix_cyclegan/ct2pet_pix2pix_lsgan_unet256_70x70PatchGan_norm_in_img_headneck_npzdata_full_no_mask_training/predict"
out_dir = "/home/tuan/linh/img2img_miccai2022/data/data/norm_in_img_full_petct_npz/"
predict_dir = "/home/tuan/linh/img2img_miccai2022/data/data/norm_in_img_full_petct_npz/pet_test"
fake_dicom_out_dir = os.path.join(out_dir, "./fake_dicom/")
origin_dicom_out_dir = os.path.join(out_dir, "./origin_dicom/")

PET_MAX_VALUE = 32767 # Some file is just 32766
PET_MIN_VALUE = 0
RESCALE_SLOPE_MAX = 20.0

def get_uid_from_path(path):
    return path.split('__')[-1].split("_")[0]

def get_index_from_predict_path(path):
    return int(path.split('__')[-1].split('_')[-1].replace('.npz', ''))

def get_all_uid_predict(data_dir):
    uids = set()
    for filename in os.listdir(data_dir):
        uid = get_uid_from_path(filename)
        uids.add(uid)
    return list(uids)
        
def get_metadata_by_uid(uid):
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f) 
    
    return metadata[uid][1]# metadata was stored in position 1

def get_metadata_of_case_by_index(idx, metadata_case):
    keys = list(metadata_case.keys())
    k = keys[idx]
    return metadata_case[k]

def get_dicom_path_by_index(idx, metadata_case):
    """
    Key of metadata_case is the path of origin dicom file
    """
    keys = list(metadata_case.keys())
    k = keys[idx]
    return k

def unnormalize_data(img):
    print("Max value before unnorm ", img.max())
    img = img / 255.0 # covert to 0-1 value range
    img = (img * (PET_MAX_VALUE * RESCALE_SLOPE_MAX - PET_MIN_VALUE * RESCALE_SLOPE_MAX) + PET_MIN_VALUE * RESCALE_SLOPE_MAX)
    rescale_slope = img.max() / PET_MAX_VALUE
    rescale_intercept = 0.0
    img = (img / rescale_slope).astype(np.int16)
    return img, rescale_intercept, rescale_slope

def unnormalize_data_with_origin_dicom(img, dataset):
    print(img.max(), img.min(), dataset.pixel_array.max(), dataset.pixel_array.min(), dataset.RescaleSlope, dataset.RescaleIntercept)
    origin_value = dataset.pixel_array * dataset.RescaleSlope + dataset.RescaleIntercept
    o_max_value = origin_value.max()
    o_min_value = origin_value.min()
    print(o_max_value, o_min_value)
    # img = img / 255.0
    img = (img * (o_max_value - o_min_value)) + o_min_value
    img = (img - dataset.RescaleIntercept) / dataset.RescaleSlope
    return img.astype(np.int16)
    # return img

def create_fake_dicom(predict_path):
    name = predict_path.split("/")[-1]
    uid = get_uid_from_path(predict_path)
    idx = get_index_from_predict_path(predict_path)
    metadata_case = get_metadata_by_uid(uid)
    # print(metadata_case)
    dicom_path = get_dicom_path_by_index(idx, metadata_case)
    print(uid, idx, dicom_path)

    img = np.load(predict_path)['data']
    # img, rescale_intercept, rescale_slope = unnormalize_data(img)

    ds = pydicom.dcmread(dicom_path)
    img = unnormalize_data_with_origin_dicom(img, copy.deepcopy(ds))

    print(np.abs(img - ds.pixel_array).max())

    ds.PixelData = pydicom.encaps.encapsulate([img.tobytes()])
    ds['PixelData'].is_undefined_length = True
    # print(rescale_slope, rescale_intercept, ds.RescaleSlope, ds.RescaleIntercept, type(ds.RescaleSlope), ds.RescaleIntercept)
    # ds.RescaleSlope = str(rescale_slope)
    # ds.RescaleIntercept = str(rescale_intercept)
    # print("After ", ds.RescaleIntercept, ds.RescaleSlope, type(ds.RescaleIntercept), type(ds.RescaleSlope))

    fake_dicom_name = name.replace(".npz", '.dcm')
    fake_dicom_name = f"fake_{fake_dicom_name}"
    out_dicom_fake_dir = os.path.join(fake_dicom_out_dir, uid)
    origin_dicom_dir = os.path.join(origin_dicom_out_dir, uid)
    os.makedirs(origin_dicom_dir, exist_ok=True)
    os.makedirs(out_dicom_fake_dir, exist_ok=True)
    out_dicom_fake_path = os.path.join(out_dicom_fake_dir, fake_dicom_name)
    origin_dicom_path = os.path.join(origin_dicom_dir, dicom_path.split("/")[-1])
#             print(out_dicom_fake_path)
    ds.save_as(out_dicom_fake_path, write_like_original=False)
    shutil.copy(dicom_path, origin_dicom_path)


# def get_all_pairs

if __name__ == "__main__":
    os.makedirs(fake_dicom_out_dir, exist_ok=True)
    os.makedirs(origin_dicom_out_dir, exist_ok=True)
    uids = get_all_uid_predict(predict_dir)
    # print(uids)
    metadata_1 = get_metadata_by_uid(uids[0])
    # print(metadata_1)
    for filename in tqdm(os.listdir(predict_dir)):
        predict_path = os.path.join(predict_dir, filename)
        create_fake_dicom(predict_path)
        # break
