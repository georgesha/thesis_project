import os
import glob
import pydicom
import cv2
import numpy as np
from multiprocessing import Pool

def remove_extension(fname):
    basename=os.path.basename(fname).split('.')[0]
    return basename

path = ''
pa_OUT_PATH = ''
lat_OUT_PATH = ''
dicom_folders = glob.glob(path)
pa_png_path = ''
lat_png_path = ''
pa_png_set = set(map(remove_extension, glob.glob(pa_png_path)))
lat_png_set = set(map(remove_extension, glob.glob(lat_png_path)))
lut_tags = ['PhotometricInterpretation', 'WindowCenter', 'WindowWidth']

def get_lut_tags(ds):
    lut = dict.fromkeys(lut_tags, None)

    for tag in lut_tags:
        try:
            lut[tag] = ds.data_element(tag).value

        except Exception as e:
            print(e)
    return lut


def process_img(ds):

    # Windowing Function implemented from
    # https://www.dabsoft.ch/dicom/3/C.11.2.1.2/

    lut = get_lut_tags(ds)
    print(lut)

    try:
        img = ds.pixel_array
    except:
        return None

    if lut['WindowCenter'] and lut['WindowWidth']:
        try:
            print("go through")
            wc = int(lut['WindowCenter'])
            ww = int(lut['WindowWidth'])

            img = np.array(img)

            img = ((img - (float(wc) - 0.5)) / (float(ww) - 1.0) + 0.5) * 255
            img[img > 255] = 255
            img[img < 0] = 0


        except Exception as e:

            print(e, 'Window Center or Window Width not an integer')
            pass
    else:
        print("not go through")
        max = np.max(img)
        img = (img / max * 255)

    img = img.astype(np.uint8)
    img = np.stack((img,) * 3, axis=-1)

    if lut['PhotometricInterpretation'] == 'MONOCHROME1':
        img = cv2.bitwise_not(img)

    return img

def create_png(dicom_folders):
    for folder in dicom_folders:
        dicom_list = glob.glob(os.path.join(folder, '*dcm'))
        for dcm in dicom_list:
            # remove dicom with small size
            size = os.path.getsize(dcm)
            if size < 2 * 1024 * 1024:
                continue
            ds = pydicom.dcmread(dcm)

            # get rid of dicom with no image
            try:
                vp = ds.ViewPosition
            except Exception as e:
                print("No image")

            # get rid of not chest dicom
            if 'chest' not in ds.SeriesDescription.lower():
                continue

            # pa image
            if 'pa' in ds.SeriesDescription.lower() and 'lat' not in ds.SeriesDescription.lower() and remove_extension(dcm) not in pa_png_set:
                filename = os.path.basename(os.path.dirname(dcm))+'.png'
                out_png = os.path.join(pa_OUT_PATH, filename)
                try:
                    dicom_image = process_img(ds)
                    dicom_image = cv2.resize(dicom_image, (500, 500))
                    cv2.imwrite(out_png, dicom_image)
                    size = os.path.getsize(out_png)
                    if size < 50 * 1024:
                        os.remove(png)
                except:
                    print("Error")
            # lateral image
            elif 'pa' not in ds.SeriesDescription.lower() and 'lat' in ds.SeriesDescription.lower() and remove_extension(dcm) not in lat_png_set:
                filename = os.path.basename(os.path.dirname(dcm))+'.png'
                out_png = os.path.join(lat_OUT_PATH, filename)
                try:
                    dicom_image = process_img(ds)
                    dicom_image = cv2.resize(dicom_image, (500, 500))
                    cv2.imwrite(out_png, dicom_image)
                    size = os.path.getsize(out_png)
                    if size < 50 * 1024:
                        os.remove(png)
                except:
                    print("Error")
def chunks(l, amount):
    if amount == 1:
        return l
    if amount < 1:
        raise ValueError('amount must be positive integer')
    chunk_len = len(l) // amount
    leap_parts = len(l) % amount
    remainder = amount // 2  # make it symmetrical
    i = 0
    result = []
    while i < len(l):
        remainder += leap_parts
        end_index = i + chunk_len
        if remainder >= amount:
            remainder -= amount
            end_index += 1
        result.append(l[i:end_index])
        i = end_index
    return result

folder_list = chunks(dicom_folders, 40)
p = Pool(40)
p.map(create_png, folder_list)
