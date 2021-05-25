import os

import pandas as pd


data_dir = os.path.join('.', 'data')

catId = '/m/01mqdt'
class_name = 'road-sign'

dirs = ['train', 'validation', 'test']

for dir_ in dirs:
    os.makedirs(os.path.join(data_dir, dir_), exist_ok=True)
    os.makedirs(os.path.join(data_dir, dir_, 'imgs'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, dir_, 'anns'), exist_ok=True)

    data = pd.read_csv(os.path.join(data_dir, 'anns-raw', '{}-annotations-bbox.csv'.format(dir_)))

    data = data[data['LabelName'] == catId]

    img_ids = data['ImageID']
    bboxes = data[['XMin', 'XMax', 'YMin', 'YMax']]

    id_list = []
    img_id_bboxes={}
    for j, id in enumerate(img_ids):
        # os.system('python downloader.py {}/{} --download_folder={} --num_processes=5'.format(
        #    dir_, id, os.path.join(data_dir, dir_, 'imgs')))
        if '{}/{}'.format(dir_, id) not in id_list:
            img_id_bboxes['{}/{}'.format(dir_, id)] = []
            id_list.append('{}/{}'.format(dir_, id))

        XMin, XMax, YMin, YMax = bboxes.iloc[j]

        w = float(XMax) - float(XMin)
        h = float(YMax) - float(YMin)
        xc = (float(XMax) + float(XMin)) / 2
        yc = (float(YMax) + float(YMin)) / 2

        img_id_bboxes['{}/{}'.format(dir_, id)].append('{} {} {} {} {}'.format(str(0), str(xc), str(yc), str(w),
                                                                               str(h)))

    with open('./image_list_{}'.format(dir_), 'w') as f:
        for id in id_list:
            with open(os.path.join(data_dir, dir_, 'anns', '{}.txt'.format(id.split('/')[1])), 'w') as f_ann:
                for det in img_id_bboxes[id]:
                    f_ann.write('{}\n'.format(det))
                f_ann.close()
            f.write('{}\n'.format(id))

        f.close()
