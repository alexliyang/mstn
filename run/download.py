import hashlib
import requests
import os
import zipfile
import shutil


def md5sum(filename):
    fd = open(filename, "rb")
    fcont = fd.read()
    fd.close()
    fmd5 = hashlib.md5(fcont)
    return fmd5.hexdigest()


def get_file_path(base, file_name=''):
    file_path = os.path.join(base, file_name)
    if not os.path.exists(base):
        os.makedirs(base)

    return file_path


def un_zip(file_name, extract_dir):
    """unzip zip file"""
    zip_file = zipfile.ZipFile(file_name)
    if os.path.isdir(extract_dir):
        pass
    else:
        os.mkdir(extract_dir)
    for names in zip_file.namelist():
        zip_file.extract(names, extract_dir)
    zip_file.close()


def main():
    file_list = [
        'VGG_imagenet.npy',
        'icpr_text_train_10000.zip'
    ]
    url_list = [
        'http://ch-10035517.cossh.myqcloud.com/VGG_imagenet.npy',
        'http://ch-10035517.cossh.myqcloud.com/icpr_text_train_10000.zip'
    ]
    base = os.path.join(os.path.abspath(os.curdir), 'data')

    md5 = [
        '102f510d020773a884e76814e197170f',  # vgg16 md5
        'e7ea68b7d69b248c98328a590dc82839'
    ]
    prefix = ['pretrain', 'tmp']

    for ix, url in enumerate(url_list):
        path = get_file_path(os.path.join(base, prefix[ix]), file_list[ix])
        if os.path.exists(path) and md5sum(path) == md5[ix]:
            print('using exits file {}'.format(file_list[ix]))
        else:
            print('starting download {}'.format(file_list[ix]))
            r = requests.get(url)
            print('download file {} successful'.format(file_list[ix]))
            with open(path, "wb") as code:
                code.write(r.content)
            print('write {} file successful'.format(file_list[ix]))
        if ix == 1:
            print('starting extracting train data......')
            extract_path = get_file_path(base, 'ICPR_text_train')
            if os.path.exists(extract_path):
                shutil.rmtree(extract_path)
            os.makedirs(extract_path)

            un_zip(path, extract_path)
            print('extracting over')
            print('starting cleaning tmp file')

            shutil.rmtree(os.path.join(extract_path, '__MACOSX'))

            '''Seperate the data in train and test'''

            img_list = os.listdir(os.path.join(extract_path, 'image_10000'))
            gt_list = os.listdir(os.path.join(extract_path, 'text_10000'))

            train_num = 9000

            train_img_path = get_file_path(os.path.join(base, 'train', 'img'))
            train_txt_path = get_file_path(os.path.join(base, 'train', 'txt'))

            test_img_path = get_file_path(os.path.join(base, 'test', 'img'))
            test_txt_path = get_file_path(os.path.join(base, 'test', 'txt'))

            print('start seperate 10000 with 9000 for train and 1000 for test')

            for img_name in img_list[:train_num]:
                gt_name = os.path.splitext(img_name)[0] + '.txt'

                shutil.copy(os.path.join(extract_path, 'image_10000', img_name), os.path.join(train_img_path, img_name))
                shutil.copy(os.path.join(extract_path, 'text_10000', gt_name), os.path.join(train_txt_path, gt_name))

            for img_name in img_list[train_num:]:
                gt_name = os.path.splitext(img_name)[0] + '.txt'

                shutil.copy(os.path.join(extract_path, 'image_10000', img_name), os.path.join(test_img_path, img_name))
                shutil.copy(os.path.join(extract_path, 'text_10000', gt_name), os.path.join(test_txt_path, gt_name))
            print('data prepare complete')


if __name__ == '__main__':
    main()
