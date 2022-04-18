import argparse
import os

import fiftyone as fo
from load_samples import load_samples, create_fo_dataset


def main(params):

    img_dir = '/data/wangxinran/dataset/object_detection/{}/JPEGImages'.format(params.dataset)
    anno_dir = '/data/wangxinran/dataset/object_detection/{}/Annotations_{}'.format(params.dataset, params.num_classes)
    dataset_name = '{}_{}c'.format(params.dataset, params.num_classes)
    split_dir = '/data/wangxinran/dataset/object_detection/{}/ImageSets/{}'.format(params.dataset, params.num_classes)
    split = 'val'

    if params.operation == 'delete':
        dataset = fo.load_dataset(dataset_name)
        print(dataset.view())
        print('\nDeleting......')
        dataset.delete()

    elif params.operation == 'create':

        samples = load_samples(img_dir=img_dir,
                               anno_dir=anno_dir,
                               samples_names_file=os.path.join(split_dir, split + '.txt'))

        dataset = create_fo_dataset(samples=samples,
                                    name=dataset_name)
        print(dataset.view())
        dataset.persistent = True
    else:
        dataset = fo.load_dataset(dataset_name)
        print(dataset.view())
        session = fo.launch_app(dataset)
        session.wait()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # the model directory
    parser.add_argument('--dataset',
                        default='bdd100k',
                        type=str,
                        help="the dataset name")

    parser.add_argument('--num-classes',
                        default=1,
                        type=int,
                        help="the number of classes")

    # target domain index
    parser.add_argument('--operation',
                        type=str,
                        choices=['launch',
                                 'create',
                                 'delete'],
                        default='launch',
                        help="True to load the dataset, False for Generate the dataset.")

    params = parser.parse_args()

    main(params)
