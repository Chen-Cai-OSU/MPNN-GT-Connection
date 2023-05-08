import os
import os.path as osp
import pickle
import random
import shutil

import torch
import torch.nn.functional as F
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_zip)
from tqdm import tqdm


class VOCSuperpixels(InMemoryDataset):
    r"""The VOCSuperpixels dataset which contains image superpixels and a semantic segmentation label
    for each node superpixel.
    
    Construction and Preparation:
    - The superpixels are extracted in a similar fashion as the MNIST and CIFAR10 superpixels. 
    - In VOCSuperpixels, the number of superpixel nodes <=500. (Note that it was <=75 for MNIST and
    <=150 for CIFAR10.)
    - The labeling of each superpixel node is done with the same value of the original pixel ground
    truth  that is on the mean coord of the superpixel node
    
    - Based on the SBD annotations from 11355 images taken from the PASCAL VOC 2011 dataset. Original
    source `here<https://github.com/shelhamer/fcn.berkeleyvision.org/tree/master/data/pascal>`_.
    
    num_classes = 21
    ignore_label = 255

    color map
    0=background, 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle, 6=bus, 7=car, 8=cat, 9=chair, 10=cow,
    11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person, 16=potted plant, 17=sheep, 18=sofa, 19=train,
    20=tv/monitor
    
    Splitting:
    - In the original image dataset there are only train and val splitting.
    - For VOCSuperpixels, we maintain train, val and test splits where the train set is AS IS. The original
    val split of the image dataset is used to divide into new val and new test split that is eventually used
    in VOCSuperpixels. The policy for this val/test splitting is below.
    - Split total number of val graphs into 2 sets (val, test) with 50:50 using a stratified split proportionate
    to original distribution of data with respect to a meta label.
    - Each image is meta-labeled by majority voting of non-background grouth truth node labels. Then new val
    and new test is created with stratified sampling based on these meta-labels. This is done for preserving
    same distribution of node labels in both new val and new test
    - Therefore, the final train, val and test splits are correspondingly original train (8498), new val (1428)
    and new test (1429) splits.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string, optional): Option to select the graph construction format.
            If :obj: `"edge_wt_only_coord"`, the graphs are 8-nn graphs with the edge weights computed based on
            only spatial coordinates of superpixel nodes.
            If :obj: `"edge_wt_coord_feat"`, the graphs are 8-nn graphs with the edge weights computed based on
            combination of spatial coordinates and feature values of superpixel nodes.
            If :obj: `"edge_wt_region_boundary"`, the graphs region boundary graphs where two regions (i.e. 
            superpixel nodes) have an edge between them if they share a boundary in the original image.
            (default: :obj:`"edge_wt_region_boundary"`)
        slic_compactness (int, optional): Option to select compactness of slic that was used for superpixels
            (:obj:`10`, :obj:`30`). (default: :obj:`30`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    url = {
        10: {
            'edge_wt_only_coord': 'https://www.dropbox.com/s/rk6pfnuh7tq3t37/voc_superpixels_edge_wt_only_coord.zip?dl=1',
            'edge_wt_coord_feat': 'https://www.dropbox.com/s/2a53nmfp6llqg8y/voc_superpixels_edge_wt_coord_feat.zip?dl=1',
            'edge_wt_region_boundary': 'https://www.dropbox.com/s/6pfz2mccfbkj7r3/voc_superpixels_edge_wt_region_boundary.zip?dl=1'
        },
        30: {
            'edge_wt_only_coord': 'https://www.dropbox.com/s/toqulkdpb1jrswk/voc_superpixels_edge_wt_only_coord.zip?dl=1',
            'edge_wt_coord_feat': 'https://www.dropbox.com/s/xywki8ysj63584d/voc_superpixels_edge_wt_coord_feat.zip?dl=1',
            'edge_wt_region_boundary': 'https://www.dropbox.com/s/8x722ai272wqwl4/voc_superpixels_edge_wt_region_boundary.zip?dl=1'
        }
    }

    def __init__(self, root, name='edge_wt_region_boundary', slic_compactness=30, split='train',
                 transform=None, pre_transform=None, pre_filter=None):
        self.name = name
        self.slic_compactness = slic_compactness
        assert split in ['train', 'val', 'test']
        assert name in ['edge_wt_only_coord', 'edge_wt_coord_feat', 'edge_wt_region_boundary']
        assert slic_compactness in [10, 30]
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return ['train.pickle', 'val.pickle', 'test.pickle']

    @property
    def raw_dir(self):
        return osp.join(self.root,
                        'slic_compactness_' + str(self.slic_compactness),
                        self.name,
                        'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root,
                        'slic_compactness_' + str(self.slic_compactness),
                        self.name,
                        'processed')

    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url[self.slic_compactness][self.name], self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, 'voc_superpixels_' + self.name), self.raw_dir)
        os.unlink(path)

    def process(self):
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                graphs = pickle.load(f)

            indices = range(len(graphs))

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for idx in indices:
                graph = graphs[idx]

                """
                Each `graph` is a tuple (x, edge_attr, edge_index, y)
                    Shape of x : [num_nodes, 14]
                    Shape of edge_attr : [num_edges, 1] or [num_edges, 2]
                    Shape of edge_index : [2, num_edges]
                    Shape of y : [num_nodes]
                """

                x = graph[0].to(torch.float)
                edge_attr = graph[1].to(torch.float)
                edge_index = graph[2]
                y = torch.LongTensor(graph[3])

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                            y=y)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()

            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))


class NewVOCSuperpixels(InMemoryDataset):
    r"""The VOCSuperpixels dataset which contains image superpixels and a semantic segmentation label
    for each node superpixel.

    Construction and Preparation:
    - The superpixels are extracted in a similar fashion as the MNIST and CIFAR10 superpixels.
    - In VOCSuperpixels, the number of superpixel nodes <=500. (Note that it was <=75 for MNIST and
    <=150 for CIFAR10.)
    - The labeling of each superpixel node is done with the same value of the original pixel ground
    truth  that is on the mean coord of the superpixel node

    - Based on the SBD annotations from 11355 images taken from the PASCAL VOC 2011 dataset. Original
    source `here<https://github.com/shelhamer/fcn.berkeleyvision.org/tree/master/data/pascal>`_.

    num_classes = 21
    ignore_label = 255

    color map
    0=background, 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle, 6=bus, 7=car, 8=cat, 9=chair, 10=cow,
    11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person, 16=potted plant, 17=sheep, 18=sofa, 19=train,
    20=tv/monitor

    Splitting:
    - In the original image dataset there are only train and val splitting.
    - For VOCSuperpixels, we maintain train, val and test splits where the train set is AS IS. The original
    val split of the image dataset is used to divide into new val and new test split that is eventually used
    in VOCSuperpixels. The policy for this val/test splitting is below.
    - Split total number of val graphs into 2 sets (val, test) with 50:50 using a stratified split proportionate
    to original distribution of data with respect to a meta label.
    - Each image is meta-labeled by majority voting of non-background grouth truth node labels. Then new val
    and new test is created with stratified sampling based on these meta-labels. This is done for preserving
    same distribution of node labels in both new val and new test
    - Therefore, the final train, val and test splits are correspondingly original train (8498), new val (1428)
    and new test (1429) splits.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string, optional): Option to select the graph construction format.
            If :obj: `"edge_wt_only_coord"`, the graphs are 8-nn graphs with the edge weights computed based on
            only spatial coordinates of superpixel nodes.
            If :obj: `"edge_wt_coord_feat"`, the graphs are 8-nn graphs with the edge weights computed based on
            combination of spatial coordinates and feature values of superpixel nodes.
            If :obj: `"edge_wt_region_boundary"`, the graphs region boundary graphs where two regions (i.e.
            superpixel nodes) have an edge between them if they share a boundary in the original image.
            (default: :obj:`"edge_wt_region_boundary"`)
        slic_compactness (int, optional): Option to select compactness of slic that was used for superpixels
            (:obj:`10`, :obj:`30`). (default: :obj:`30`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    url = {
        10: {
            'edge_wt_only_coord': 'https://www.dropbox.com/s/rk6pfnuh7tq3t37/voc_superpixels_edge_wt_only_coord.zip?dl=1',
            'edge_wt_coord_feat': 'https://www.dropbox.com/s/2a53nmfp6llqg8y/voc_superpixels_edge_wt_coord_feat.zip?dl=1',
            'edge_wt_region_boundary': 'https://www.dropbox.com/s/6pfz2mccfbkj7r3/voc_superpixels_edge_wt_region_boundary.zip?dl=1'
        },
        30: {
            'edge_wt_only_coord': 'https://www.dropbox.com/s/toqulkdpb1jrswk/voc_superpixels_edge_wt_only_coord.zip?dl=1',
            'edge_wt_coord_feat': 'https://www.dropbox.com/s/xywki8ysj63584d/voc_superpixels_edge_wt_coord_feat.zip?dl=1',
            'edge_wt_region_boundary': 'https://www.dropbox.com/s/8x722ai272wqwl4/voc_superpixels_edge_wt_region_boundary.zip?dl=1'
        }
    }

    def __init__(self, root, name='edge_wt_region_boundary', slic_compactness=30, split='train',
                 transform=None, pre_transform=None, pre_filter=None, suffix=''):
        self.name = name
        self.slic_compactness = slic_compactness
        assert split in ['train', 'val', 'test']
        assert name in ['edge_wt_only_coord', 'edge_wt_coord_feat', 'edge_wt_region_boundary']
        self.suffix = suffix
        assert slic_compactness in [10, 30]
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return ['train.pickle', 'val.pickle', 'test.pickle']

    @property
    def raw_dir(self):
        return osp.join(self.root,
                        'slic_compactness_' + str(self.slic_compactness),
                        self.name,
                        'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root,
                        'slic_compactness_' + str(self.slic_compactness),
                        self.name + self.suffix,
                        'processed')

    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url[self.slic_compactness][self.name], self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, 'voc_superpixels_' + self.name), self.raw_dir)
        os.unlink(path)

    def _sample(self, split, n):
        random.seed(42)
        indices = range(n)
        if self.suffix == 'small':
            epsilon = 0.15 if split == 'train' else 1
            indices = random.sample(indices, int(n * epsilon))
            print(f'first 5 indices for {split}: {indices[:5]}')
        elif self.suffix == 'tiny':
            indices = random.sample(indices, int(n * 0.01))
        elif self.suffix in ['', None, 'None']:
            pass
        else:
            raise NotImplementedError
        return indices

    def process(self):
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                graphs = pickle.load(f)
            indices = self._sample(split, len(graphs))
            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for idx in indices:
                graph = graphs[idx]

                """
                Each `graph` is a tuple (x, edge_attr, edge_index, y)
                    Shape of x : [num_nodes, 14]
                    Shape of edge_attr : [num_edges, 1] or [num_edges, 2]
                    Shape of edge_index : [2, num_edges]
                    Shape of y : [num_nodes]
                """

                x = graph[0].to(torch.float)
                edge_attr = graph[1].to(torch.float)
                edge_index = graph[2]
                y = torch.LongTensor(graph[3])

                n = x.size(0)
                default_mask = torch.zeros(n, dtype=torch.bool)
                train_mask = torch.ones(n, dtype=torch.bool) if split == 'train' else default_mask
                val_mask = torch.ones(n, dtype=torch.bool) if split == 'val' else default_mask
                test_mask = torch.ones(n, dtype=torch.bool) if split == 'test' else default_mask

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                            y=y, train_mask=train_mask, val_mask=val_mask,
                            test_mask=test_mask)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()

            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))


if __name__ == '__main__':
    dataset = NewVOCSuperpixels('./datasets', suffix='small')
    print(dataset)
