# Created by Chen at 11/1/22
# generate config files for Peptides-func and Peptides-struc
import os
import os.path as osp

import yaml


class VN_config(object):
    def __init__(self, dataset, gcn):
        self.init_dataset_dict()
        assert dataset in self.dataset_dict.keys(), f'{dataset} not supported'
        self.dir = osp.normpath(osp.join(__file__, '..', gcn.split('+')[0]), )
        self.f = osp.join(self.dir, f'{self.dataset_dict[dataset]}-{gcn}.yaml')
        self.dataset = dataset
        assert osp.exists(self.f), f'{self.f} does not exists!'
        print(self.f)

    def init_dataset_dict(self):
        # value is used to look up file
        d = {'coco': 'cocosuperpixels',
             'pcqm': 'pcqm-contact',
             'pep-func': 'peptides-func',
             'pep-struct': 'peptides-struct',
             'voc': 'vocsuperpixels'}
        self.dataset_dict = d

        # values is used to look up dataset name in yaml.file
        d = {'pep-func': 'peptides-functional',
             'pep-struct': 'peptides-structural',
             'voc': 'PyG-VOCSuperpixels',
             'coco': 'PyG-COCOSuperpixels',
             'pcqm': 'PCQM4Mv2Contact-shuffle',
             }

    @staticmethod
    def load_cfg(f):
        with open(f, 'r', encoding='utf-8') as ymlfile:
            cfg = yaml.load(ymlfile, yaml.FullLoader)
        return cfg

    @staticmethod
    def write_cfg(cfg, f):
        assert 'yaml' in f
        with open(f, 'w') as file:
            yaml.dump(cfg, file, sort_keys=False, encoding="utf-8")
        print(f'write to {f}')

    def writie_to_new_file(self):
        tmp = self.f.split('/')
        tmp[-1] = 'VN-' + tmp[-1]
        newf = '/'.join(tmp)
        cfg = self.load_cfg(self.f)

        # handle change of cfg
        if self.dataset in ['pep-func', 'pep-struct', 'pcqm']:
            # only change cfg['dataset']['name']
            cfg['dataset']['name'] = 'VN-' +  cfg['dataset']['name']
        elif self.dataset in ['voc', 'coco']:
            # PyG-VOCSuperpixels
            tmp = cfg['dataset']['format']
            assert tmp[3] == '-', tmp
            cfg['dataset']['format'] = '-'.join([tmp[:3], 'VN', tmp[4:]])
            cfg['dataset']['name'] = 'edge_wt_region_boundary'
            cfg['dataset']['slic_compactness'] = 30
        else:
            raise NotImplementedError

        # write cfg to newyaml file
        self.write_cfg(cfg, newf)
        cmd = f'diff -y {self.f} {newf}'
        print(cmd)

        # os.system(cmd)




if __name__ == '__main__':
    gcns = ['GatedGCN', 'GCN', 'GCNII', 'GINE', 'GatedGCN+RWSE'] # 'GatedGCN+LapPE'
    datasets = ['pcqm',] # ['pep-func', 'pep-struct']
    for gcn in gcns:
        for dataset in datasets:
            vn = VN_config(dataset, gcn)
            vn.writie_to_new_file()
            print('-' * 50)

