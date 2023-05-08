This is the code repository for the ICML 2023 paper *On the Connection Between MPNN and Graph Transformer*. There are three folders corresponding to the three experiment subsections.

The `lrgb` contains the code for section 7.1 where we showed adding virtual node can significantly improve the result on lrgb datasets. The code is modified from the original lrgb paper. See [Readme](./lrgb/README.md) for instructions. 

The  `GraphGPS` contains the code for section 7.2, where we showed that by leveraging the the framework of GraphGPS, simple MPNN + VN can achieve competitive on `ogb` datasets.  See [Readme](./GraphGPS/README.md)

The `climate-graph-main` contains the code for section 7.3, where we experiment the MPNN + VN for sea-surface temperature prediction. 

```
@article{cai2023connection,
  title={On the Connection Between MPNN and Graph Transformer},
  author={Cai, Chen and Hy, Truong Son and Yu, Rose and Wang, Yusu},
  journal={arXiv preprint arXiv:2301.11956},
  year={2023}
}
```

