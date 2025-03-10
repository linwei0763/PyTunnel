# This repository includes the utils supporting Seg2Tunnel, LiningNet, and Pointcept.

# installation
```bash
See Seg2Lining
```

# usage
```bash
conda activate seg2lining
cd seg2lining
```

# feature seg2tunnel
```bash
python count_num_point.py
python count_ring.py
python label_crack.py
python generate_synthetic_data.py --dataset=dublin
python generate_synthetic_data.py --dataset=wuxi
python transform_rgb.py
```

# feature liningnet
```bash
python evaluate_data.py
python evaluate_sparse_data.py
python find_neighbour.py
python read_training_log.py
```

# feature pointcept
```bash
python merge_result.py
```

# feature deformation
```bash
python analyse_convergence.py
python analyse_dislocation.py
python analyse_error.py
python compute_deformation.py
python compute_deviation.py
```