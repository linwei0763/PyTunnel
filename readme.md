# install

conda create -n pytunnel python=3.8

conda activate pytunnel

cd PyTunnel

pip install -r requirements.txt

pip install open3d


# implement

conda activate pytunnel

cd PyTunnel

# feature

* compute_deformation_ring.py  
compute the deformation of lining rings  
* count_num_point.py  
count the number of points  
* count_ring.py  
count the number of rings in each station  
* find_neighbour_tunnel.py  
find points in the local neighbourhood and structural neighbourhood of the centre point  
* generate_synthetic_data.py  
generate synthetic point clouds  
