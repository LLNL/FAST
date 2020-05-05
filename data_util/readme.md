
1. Download pdbbind dataset

2. run parse_affinity_data.sh : to extract csv from index files in pdbbind 

3. run parallel_chimera_preprocess_pdbbind_2018_refined_docking.sh : to add hydrogen in the original dataset, to generate new pdb? (for larger dataset)

3a: run prepare_complexes_chimera.sh (for smaller dataset) 

4. run extract_pafnucy_data_with_docking.py : to generate hdf5 from the csv and the original pdbbind dataset


- 3D CNN:

5. run main_generate_3dcnn_pdbbind_hdf.py : to generate 3D representation for 3D CNN training


- SG-CNN: 
5. use the hdf5 in 4 directly, and convert it to SG format on the fly



 
* preprocess_water_pdbbind.py : for experiments on without-water datasets
