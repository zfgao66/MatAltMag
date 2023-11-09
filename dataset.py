"""
Dataset Class.

At the first time, CIF file will be read into memory and processed into tensor
data, which will be saved in disk. At the next time, tensor data will be read
into memory from disk directly without processing CIF file.

@reference: https://github.com/txie-93/cgcnn
"""
import csv
import os
import random
from torch.utils.data import Dataset
from dataset_helper import GaussianDistance, AtomCustomJSONInitializer, read_cif_data, process_data, load_data
import os.path as osp


class TrainCIFData(Dataset):
    """
    The CIFData dataset is a wrapper for a dataset where the crystal structures
    are stored in the form of CIF files. The dataset should have the following
    directory structure:

    root_dir
    ├── id_prop.csv
    ├── atom_init.json
    ├── id0.cif
    ├── id1.cif
    ├── ...

    id_prop.csv: a CSV file with two columns. The first column recodes a
    unique ID for each crystal, and the second column recodes the value of
    target property.

    atom_init.json: a JSON file that stores the initialization vector for each
    element.

    ID.cif: a CIF file that recodes the crystal structure, where ID is the
    unique ID for the crystal.

    Parameters
    ----------

    root_dir: str
        The path to the root directory of the dataset
    max_num_nbr: int
        The maximum number of neighbors while constructing the crystal graph
    radius: float
        The cutoff radius for searching neighbors
    dmin: float
        The minimum distance for constructing GaussianDistance
    step: float
        The step size for constructing GaussianDistance
    random_seed: int
        Random seed for shuffling the dataset

    Returns
    -------

    atom_fea: torch.Tensor shape (n_i, atom_fea_len)
    nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
    nbr_fea_idx: torch.LongTensor shape (n_i, M)
    target: torch.Tensor shape (1, )
    cif_id: str or int
    """
    def __init__(self, root_dir, processed_dir, logger, max_num_nbr, radius, dmin, step, ratio):
        self.root_dir = root_dir
        self.processed_dir_nega = osp.join(self.root_dir, processed_dir['negative'])
        self.processed_dir_posi = osp.join(self.root_dir, processed_dir['positive'])
        self.logger = logger
        self.ratio = ratio
        self.max_num_nbr, self.radius = max_num_nbr, radius
        assert os.path.exists(root_dir), 'root_dir does not exist!'

        id_prop_nega_file = os.path.join(self.root_dir, 'id_prop_0.csv')
        id_prop_posi_file = os.path.join(self.root_dir, 'id_prop_1.csv')
        atom_init_file = os.path.join(self.root_dir, 'atom_init.json')
        assert os.path.exists(id_prop_nega_file), 'id_prop_0.csv does not exist!'
        assert os.path.exists(id_prop_posi_file), 'id_prop_1.csv does not exist!'
        assert os.path.exists(atom_init_file), 'atom_init.json does not exist!'

        with open(id_prop_nega_file) as f:
            reader = csv.reader(f)
            self.id_prop_nega_data = [row for row in reader]
        with open(id_prop_posi_file) as f:
            reader = csv.reader(f)
            self.id_prop_posi_data = [row for row in reader]

        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)

        self.dataset_posi, self.dataset_nega = self._check_data()
        self.len_posi_true = len(self.dataset_posi[0])
        self.len_posi_fake = self.len_posi_true * self.ratio
        self.len_nega = len(self.dataset_nega[0])

    def _check_data(self):
        if not osp.exists(self.processed_dir_posi):
            self.logger.info('Process...')
            process_data(
                ari=self.ari,
                gdf=self.gdf,
                id_prop_data=self.id_prop_posi_data,
                radius=self.radius,
                max_num_nbr=self.max_num_nbr,
                root_dir=self.root_dir,
                save_dir=self.processed_dir_posi
            )
        dataset_posi = load_data(self.processed_dir_posi)

        if not osp.exists(self.processed_dir_nega):
            process_data(
                ari=self.ari,
                gdf=self.gdf,
                id_prop_data=self.id_prop_nega_data,
                radius=self.radius,
                max_num_nbr=self.max_num_nbr,
                root_dir=self.root_dir,
                save_dir=self.processed_dir_nega
            )
        dataset_nega = load_data(self.processed_dir_nega)
        return dataset_posi, dataset_nega


    def __len__(self):
        return self.len_posi_fake + self.len_nega

    def __getitem__(self, idx):
        if idx < self.len_posi_fake:
            idx = idx % self.len_posi_true
            atom_fea, nbr_fea, nbr_fea_idx, degree, target, cif_id = self.dataset_posi[0][idx], self.dataset_posi[1][idx], self.dataset_posi[2][idx], self.dataset_posi[3][idx], self.dataset_posi[4][idx], self.dataset_posi[5][idx]
        else:
            idx = idx - self.len_posi_fake
            atom_fea, nbr_fea, nbr_fea_idx, degree, target, cif_id = self.dataset_nega[0][idx], self.dataset_nega[1][idx], \
            self.dataset_nega[2][idx], self.dataset_nega[3][idx], self.dataset_nega[4][idx], self.dataset_nega[5][idx]

        return (atom_fea, nbr_fea, nbr_fea_idx, degree), target, cif_id


class PretrainCIFData(Dataset):
    def __init__(self, root_dir, processed_dir, logger, max_num_nbr, radius, dmin, step):
        self.root_dir = root_dir
        self.processed_dir_nega = osp.join(self.root_dir, processed_dir['negative'])
        self.processed_dir_cand = osp.join(self.root_dir, processed_dir['candidate'])
        self.logger = logger
        self.max_num_nbr, self.radius = max_num_nbr, radius
        assert os.path.exists(root_dir), 'root_dir does not exist!'

        id_prop_nega_file = os.path.join(self.root_dir, 'id_prop_0.csv')
        id_prop_cand_file = os.path.join(self.root_dir, 'id_prop_-1.csv')
        atom_init_file = os.path.join(self.root_dir, 'atom_init.json')
        assert os.path.exists(id_prop_nega_file), 'id_prop_0.csv does not exist!'
        assert os.path.exists(id_prop_cand_file), 'id_prop_-1.csv does not exist!'
        assert os.path.exists(atom_init_file), 'atom_init.json does not exist!'

        with open(id_prop_nega_file) as f:
            reader = csv.reader(f)
            self.id_prop_nega_data = [row for row in reader]
        with open(id_prop_cand_file) as f:
            reader = csv.reader(f)
            self.id_prop_cand_data = [row for row in reader]

        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)

        self.dataset_cand, self.dataset_nega = self._check_data()
        self.len_cand = len(self.dataset_cand[0])
        self.len_nega = len(self.dataset_nega[0])

    def _check_data(self):
        if not osp.exists(self.processed_dir_cand):
            self.logger.info('Process...')
            process_data(
                ari=self.ari,
                gdf=self.gdf,
                id_prop_data=self.id_prop_cand_data,
                radius=self.radius,
                max_num_nbr=self.max_num_nbr,
                root_dir=self.root_dir,
                save_dir=self.processed_dir_cand
            )
        dataset_cand = load_data(self.processed_dir_cand)

        if not osp.exists(self.processed_dir_nega):
            self.logger.info('Process...')
            process_data(
                ari=self.ari,
                gdf=self.gdf,
                id_prop_data=self.id_prop_nega_data,
                radius=self.radius,
                max_num_nbr=self.max_num_nbr,
                root_dir=self.root_dir,
                save_dir=self.processed_dir_nega
            )
        dataset_nega = load_data(self.processed_dir_nega)
        return dataset_cand, dataset_nega

    def __len__(self):
        return self.len_cand + self.len_nega

    def __getitem__(self, idx):
        if idx < self.len_cand:
            atom_fea, nbr_fea, nbr_fea_idx, degree, target, cif_id = self.dataset_cand[0][idx], self.dataset_cand[1][idx], \
            self.dataset_cand[2][idx], self.dataset_cand[3][idx], self.dataset_cand[4][idx], self.dataset_cand[5][idx]
        else:
            idx = idx - self.len_cand
            atom_fea, nbr_fea, nbr_fea_idx, degree, target, cif_id = self.dataset_nega[0][idx], self.dataset_nega[1][idx], \
                self.dataset_nega[2][idx], self.dataset_nega[3][idx], self.dataset_nega[4][idx], self.dataset_nega[5][idx]

        return (atom_fea, nbr_fea, nbr_fea_idx, degree), target, cif_id


class PredictCIFData(Dataset):
    def __init__(self, root_dir, processed_dir, logger, max_num_nbr, radius, dmin, step):
        self.root_dir = root_dir
        self.processed_dir_cand = osp.join(self.root_dir, processed_dir)
        self.logger = logger
        self.max_num_nbr, self.radius = max_num_nbr, radius
        assert os.path.exists(root_dir), 'root_dir does not exist!'

        id_prop_cand_file = os.path.join(self.root_dir, 'id_prop_-1.csv')
        atom_init_file = os.path.join(self.root_dir, 'atom_init.json')
        assert os.path.exists(id_prop_cand_file), 'id_prop_-1.csv does not exist!'
        assert os.path.exists(atom_init_file), 'atom_init.json does not exist!'

        with open(id_prop_cand_file) as f:
            reader = csv.reader(f)
            self.id_prop_cand_data = [row for row in reader]

        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)

        self.dataset_cand = self._check_data()
        self.len_cand = len(self.dataset_cand[0])

    def _check_data(self):
        if not osp.exists(self.processed_dir_cand):
            self.logger.info('Process...')
            process_data(
                ari=self.ari,
                gdf=self.gdf,
                id_prop_data=self.id_prop_cand_data,
                radius=self.radius,
                max_num_nbr=self.max_num_nbr,
                root_dir=self.root_dir,
                save_dir=self.processed_dir_cand
            )
        dataset_cand = load_data(self.processed_dir_cand)

        return dataset_cand

    def __len__(self):
        return self.len_cand

    def __getitem__(self, idx):
        atom_fea, nbr_fea, nbr_fea_idx, degree, target, cif_id = self.dataset_cand[0][idx], self.dataset_cand[1][idx], self.dataset_cand[2][idx], self.dataset_cand[3][idx], self.dataset_cand[4][idx], self.dataset_cand[5][idx]

        return (atom_fea, nbr_fea, nbr_fea_idx, degree), target, cif_id