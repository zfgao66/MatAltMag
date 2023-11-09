"""
some helper methods and classes of Dataset.

@reference: https://github.com/txie-93/cgcnn
"""
import torch
import numpy as np
import json
from tqdm import tqdm
import warnings
import os.path as osp
import os
from pymatgen.core.structure import Structure
import pickle


def my_collate_fn(dataset_list):
    return dataset_list[0]


def collate_pool(dataset_list):
    """
    Collate a list of data and return a batch for predicting crystal
    properties.

    Parameters
    ----------

    dataset_list: list of tuples for each data point.
      (atom_fea, nbr_fea, nbr_fea_idx, target)

      atom_fea: torch.Tensor shape (n_i, atom_fea_len)
      nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
      nbr_fea_idx: torch.LongTensor shape (n_i, M)
      target: torch.Tensor shape (1, )
      cif_id: str or int

    Returns
    -------
    N = sum(n_i); N0 = sum(i)

    batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
      Atom features from atom type
    batch_nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
      Bond features of each atom's M neighbors
    batch_nbr_fea_idx: torch.LongTensor shape (N, M)
      Indices of M neighbors of each atom
    crystal_atom_idx: list of torch.LongTensor of length N0
      Mapping from the crystal idx to atom idx
    target: torch.Tensor shape (N, 1)
      Target value for prediction
    batch_cif_ids: list
    """
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx, batch_degree = [], [], [], []
    crystal_atom_idx, batch_target = [], []
    batch_cif_id = []
    base_idx = 0
    for i, ((atom_fea, nbr_fea, nbr_fea_idx, degree), target, cif_id)\
            in enumerate(dataset_list):
        n_i = atom_fea.shape[0]  # number of atoms for this crystal
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx+base_idx)
        batch_degree.append(degree)
        new_idx = torch.LongTensor(np.arange(n_i)+base_idx)
        crystal_atom_idx.append(new_idx)
        batch_target.append(target)
        batch_cif_id.append(cif_id)
        base_idx += n_i
    return (torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            torch.cat(batch_degree, dim=0),
            crystal_atom_idx),\
        torch.stack(batch_target, dim=0),\
        batch_cif_id


class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """
    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------

        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array

        Parameters
        ----------

        distance: np.array shape n-d array
          A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)


class AtomInitializer(object):
    """
    Base class for intializing the vector representation for atoms.

    !!! Use one AtomInitializer per dataset !!!
    """
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.

    Parameters
    ----------

    elem_embedding_file: str
        The path to the .json file
    """
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


def read_cif_data(root_dir, ari, gdf, id_prop_data, radius, max_num_nbr):
    """
    Read the cif data into tensor

    Parameters
    ----------
    root_dir
    ari: AtomCustomJSONInitializer
    gdf: GaussianDistance
    id_prop_data
    radius
    max_num_nbr

    Returns
    -------

    """
    atom_fea_data, nbr_fea_data, nbr_fea_idx_data, target_data, cif_id_data, node_num_data, degree_data = [], [], [], [], [], [], []
    length = len(id_prop_data)
    for i in tqdm(range(length)):
        cif_id, target = id_prop_data[i]
        crystal = Structure.from_file(osp.join(root_dir, cif_id + '.cif'))
        atom_fea = np.vstack([ari.get_atom_fea(crystal[i].specie.number)
                              for i in range(len(crystal))])
        atom_fea = torch.Tensor(atom_fea)
        all_nbrs = crystal.get_all_neighbors(radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        nbr_fea_idx, nbr_fea, degree = [], [], []
        for nbr in all_nbrs:

            if len(nbr) < max_num_nbr:
                warnings.warn('{} not find enough neighbors to build graph. '
                              'If it happens frequently, consider increase '
                              'radius.'.format(cif_id))
                idx = list(map(lambda x: x[2], nbr)) + [0] * (max_num_nbr - len(nbr))
                fea = list(map(lambda x: x[1], nbr)) + [radius + 1.] * (max_num_nbr -
                                                     len(nbr))

            else:
                idx = list(map(lambda x: x[2], nbr[:max_num_nbr]))
                fea = list(map(lambda x: x[1], nbr[:max_num_nbr]))
            nbr_fea_idx.append(idx)
            nbr_fea.append(fea)
            degree.append(len(set(idx)))
        nbr_fea_idx, nbr_fea, degree = np.array(nbr_fea_idx), np.array(nbr_fea), np.array(degree)
        nbr_fea = gdf.expand(nbr_fea)
        atom_fea = torch.Tensor(atom_fea)
        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        degree = torch.Tensor(degree)
        target = torch.Tensor([float(target)])

        atom_fea_data.append(atom_fea)
        nbr_fea_data.append(nbr_fea)
        nbr_fea_idx_data.append(nbr_fea_idx)
        degree_data.append(degree)
        target_data.append(target)
        cif_id_data.append(cif_id)
        node_num_data.append(atom_fea.shape[0])

    return torch.cat(atom_fea_data, dim=0), \
                torch.cat(nbr_fea_data, dim=0), \
                torch.cat(nbr_fea_idx_data, dim=0), \
                torch.cat(degree_data, dim=0), \
                torch.cat(target_data, dim=0), \
                cif_id_data, \
                torch.tensor(node_num_data, dtype=torch.long)


def process_data(ari, gdf, id_prop_data, radius, max_num_nbr, root_dir, save_dir):
    atom_fea_data, nbr_fea_data, nbr_fea_idx_data, degree_data, target_data, cif_id_data, node_num_data = read_cif_data(
        root_dir=root_dir,
        ari=ari,
        gdf=gdf,
        id_prop_data=id_prop_data,
        radius=radius,
        max_num_nbr=max_num_nbr
    )
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(atom_fea_data, osp.join(save_dir, 'atom_fea.pt'))
    torch.save(nbr_fea_data, osp.join(save_dir, 'nbr_fea.pt'))
    torch.save(nbr_fea_idx_data, osp.join(save_dir, 'nbr_fea_idx.pt'))
    torch.save(degree_data, osp.join(save_dir, 'degree.pt'))
    with open(osp.join(save_dir, 'cif_id.pkl'), 'wb') as f:
        pickle.dump(cif_id_data, f)
    torch.save(target_data, osp.join(save_dir, 'target.pt'))
    torch.save(node_num_data, osp.join(save_dir, 'node_num.pt'))


def load_data(processed_dir):
    atom_fea_data, nbr_fea_data, nbr_fea_idx_data, degree_data, target_data, node_num_data = torch.load(osp.join(processed_dir, 'atom_fea.pt')), torch.load(osp.join(processed_dir, 'nbr_fea.pt')), torch.load(osp.join(processed_dir, 'nbr_fea_idx.pt')), torch.load(osp.join(processed_dir, 'degree.pt')), torch.load(osp.join(processed_dir, 'target.pt')), torch.load(osp.join(processed_dir, 'node_num.pt'))
    with open(osp.join(processed_dir, 'cif_id.pkl'), 'rb') as f:
        cif_id_data = pickle.load(f)

    node_num_list = node_num_data.numpy().tolist()
    atom_fea_list, nbr_fea_list, nbr_fea_idx_list, degree_list = torch.split(atom_fea_data, node_num_list), torch.split(nbr_fea_data, node_num_list), torch.split(nbr_fea_idx_data, node_num_list), torch.split(degree_data, node_num_list)
    target_data = target_data.long()
    target_list = target_data
    return atom_fea_list, nbr_fea_list, nbr_fea_idx_list, degree_list, target_list, cif_id_data