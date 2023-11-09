"""
Download materials from materialsproject.org

"""
from mp_api.client import MPRester
import os.path as osp
import pandas as pd
import kogger
from tqdm import tqdm


def read_material_ids(data_path):
    """
    Parameters
    ----------
    data_path: file path of sample.csv

    Returns
    -------
    material_ids: list

    """
    df = pd.read_csv(data_path, header=None)
    kogger.info('Read {} samples from {}'.format(len(df), data_path))
    material_ids = df[1].tolist()
    return material_ids


def download_materials(material_ids, save_dir, label=1):
    kogger.info('Download...')
    with MPRester(api_key) as mpr:
        ids = []
        docs = mpr.bonds.search(material_ids=material_ids, fields=['material_id', 'structure_graph'])
        length = len(docs)
        for idx in tqdm(range(length)):
            struc = docs[idx].structure_graph.structure
            id = docs[idx].material_id
            ids.append(id)
            # save file,  'data/mp-22862.cif'
            struc.to(filename=osp.join(save_dir, '{}.cif'.format(id)))

        # append to 'data/id_prop_{}.csv'
        labels = [label] * len(ids)
        df = pd.DataFrame(list(zip(ids, labels)))
        path = osp.join(save_dir, 'id_prop_{}.csv'.format(label))
        kogger.info('Appending to {}'.format(path))
        df.to_csv(path, header=None, index=None, mode='a')


def main():
    # label = 1 for label1.csv, 0 for label0.csv, -1 for candidate.csv
    label = -1
    # data_path = 'data/label1.csv'
    # data_path = 'data/label0.csv'
    data_path = 'data/candidate.csv'
    save_dir = 'data'

    material_ids = read_material_ids(data_path)
    download_materials(material_ids, save_dir, label=label)


if __name__ == '__main__':
    api_key = 'YOUR API KEY'

    main()
