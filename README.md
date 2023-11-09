# MatAltMag
The pytorch implement of papar *AI-accelerated Discovery of Altermagnetic Materials*

# Prerequisites
package required:
- pytorch
- accelerate
- pymatgen
- PyYAML
- tensorboard
- tqdm
- pandas

# Run
1. check required data files in `root_dir`
- `atom_init.json`: a JSON file that stores the initialization vector for each element.
- `label0.csv`: a CSV file that stores the `ID` for the non-altermagnetic crystal
- `label1.csv`: a CSV file that stores the `ID` for the altermagnetic crystal
- `candidate.csv`: a CSV file that stores the `ID` for the crystal in candidate datasets
2. run `download.py` to download `CIF` files of all crystals in the three CSV files from [Materials Project](https://materialsproject.org/). Download time depends on your internet speed. Once completed, the structure under `root_dir` will be
```shell
root_dir
├── atom_init.json
├── label0.csv
├── label1.csv
├── candidate.csv
├── id_prop_0.csv
├── id_prop_1.csv
├── id_prop_-1.csv
├── id0.cif
├── id1.cif
├── ...
```
3. set the configuration of accelerate in a proper path, for example
```shell
accelerate config --config_file yamls/accelerate.yaml
```
4. check and update yamls/pretrain.yaml, then run `pretrain.py`
```shell
nohup sh pretrain.sh &
```
or
```shell
accelerate launch --config_file yamls/accelerate.yaml pretrain.py --file yamls/pretrain.yaml 
```
5. check and update yamls/train.yaml, then run `train.py`
```shell
nohup sh train.sh &
```
or 
```shell
accelerate launch --config_file yamls/accelerate.yaml train.py --file yamls/train.yaml
```
6. check and update yamls/predict.yaml, then run `predict.py`
```shell
python predict.py --file yamls/predict.py
```
You can also directly load our trained model without pre-training and training yourself. Our weights of classifier model can be download from [Google Drive](https://drive.google.com/drive/folders/1xYQrIfC71z-IlD33hkTdunTsUgMLb_hA?usp=drive_link).
