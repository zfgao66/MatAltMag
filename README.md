# MatAltMag
The pytorch implement of papar *AI-accelerated Discovery of Altermagnetic Materials*

# Prerequisites
some package required:
- pytorch: 2.0.1
- accelerate: 0.20.0
- pymatgen: 2023.5.10
- PyYAML: 6.0
- tqdm: 4.64.0

Other required packages are listed in the `requirements.txt` file.

# Run
1. run files in the `preprocess` directory, move the output files of `label0.csv` and `candidate.csv` to `root_dir` directory
2. check required data files in `root_dir`
- `atom_init.json`: a JSON file that stores the initialization vector for each element.
- `label0.csv`: a CSV file that stores the `ID` for the non-altermagnetic crystal
- `label1.csv`: a CSV file that stores the `ID` for the altermagnetic crystal
- `candidate.csv`: a CSV file that stores the `ID` for the crystal in candidate datasets
3. run `download.py` to download `CIF` files of all crystals in the three CSV files from [Materials Project](https://materialsproject.org/). Download time depends on your internet speed. Once completed, the structure under `root_dir` will be
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
4. set the configuration of accelerate in a proper path, for example
```shell
accelerate config --config_file yamls/accelerate.yaml
```
5. check and update yamls/pretrain.yaml, then run `pretrain.py`
```shell
nohup sh pretrain.sh &
```
or
```shell
accelerate launch --config_file yamls/accelerate.yaml pretrain.py --file yamls/pretrain.yaml 
```
6. check and update yamls/train.yaml, then run `train.py`
```shell
nohup sh train.sh &
```
or 
```shell
accelerate launch --config_file yamls/accelerate.yaml train.py --file yamls/train.yaml
```
7. check and update yamls/predict.yaml, then run `predict.py`
```shell
python predict.py --file yamls/predict.yaml
```
Downloading all `CIF` files of all crystals using the `download.py` script takes about 1 hour, depending on your network speed. Pretraining the auto-encoder model for 10 epochs with a batch size of 64 on 2 NVIDIA A100 GPUs takes over 2 days. Predicting the candidate datasets of over 42,000 samples takes about 10 seconds with a batch size of 512.

You can also directly load our trained model, which has undergone multiple iterative training processes, without pre-training and training it yourself. The weights of our classifier model can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1Dbb3u-_LGZ8trq1w4o173GWeGjtksghx?usp=sharing). The corresponding output is presented in the `out/output.csv`.  In addition, over three hundred additional candidate materials (unconfirmed yet by DFT calculations) were predicted by the proposed AI search engine. These candidates were listed in `out/Candidate_for_DFT_validate.csv`.
