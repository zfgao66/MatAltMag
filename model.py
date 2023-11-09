"""
Model class contained encoder, decoder and head modules.

"""
import torch.nn as nn
from cgcnn_modules import CrystalGraphConvNet
from nwr_gae_modules import GNNAutoDecoder


class CrystalGraph(nn.Module):
    def __init__(self, crystal_gnn_config, head_output_dim, drop_rate, decoder_sample_size, device):
        super().__init__()
        self.encoder = CrystalGraphConvNet(
            orig_atom_fea_len=crystal_gnn_config['orig_atom_fea_len'],
            atom_fea_len=crystal_gnn_config['atom_fea_len'],
            nbr_fea_len=crystal_gnn_config['nbr_fea_len'],
            n_conv=crystal_gnn_config['n_conv']
        )
        self.decoder = GNNAutoDecoder(
            input_dim=crystal_gnn_config['orig_atom_fea_len'],
            hidden_dim=crystal_gnn_config['atom_fea_len'],
            layer_num=crystal_gnn_config['n_conv']-1,
            sample_size=decoder_sample_size,
            device=device,
            degree_lambda=1e-4,
            lambda_loss1=1e-6,
            lambda_loss2=1
        )
        self.head = nn.Sequential(
            nn.Linear(crystal_gnn_config['atom_fea_len'], crystal_gnn_config['atom_fea_len']),
            nn.Dropout(p=drop_rate),
            nn.ReLU(),
            nn.Linear(crystal_gnn_config['atom_fea_len'], head_output_dim)
        )
        self.softmax = nn.Softmax(dim=1)

    def pretrain(self, atom_fea, nbr_fea, nbr_fea_idx, degree, crystal_atom_idx):
        self.encoder.set_pre_train(True)
        outputs_per_layer = self.encoder(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
        loss = self.decoder(degree, nbr_fea_idx, outputs_per_layer)
        return loss

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        self.encoder.set_pre_train(False)
        feature = self.encoder(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
        # feature = F.normalize(self.head(feature), dim=1)
        feature = self.head(feature)
        return feature

    def predict(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        feature = self.forward(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
        prob = self.softmax(feature)
        return prob