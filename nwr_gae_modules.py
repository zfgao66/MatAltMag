"""

@reference: https://github.com/mtang724/NWR-GAE
"""
import torch.nn as nn
import torch
from nwr_gae_layers import FNN, MLPGenerator
import torch.nn.functional as F
from utils import hungarian_loss
import torch.multiprocessing as mp
import random


class GNNAutoDecoder(nn.Module):
    """
    Compute the loss computed with the method of NWR-GAE
    """
    def __init__(self, input_dim, hidden_dim, layer_num, sample_size, device, degree_lambda, lambda_loss1, lambda_loss2):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        self.sample_size = sample_size
        self.device = device

        # modules
        # self.degree_decoder = FNN(hidden_dim, hidden_dim, 1, 4)
        self.feature_decoder = FNN(hidden_dim, hidden_dim, input_dim, 3)
        self.layer_generators = self.build_layer_generator()
        self.mlp_mean = nn.Linear(hidden_dim, hidden_dim)
        self.mlp_sigma = nn.Linear(hidden_dim, hidden_dim)
        self.m = torch.distributions.Normal(torch.zeros(sample_size, hidden_dim),
                                            torch.ones(sample_size, hidden_dim))

        self.pool = mp.Pool(4)

        # loss_func
        # self.degree_loss_func = nn.MSELoss()
        self.feature_loss_func = nn.MSELoss()

        self.degree_lambda = degree_lambda
        self.lambda_loss1 = lambda_loss1
        self.lambda_loss2 = lambda_loss2

    def build_layer_generator(self):
        layers = nn.ModuleList()
        for i in range(self.layer_num):
            layers.append(MLPGenerator(self.hidden_dim, self.hidden_dim))
        return layers

    # def degree_decoding(self, node_embeddings):
    #     degree_prd = F.relu(self.degree_decoder(node_embeddings))
    #     return degree_prd

    def sample_neighbors(self, indexes, neighbor_dict, gt_embeddings):
        sampled_embeddings_list = []
        mark_len_list = []
        for index in indexes:
            neighbor_indexes = neighbor_dict[index]
            neighbor_indexes = torch.unique(neighbor_indexes)
            if neighbor_indexes.shape[0] < self.sample_size:
                mask_len = neighbor_indexes.shape[0]
                sample_indexes = neighbor_indexes
            else:
                rand_ints = torch.randint(high=neighbor_indexes.shape[0], size=(self.sample_size,), device=self.device)
                sample_indexes = torch.index_select(neighbor_indexes, 0, rand_ints)
                mask_len = self.sample_size
            sampled_embeddings = torch.index_select(gt_embeddings, 0, sample_indexes)
            if sampled_embeddings.shape[0] < self.sample_size:
                padding = torch.zeros((self.sample_size - sampled_embeddings.shape[0], self.hidden_dim), device=self.device)
                sampled_embeddings = torch.cat((sampled_embeddings, padding), dim=0)
            sampled_embeddings_list.append(sampled_embeddings)
            mark_len_list.append(mask_len)
        return sampled_embeddings_list, mark_len_list

    def reconstruction_neighbors(self, fnn_decoder, neighbor_indexes, neighbors, from_layer, to_layer):
        """
        Reconstruction Neighbors

        Parameters
        ----------
        fnn_decoder: nn.Module, FNN decoder
        neighbor_indexes: torch.Tensor [n,], new neighbor indexes after hungarian matching
        neighbors: torch.Tensor [n, n], neighbors
        from_layer: torch.Tensor [n, h_dim], output of layer K
        to_layer: torch.Tensor [n, h_dim], input of layer K

        Returns
        -------
        loss: reconstruction loss
        new_index: new indexes after hungarian matching
        """
        local_index_loss = 0
        sampled_embeddings_list, mark_len_list = self.sample_neighbors(neighbor_indexes, neighbors, to_layer)
        for i, neighbor_embeddings1 in enumerate(sampled_embeddings_list):
            # Generating h^k_v, reparameterization trick
            index = neighbor_indexes[i]
            mask_len1 = mark_len_list[i]
            mean = from_layer[index].repeat(self.sample_size, 1)
            mean = self.mlp_mean(mean)
            sigma = from_layer[index].repeat(self.sample_size, 1)
            sigma = self.mlp_sigma(sigma)
            std_z = self.m.sample().to(self.device)
            var = mean + sigma.exp() * std_z
            nhij = fnn_decoder(var)
            generated_neighbors = nhij
            # Caculate 2-Wasserstein distance
            # sum_neighbor_norm = 0
            # # For appendix D approximate experiment
            # for indexi, generated_neighbor in enumerate(generated_neighbors):
            #     sum_neighbor_norm += torch.norm(generated_neighbor) / torch.sqrt(self.out_dim)
            generated_neighbors = torch.unsqueeze(generated_neighbors, dim=0)
            target_neighbors = torch.unsqueeze(neighbor_embeddings1, dim=0)
            hun_loss, new_index = hungarian_loss(generated_neighbors, target_neighbors, mask_len1, self.pool)
            local_index_loss += hun_loss
        return local_index_loss, new_index

    def forward(self, degree, neighbors, encoder_outputs):
        """
        Parameters
        ----------
        degree: Tensor [n,], degree of nodes
        neighbors: Tensor [n, N_n], neighbor features of each node in a graph
        encoder_outputs: List of Tensor [n, f_dim], each element of the list is the output of each layers of encoder. 0 element is the input of encoder, i element is the output of i_th layer of encoder

        Returns
        -------
        loss: Tensor, the loss computed with the method of NWR-GAE
        """
        node_features = encoder_outputs[-1]  # [n, f_dim]
        # degree
        # degree_prd = self.degree_decoding(node_features)
        # degree = degree.unsqueeze(-1)  # [n, 1]
        # degree_loss = self.degree_loss_func(degree_prd, degree.float())

        loss_list = []
        feature_loss_list = []
        # Sample multiple times to remove noise
        for _ in range(3):
            h0 = encoder_outputs[0]  # [n, f_dim]
            feature_losses = self.feature_loss_func(h0, self.feature_decoder(node_features))
            feature_loss_list.append(feature_losses)

            local_index_loss_sum = 0
            indexes = torch.arange(node_features.shape[0])  # [n,]
            for i in range(len(encoder_outputs)-1, 1, -1):
                local_index_loss, new_index = self.reconstruction_neighbors(self.layer_generators[i-2], indexes, neighbors, encoder_outputs[i], encoder_outputs[i-1])
                local_index_loss_sum += local_index_loss
            loss_list.append(local_index_loss_sum)
        loss_list = torch.stack(loss_list)
        h_loss = torch.mean(loss_list)
        feature_loss_list = torch.stack(feature_loss_list)
        feature_loss = torch.mean(feature_loss_list)
        # loss = self.lambda_loss1 * h_loss + self.degree_lambda * degree_loss + self.lambda_loss2 * feature_loss
        loss = self.lambda_loss1 * h_loss + self.lambda_loss2 * feature_loss
        return loss