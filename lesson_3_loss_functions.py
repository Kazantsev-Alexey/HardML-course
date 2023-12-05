import math

import numpy as np
import torch
from catboost.datasets import msrank_10k
from sklearn.preprocessing import StandardScaler

from typing import List



class ListNet(torch.nn.Module):
    def __init__(self, num_input_features: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.model = torch.nn.Sequential(
            torch.nn.Linear(num_input_features, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, input_1: torch.Tensor) -> torch.Tensor:
        logits = self.model(input_1)
        return logits


class Solution:
    def __init__(self, n_epochs: int = 5, listnet_hidden_dim: int = 30,
                 lr: float = 0.001, ndcg_top_k: int = 10):
        self._prepare_data()
        self.num_input_features = self.X_train.shape[1]
        self.ndcg_top_k = ndcg_top_k
        self.n_epochs = n_epochs

        self.model = self._create_model(
            self.num_input_features, listnet_hidden_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def _get_data(self) -> List[np.ndarray]:
        train_df, test_df = msrank_10k()

        X_train = train_df.drop([0, 1], axis=1).values
        y_train = train_df[0].values
        query_ids_train = train_df[1].values.astype(int)

        X_test = test_df.drop([0, 1], axis=1).values
        y_test = test_df[0].values
        query_ids_test = test_df[1].values.astype(int)

        return [X_train, y_train, query_ids_train, X_test, y_test, query_ids_test]
    
    def _scale_features_in_query_groups(self, inp_feat_array: np.ndarray,
                                        inp_query_ids: np.ndarray) -> np.ndarray:
        scaler = StandardScaler()
        for i in np.unique(inp_query_ids):
            inp_feat_array[np.where(inp_query_ids==i)] = scaler.fit_transform(inp_feat_array[np.where(inp_query_ids==i)])

        return inp_feat_array
    
    def _prepare_data(self) -> None:
        (X_train, self.ys_train, self.query_ids_train,
            X_test, self.ys_test, self.query_ids_test) = self._get_data()
        
        self.X_train = self._scale_features_in_query_groups(X_train, self.query_ids_train)
        self.X_test = self._scale_features_in_query_groups(X_test, self.query_ids_test)
        
        self.X_train = torch.from_numpy(self.X_train).float()
        self.X_test = torch.from_numpy(self.X_test).float()
        self.ys_train = torch.from_numpy(self.ys_train).float()
        self.ys_test = torch.from_numpy(self.ys_test).float()

    def _create_model(self, listnet_num_input_features: int,
                      listnet_hidden_dim: int) -> torch.nn.Module:
        torch.manual_seed(0)
        net = ListNet(num_input_features=listnet_num_input_features, hidden_dim=listnet_hidden_dim)
        return net

    def fit(self) -> List[float]:
        valid_ngcg_list = []
        for epoch in range(self.n_epochs):
            print('epoch number', epoch)
            self._train_one_epoch()
            ndcg_on_epoch = self._eval_test_set()
            valid_ngcg_list.append(ndcg_on_epoch)
        return valid_ngcg_list
    
    def _calc_loss(self, batch_ys: torch.FloatTensor,
                   batch_pred: torch.FloatTensor) -> torch.FloatTensor:
        ys = torch.softmax(batch_ys, dim=0)
        preds = torch.softmax(batch_pred, dim=0)
        return -torch.sum(ys * torch.log(preds))

    def _train_one_epoch(self) -> None:
        self.model.train()
        for i in np.unique(self.query_ids_train):
            batch_X = self.X_train[np.where(self.query_ids_train==i)].float()
            batch_ys = self.ys_train[np.where(self.query_ids_train==i)].float()
            
            self.optimizer.zero_grad()
            
            batch_pred = self.model(batch_X).reshape(-1, )
            batch_loss = self._calc_loss(batch_ys, batch_pred)
            batch_loss.backward(retain_graph=True)
            self.optimizer.step()

    def _eval_test_set(self) -> float:
        with torch.no_grad():
            self.model.eval()
            ndcgs = []
            
            for i in np.unique(self.query_ids_test):
                valid_X = self.X_test[np.where(self.query_ids_test==i)].float()
                valid_ys = self.ys_test[np.where(self.query_ids_test==i)].float()
                
                valid_pred = self.model(valid_X)
                ndcg_score = self._ndcg_k(valid_ys, valid_pred, self.ndcg_top_k)
                ndcgs.append(ndcg_score)
            return np.mean(ndcgs)

    def compute_gain(self, y_value: float, gain_scheme: str) -> float:
        if gain_scheme=='const':
            return y_value
        else:
            return  (2**y_value - 1)


    def dcg(self, ys_true: torch.Tensor, ys_pred: torch.Tensor,ndcg_top_k: int, gain_scheme: str) -> float:
        _, argsort = torch.sort(ys_pred, descending=True, dim=0)
        ys_true_sorted = ys_true[argsort][:ndcg_top_k]
        ret = 0
        for idx, cur_y in enumerate(ys_true_sorted,1):
            gain = self.compute_gain(cur_y.item(), gain_scheme)
            ret += gain / math.log2(idx+1)

        return ret
    
    def _ndcg_k(self, ys_true: torch.Tensor, ys_pred: torch.Tensor, 
                ndcg_top_k: int, gain_scheme: str = 'exp') -> float:
        dcg_result = self.dcg(ys_true,ys_pred,ndcg_top_k,gain_scheme)
        idcg_result = self.dcg(ys_true,ys_true,ndcg_top_k,gain_scheme)
        
        if idcg_result == 0:
            result = 0
        else:
            result = dcg_result/idcg_result
        return result
