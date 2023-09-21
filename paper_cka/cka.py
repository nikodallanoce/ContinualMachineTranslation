import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial
from warnings import warn
from typing import List, Dict, Tuple, Set, Optional
import matplotlib.pyplot as plt
from .utils import add_colorbar, cka, gram_rbf, gram_linear, linear_CKA


class CKA:
    def __init__(self,
                 model1: nn.Module,
                 model2: nn.Module,
                 model1_name: str = None,
                 model2_name: str = None,
                 model1_layers: List[str] = None,
                 model2_layers: List[str] = None,
                 device: str = 'cpu'):
        """

        :param model1: (nn.Module) Neural Network 1
        :param model2: (nn.Module) Neural Network 2
        :param model1_name: (str) Name of model 1
        :param model2_name: (str) Name of model 2
        :param model1_layers: (List) List of layers to extract features from
        :param model2_layers: (List) List of layers to extract features from
        :param device: Device to run the model
        """

        self.model1 = model1
        self.model2 = model2

        self.device = device

        self.model1_info = {}
        self.model2_info = {}

        if model1_name is None:
            self.model1_info['Name'] = model1.__repr__().split('(')[0]
        else:
            self.model1_info['Name'] = model1_name

        if model2_name is None:
            self.model2_info['Name'] = model2.__repr__().split('(')[0]
        else:
            self.model2_info['Name'] = model2_name

        if self.model1_info['Name'] == self.model2_info['Name']:
            warn(f"Both model have identical names - {self.model2_info['Name']}. "
                 "It may cause confusion when interpreting the results. "
                 "Consider giving unique names to the models :)")

        self.model1_info['Layers'] = []
        self.model2_info['Layers'] = []

        self.model1_features = {}
        self.model2_features = {}

        if len(list(model1.modules())) > 150 and model1_layers is None:
            warn("Model 1 seems to have a lot of layers. "
                 "Consider giving a list of layers whose features you are concerned with "
                 "through the 'model1_layers' parameter. Your CPU/GPU will thank you :)")

        self.model1_layers = model1_layers

        if len(list(model2.modules())) > 150 and model2_layers is None:
            warn("Model 2 seems to have a lot of layers. "
                 "Consider giving a list of layers whose features you are concerned with "
                 "through the 'model2_layers' parameter. Your CPU/GPU will thank you :)")

        self.model2_layers = model2_layers

        self._insert_hooks()
        assert set(self.model1_info['Layers']) == set(model1_layers)
        assert set(self.model2_info['Layers']) == set(model2_layers)

        self.model1 = self.model1.to(self.device) if self.device != self.model1.device else self.model1
        self.model2 = self.model2.to(self.device) if self.device != self.model2.device else self.model2

        self.model1.eval()
        self.model2.eval()

    def _log_layer(self,
                   model: str,
                   name: str,
                   layer: nn.Module,
                   inp: torch.Tensor,
                   out: torch.Tensor):

        if model == "model1":
            self.model1_features[name] = out

        elif model == "model2":
            self.model2_features[name] = out

        else:
            raise RuntimeError("Unknown model name for _log_layer.")

    def _insert_hooks(self):
        # Model 1
        for name, layer in self.model1.named_modules():
            if self.model1_layers is not None:
                if name in self.model1_layers:
                    self.model1_info['Layers'] += [name]
                    # layer.register_forward_hook(partial(self._log_layer, "model1", name))
                    layer.register_forward_hook(partial(self._log_layer, "model1", name))
            else:
                self.model1_info['Layers'] += [name]
                # layer.register_module_forward_hook(partial(self._log_layer, "model2", name))
                layer.register_forward_hook(partial(self._log_layer, "model1", name))

        # Model 2
        for name, layer in self.model2.named_modules():
            if self.model2_layers is not None:
                if name in self.model2_layers:
                    self.model2_info['Layers'] += [name]
                    layer.register_forward_hook(partial(self._log_layer, "model2", name))
            else:

                self.model2_info['Layers'] += [name]
                layer.register_forward_hook(partial(self._log_layer, "model2", name))

    # def compare(self,
    #             dataloader1: DataLoader,
    #             dataloader2: DataLoader = None,
    #             debiased: bool = False,
    #             gram_threshold: float = None) -> None:
    #     """
    #     Computes the feature similarity between the models on the
    #     given datasets.
    #     :param dataloader1: (DataLoader)
    #     :param dataloader2: (DataLoader) If given, model 2 will run on this
    #                         dataset. (default = None)
    #     """
    #     with torch.no_grad():
    #         if dataloader2 is None:
    #             warn("Dataloader for Model 2 is not given. Using the same dataloader for both models.")
    #             dataloader2 = dataloader1
    #
    #         self.model1_info['Dataset'] = dataloader1.dataset.__repr__().split('\n')[0]
    #         self.model2_info['Dataset'] = dataloader2.dataset.__repr__().split('\n')[0]
    #
    #         N = len(self.model1_layers) if self.model1_layers is not None else len(list(self.model1.modules()))
    #         M = len(self.model2_layers) if self.model2_layers is not None else len(list(self.model2.modules()))
    #
    #         num_batches = min(len(dataloader1), len(dataloader2))
    #         # self.cka = torch.zeros(N, M, 3)
    #         self.cka = torch.zeros(N, M, device=self.device)
    #         b = 0
    #
    #         for x1, x2 in tqdm(zip(dataloader1, dataloader2), desc="| Comparing features |", total=num_batches):
    #
    #             self.model1_features = {}
    #             self.model2_features = {}
    #             for e1, e2 in zip(x1, x2):
    #                 if isinstance(x1[e1], torch.Tensor):
    #                     x1[e1] = x1[e1].to(self.device)
    #                 if isinstance(x2[e2], torch.Tensor):
    #                     x2[e2] = x2[e2].to(self.device)
    #             _ = self.model1(**x1)
    #             _ = self.model2(**x2)
    #
    #             for i, (name1, feat1) in enumerate(self.model1_features.items()):
    #                 # X: torch.Tensor = feat1.flatten(1)
    #                 x_last_sh: torch.Tensor = feat1.shape[-1]
    #                 X: torch.Tensor = feat1.view(-1, x_last_sh)
    #                 for j, (name2, feat2) in enumerate(self.model2_features.items()):
    #                     # Y: torch.Tensor = feat2.flatten(1)
    #                     y_last_sh: torch.Tensor = feat2.shape[-1]
    #                     Y: torch.Tensor = feat2.view(-1, y_last_sh)
    #                     # assert K.shape == L.shape, f"Feature shape mistach! {K.shape}, {L.shape}"
    #
    #                     # self.cka[i, j] += cka((X @ X.T), (Y @ Y.T), debiased=False) / num_batches
    #                     if gram_threshold is None:
    #                         c = linear_CKA(X, Y)
    #                         c_r = linear_CKA(Y, X)
    #                         self.cka[i, j] += c
    #                         self.cka[j, i] += c
    #
    #                     else:
    #                         self.cka[i, j] += cka(gram_rbf(X, gram_threshold), gram_rbf(Y, gram_threshold),
    #                                               debiased=debiased) / num_batches
    #             b = b + 1
    #     assert not torch.isnan(self.cka).any(), "HSIC computation resulted in NANs"

    def compare(self,
                dataloader1: DataLoader,
                dataloader2: DataLoader = None,
                debiased: bool = False,
                gram_threshold: float = None) -> float:
        """
        Computes the feature similarity between the models on the
        given datasets.
        :param dataloader1: (DataLoader)
        :param dataloader2: (DataLoader) If given, model 2 will run on this
                            dataset. (default = None)
        """
        with torch.no_grad():
            if dataloader2 is None:
                warn("Dataloader for Model 2 is not given. Using the same dataloader for both models.")
                dataloader2 = dataloader1

            self.model1_info['Dataset'] = dataloader1.dataset.__repr__().split('\n')[0]
            self.model2_info['Dataset'] = dataloader2.dataset.__repr__().split('\n')[0]

            N = len(self.model1_layers) if self.model1_layers is not None else len(list(self.model1.modules()))
            M = len(self.model2_layers) if self.model2_layers is not None else len(list(self.model2.modules()))

            # self.cka = torch.zeros(N, M, 3)
            self.cka = torch.zeros(N, M, device=self.device)

            num_batches = min(len(dataloader1), len(dataloader2))

            for x1, x2 in tqdm(zip(dataloader1, dataloader2), desc="| Comparing features |", total=num_batches):
                self.model1_features = {}
                self.model2_features = {}
                for e1, e2 in zip(x1, x2):
                    if isinstance(x1[e1], torch.Tensor):
                        x1[e1] = x1[e1].to(self.device)
                    if isinstance(x2[e2], torch.Tensor):
                        x2[e2] = x2[e2].to(self.device)
                _ = self.model1(**x1)
                _ = self.model2(**x2)

                for i, (name1, feat1) in enumerate(self.model1_features.items()):
                    # X: torch.Tensor = feat1.flatten(1)
                    x_last_sh: torch.Tensor = feat1.shape[-1]
                    X: torch.Tensor = feat1.view(-1, x_last_sh)
                    for j, (name2, feat2) in enumerate(self.model2_features.items()):
                        # Y: torch.Tensor = feat2.flatten(1)
                        y_last_sh: torch.Tensor = feat2.shape[-1]
                        Y: torch.Tensor = feat2.view(-1, y_last_sh)
                        # assert K.shape == L.shape, f"Feature shape mistach! {K.shape}, {L.shape}"

                        # self.cka[i, j] += cka((X @ X.T), (Y @ Y.T), debiased=False) / num_batches
                        if gram_threshold is None:
                            r = cka(gram_linear(X), gram_linear(Y), debiased=debiased) / num_batches
                            # r = linear_CKA(X, Y)
                            self.cka[i, j] += r
                        else:
                            self.cka[i, j] += cka(gram_rbf(X, gram_threshold), gram_rbf(Y, gram_threshold),
                                                  debiased=debiased) / num_batches
        assert not torch.isnan(self.cka).any(), "HSIC computation resulted in NANs"
        return float(torch.mean(torch.diag(self.cka)).cpu())

    def export(self) -> Dict:
        """
        Exports the CKA data along with the respective model layer names.
        :return:
        """
        return {
            "model1_name": self.model1_info['Name'],
            "model2_name": self.model2_info['Name'],
            "CKA": self.cka,
            "model1_layers": self.model1_info['Layers'],
            "model2_layers": self.model2_info['Layers'],
            "dataset1_name": self.model1_info['Dataset'],
            "dataset2_name": self.model2_info['Dataset'],

        }

    def plot_results(self,
                     save_path: str = None,
                     title: str = None,
                     show_ticks_labels: bool = False,
                     short_tick_labels_splits: Optional[int] = None,
                     show_annotations: bool = True,
                     show_img: bool = True):
        import seaborn as sns
        # fig, ax = plt.subplots()
        ax = sns.heatmap(self.cka.cpu(), annot=show_annotations, cmap="magma")
        ax.invert_yaxis()
        ax.set_xlabel(f"Layers {self.model2_info['Name']}", fontsize=10)
        ax.set_ylabel(f"Layers {self.model1_info['Name']}", fontsize=10)
        if show_ticks_labels:
            if short_tick_labels_splits is None:
                ax.set_xticklabels(self.model2_info['Layers'])
                ax.set_yticklabels(self.model1_info['Layers'])
            else:
                ax.set_xticklabels(
                    ["-".join(module.split(".")[-short_tick_labels_splits:]) for module in self.model2_info['Layers']])
                ax.set_yticklabels(
                    ["-".join(module.split(".")[-short_tick_labels_splits:]) for module in self.model1_info['Layers']])
            plt.xticks(rotation=90)
            plt.yticks(rotation=0)
        chart_title = title
        if title is not None:
            ax.set_title(f"{title}", fontsize=10)
        else:
            chart_title = f"{self.model1_info['Name']} vs {self.model2_info['Name']}"
            ax.set_title(chart_title, fontsize=10)

        # add_colorbar(im)
        # plt.tight_layout()
        # plt.figure(figsize=fig_size, dpi=300)
        if save_path is not None:
            chart_title = chart_title.replace("/", "-")
            path_rel = f"{save_path}/{chart_title}.png"
            plt.savefig(path_rel, dpi=400, bbox_inches="tight")
        if show_img:
            plt.show()

    def sanity_check(self) -> Set[Tuple[int, int]]:
        idx_sanity_fail: Set[Tuple[int, int]] = set()
        for i in range(self.cka.shape[0]):
            for j in range(self.cka.shape[1]):
                if self.cka[i, j] > self.cka[i, i]:
                    idx_sanity_fail.add((j, i))  # inverted index due to inverted y-axis

        return idx_sanity_fail
