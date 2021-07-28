"""
Tree Model
"""
from typing import Dict, List

import numpy as np
from pandas import DataFrame

from regmod.composite_models.base import BaseModel
from regmod.composite_models.composite import CompositeModel
from regmod.composite_models.interface import NodeModel


class TreeModel(CompositeModel):
    """
    Tree Model with hierarchy structure. This model is also called cascade.
    """

    def _fit(self, model: NodeModel, **fit_options):
        model.fit(**fit_options)
        prior = model.get_posterior()
        for sub_model in model.children.named_lists[0]:
            sub_model.set_prior(prior)
            self._fit(sub_model, **fit_options)

    def fit(self, **fit_options):
        if len(self.children.named_lists[1]) != 1:
            raise ValueError(f"{type(self).__name__} must only have one "
                             "computational tree.")
        self._fit(self.children.named_lists[1][0], **fit_options)

    @classmethod
    def get_simple_tree(cls, name: str, *args, **kwargs) -> "TreeModel":
        return cls(name, models=[get_simple_basetree(*args, **kwargs)])


def get_simple_basetree(df: DataFrame,
                        col_label: List[str],
                        model_specs: Dict,
                        var_masks: Dict[str, float] = None,
                        lvl_masks: List[float] = None) -> BaseModel:
    # check data before create model
    data = model_specs["data"]
    variables = model_specs["variables"]
    data.attach_df(df)
    for v in variables:
        v.check_data(data)

    # create model
    model = BaseModel(**model_specs)
    data.detach_df()
    if len(col_label) == 0:
        return model

    # process masks
    if var_masks is None:
        var_masks = {v.name: np.ones(v.size) for v in variables}
    if lvl_masks is None:
        lvl_masks = [1.0]*len(col_label)
    mask = {name: prior*lvl_masks[0] for name, prior in var_masks.items()}
    model_specs["prior_mask"] = mask

    # create children model
    df_group = df.groupby(col_label[0])
    for name in df_group.groups.keys():
        model_specs["name"] = name
        model.append(get_simple_basetree(df_group.get_group(name),
                                         col_label[1:],
                                         model_specs,
                                         var_masks=var_masks,
                                         lvl_masks=lvl_masks[1:]))

    return model
