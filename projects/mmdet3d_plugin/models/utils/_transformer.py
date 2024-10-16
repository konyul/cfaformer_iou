# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
import warnings
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import (Linear, build_activation_layer, build_conv_layer,
                      build_norm_layer)
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from mmcv.utils import (ConfigDict, build_from_cfg, deprecated_api_warning,
                        to_2tuple)
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.registry import (ATTENTION, FEEDFORWARD_NETWORK, POSITIONAL_ENCODING,
                       TRANSFORMER_LAYER, TRANSFORMER_LAYER_SEQUENCE)

# Avoid BC-breaking of importing MultiScaleDeformableAttention from this file
try:
    from mmcv.ops.multi_scale_deform_attn import \
        MultiScaleDeformableAttention  # noqa F401
    warnings.warn(
        ImportWarning(
            '``MultiScaleDeformableAttention`` has been moved to '
            '``mmcv.ops.multi_scale_deform_attn``, please change original path '  # noqa E501
            '``from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention`` '  # noqa E501
            'to ``from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention`` '  # noqa E501
        ))

except ImportError:
    warnings.warn('Fail to import ``MultiScaleDeformableAttention`` from '
                  '``mmcv.ops.multi_scale_deform_attn``, '
                  'You should install ``mmcv-full`` if you need this module. ')


def build_positional_encoding(cfg, default_args=None):
    """Builder for Position Encoding."""
    return build_from_cfg(cfg, POSITIONAL_ENCODING, default_args)


def build_attention(cfg, default_args=None):
    """Builder for attention."""
    return build_from_cfg(cfg, ATTENTION, default_args)


def build_feedforward_network(cfg, default_args=None):
    """Builder for feed-forward network (FFN)."""
    return build_from_cfg(cfg, FEEDFORWARD_NETWORK, default_args)


def build_transformer_layer(cfg, default_args=None):
    """Builder for transformer layer."""
    return build_from_cfg(cfg, TRANSFORMER_LAYER, default_args)


def build_transformer_layer_sequence(cfg, default_args=None):
    """Builder for transformer encoder and transformer decoder."""
    return build_from_cfg(cfg, TRANSFORMER_LAYER_SEQUENCE, default_args)


@TRANSFORMER_LAYER.register_module()
class cfaBaseTransformerLayer(BaseModule):
    """Base `TransformerLayer` for vision transformer.

    It can be built from `mmcv.ConfigDict` and support more flexible
    customization, for example, using any number of `FFN or LN ` and
    use different kinds of `attention` by specifying a list of `ConfigDict`
    named `attn_cfgs`. It is worth mentioning that it supports `prenorm`
    when you specifying `norm` as the first element of `operation_order`.
    More details about the `prenorm`: `On Layer Normalization in the
    Transformer Architecture <https://arxiv.org/abs/2002.04745>`_ .

    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None )):
            Configs for `self_attention` or `cross_attention` modules,
            The order of the configs in the list should be consistent with
            corresponding attentions in operation_order.
            If it is a dict, all of the attention modules in operation_order
            will be built with this config. Default: None.
        ffn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None )):
            Configs for FFN, The order of the configs in the list should be
            consistent with corresponding ffn in operation_order.
            If it is a dict, all of the attention modules in operation_order
            will be built with this config.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Support `prenorm` when you specifying first element as `norm`.
            Default：None.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): Key, Query and Value are shape
            of (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
    """

    def __init__(self,
                 attn_cfgs=None,
                 ffn_cfgs=dict(
                     type='FFN',
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True),
                 ),
                 operation_order=None,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None,
                 batch_first=False,
                 **kwargs):
                 

        deprecated_args = dict(
            feedforward_channels='feedforward_channels',
            ffn_dropout='ffn_drop',
            ffn_num_fcs='num_fcs')
        for ori_name, new_name in deprecated_args.items():
            if ori_name in kwargs:
                warnings.warn(
                    f'The arguments `{ori_name}` in BaseTransformerLayer '
                    f'has been deprecated, now you should set `{new_name}` '
                    f'and other FFN related arguments '
                    f'to a dict named `ffn_cfgs`. ', DeprecationWarning)
                ffn_cfgs[new_name] = kwargs[ori_name]

        super().__init__(init_cfg)
        self.batch_first = batch_first
        
        assert set(operation_order) & {
            'self_attn', 'norm', 'ffn', 'cross_attn'} == \
            set(operation_order), f'The operation_order of' \
            f' {self.__class__.__name__} should ' \
            f'contains all four operation type ' \
            f"{['self_attn', 'norm', 'ffn', 'cross_attn']}"

        num_attn = operation_order.count('self_attn') + operation_order.count(
            'cross_attn')
        if isinstance(attn_cfgs, dict):
            attn_cfgs = [copy.deepcopy(attn_cfgs) for _ in range(num_attn)]
        else:
            assert num_attn == len(attn_cfgs), f'The length ' \
                f'of attn_cfg {num_attn} is ' \
                f'not consistent with the number of attention' \
                f'in operation_order {operation_order}.'

        self.num_attn = num_attn
        self.operation_order = operation_order
        self.norm_cfg = norm_cfg
        self.pre_norm = operation_order[0] == 'norm'
        self.attentions = ModuleList()

        index = 0
        for operation_name in operation_order:
            if operation_name in ['self_attn', 'cross_attn']:
                if 'batch_first' in attn_cfgs[index]:
                    assert self.batch_first == attn_cfgs[index]['batch_first']
                else:
                    attn_cfgs[index]['batch_first'] = self.batch_first
                attention = build_attention(attn_cfgs[index])
                # Some custom attentions used as `self_attn`
                # or `cross_attn` can have different behavior.
                attention.operation_name = operation_name
                self.attentions.append(attention)
                index += 1

        self.embed_dims = self.attentions[0].embed_dims

        self.ffns = ModuleList()
        num_ffns = operation_order.count('ffn')
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = ConfigDict(ffn_cfgs)
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = [copy.deepcopy(ffn_cfgs) for _ in range(num_ffns)]
        assert len(ffn_cfgs) == num_ffns
        for ffn_index in range(num_ffns):
            if 'embed_dims' not in ffn_cfgs[ffn_index]:
                ffn_cfgs[ffn_index]['embed_dims'] = self.embed_dims
            else:
                assert ffn_cfgs[ffn_index]['embed_dims'] == self.embed_dims
            self.ffns.append(
                build_feedforward_network(ffn_cfgs[ffn_index],
                                          dict(type='FFN')))

        self.norms = ModuleList()
        num_norms = operation_order.count('norm')
        for _ in range(num_norms):
            self.norms.append(build_norm_layer(norm_cfg, self.embed_dims)[1])
        
        self.mlp = nn.Sequential(
            nn.Linear(256*3,256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256,3)
        )
        self.mlpv2 = nn.Sequential(
            nn.Linear(256,64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64,3)
        )
        
        
    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                        f'attn_masks {len(attn_masks)} must be equal ' \
                        f'to the number of attention in ' \
                        f'operation_order {self.num_attn}'
        weight_list = []
        ca_query = query.clone()
        for layer in self.operation_order:
            if layer == 'self_attn':
                temp_key = temp_value = query
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1
                if 'failure_pred' in kwargs and kwargs['failure_pred'] and (attn_index-1)%2==0:
                    _attn_masks = ~attn_masks[attn_index-1]
                    _num_queries,batch_size, _query_dim = query.shape
                    _attn_masks = _attn_masks.view(batch_size,self.attentions[0].num_heads,_num_queries,_num_queries)
                    _attn_masks = _attn_masks[:,0,:,:]
                    attn_query = query.clone().transpose(0,1)
                    non_zero_masks = _attn_masks.nonzero(as_tuple=True)
                    attn_query = attn_query[non_zero_masks[0],non_zero_masks[2]].view(batch_size, _num_queries, 3, _query_dim)
                    attn_query = attn_query.reshape(batch_size,_num_queries,-1)
                    attn_query = self.mlp(attn_query)
                    _attn_weight = attn_query.permute(1,0,2)
                    # query = query * _attn_weight
                    weight_list.append(_attn_weight)

            elif layer == 'cross_attn':
                if 'failure_pred' in kwargs and kwargs['failure_pred']:
                    query = ca_query
                    num_queries, batch_size, _ = _attn_weight.shape
                    total_elements, _, _ = key.shape
                    attn_mask = torch.zeros((batch_size, num_queries, total_elements), dtype=torch.bool, device=query.device)
                    _attn_idx = _attn_weight.argmax(-1).permute(1,0)
                    mask_0 = (_attn_idx == 0).unsqueeze(-1)
                    attn_mask[:, :, :] |= mask_0
                    mask_1 = (_attn_idx == 1).unsqueeze(-1)
                    attn_mask[:, :, :32400] |= mask_1
                    mask_2 = (_attn_idx == 2).unsqueeze(-1)
                    attn_mask[:, :, 32400:] |= mask_2
                    attn_mask = attn_mask.repeat_interleave(8,dim=0)
                    attn_masks[attn_index] = ~attn_mask
                if 'locality_aware_failure_pred' in kwargs and kwargs['locality_aware_failure_pred'] and attn_index == 0:
                    _query_list = []
                    new_key = key
                    new_value = value
                    new_key_pos = key_pos
                    _idx = 0
                    # if _idx > 0:
                    #     new_key = key[(_idx-1)*32400:(_idx)*32400]
                    #     new_value = value[(_idx-1)*32400:(_idx)*32400]
                    #     new_key_pos = key_pos[(_idx-1)*32400:(_idx)*32400]
                    attn_masks[attn_index][_idx] = attn_masks[attn_index][_idx].repeat_interleave(8,dim=0)
                    _new_query = self.attentions[attn_index](
                        query,
                        new_key,
                        new_value,
                        identity if self.pre_norm else None,
                        query_pos=query_pos,
                        key_pos=new_key_pos,
                        attn_mask=attn_masks[attn_index][_idx],
                        key_padding_mask=key_padding_mask,
                        **kwargs)
                    _query_list.append(_new_query)
                    attn_query = torch.cat(_query_list,dim=-1)
                    _attn_weight = self.mlpv2(attn_query)
                    weight_list.append(_attn_weight)
                elif 'locality_aware_failure_pred' in kwargs and kwargs['locality_aware_failure_pred'] and attn_index == 1:
                    query = ca_query
                    num_queries, batch_size, _ = _attn_weight.shape
                    total_elements, _, _ = key.shape
                    attn_mask = torch.zeros((batch_size, num_queries, total_elements), dtype=torch.bool, device=query.device)
                    _attn_idx = _attn_weight.argmax(-1).permute(1,0)
                    mask_0 = (_attn_idx == 0).unsqueeze(-1)
                    attn_mask[:, :, :] |= mask_0
                    mask_1 = (_attn_idx == 1).unsqueeze(-1)
                    attn_mask[:, :, :32400] |= mask_1
                    mask_2 = (_attn_idx == 2).unsqueeze(-1)
                    attn_mask[:, :, 32400:] |= mask_2
                    attn_mask = attn_mask.repeat_interleave(8,dim=0)
                    attn_masks[attn_index] = ~attn_mask
                else:
                    query = self.attentions[attn_index](
                        query,
                        key,
                        value,
                        identity if self.pre_norm else None,
                        query_pos=query_pos,
                        key_pos=key_pos,
                        attn_mask=attn_masks[attn_index],
                        key_padding_mask=key_padding_mask,
                        **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1
        if len(weight_list) !=0 :
            return query, weight_list
        else:
            return query, None
