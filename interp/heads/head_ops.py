import torch
from interp.helper import get_transformer_layers


def head_geometry(attention):
    projection = attention.o_proj
    head_dim = getattr(attention, 'head_dim', None)
    if head_dim is None:
        config = attention.config
        head_dim = getattr(config, 'head_dim', None) or config.hidden_size // config.num_attention_heads
    if projection.in_features % head_dim:
        raise ValueError('Output projection input is not divisible by head_dim')
    return projection.in_features // head_dim, head_dim


class AttentionHeadMaskingHook:
    def __init__(self, heads_to_mask):
        self.heads_to_mask = set(heads_to_mask)
        self.hooks = []

    def register_hooks(self, model):
        self.remove_hooks()
        layers = get_transformer_layers(model)
        for layer_idx, head_idx in self.heads_to_mask:
            if not 0 <= layer_idx < len(layers):
                raise ValueError(f'Invalid layer {layer_idx}')
            n_heads, _ = head_geometry(layers[layer_idx].self_attn)
            if not 0 <= head_idx < n_heads:
                raise ValueError(f'Invalid head {head_idx} in layer {layer_idx}')
        for layer_idx in sorted({l for l, _ in self.heads_to_mask}):
            attention = layers[layer_idx].self_attn
            n_heads, head_dim = head_geometry(attention)
            selected = sorted(h for l, h in self.heads_to_mask if l == layer_idx)
            def mask(module, inputs, selected=selected, n_heads=n_heads, head_dim=head_dim):
                values = inputs[0].clone()
                heads = values.reshape(*values.shape[:-1], n_heads, head_dim)
                heads[..., selected, :] = 0
                return (heads.reshape_as(values), *inputs[1:])
            self.hooks.append(attention.o_proj.register_forward_pre_hook(mask))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def extract_attention_head_data(language_model, embeddings, attention_mask):
    layers = language_model.layers
    weights, values = {}, {}
    hooks = []
    original = language_model.config._attn_implementation
    try:
        language_model.config._attn_implementation = 'eager'
        for idx, layer in enumerate(layers):
            n_heads, head_dim = head_geometry(layer.self_attn)
            def capture_input(module, inputs, idx=idx, n_heads=n_heads, head_dim=head_dim):
                x = inputs[0]
                values[idx] = x.reshape(*x.shape[:-1], n_heads, head_dim)[0].permute(1, 0, 2).detach().cpu()
            def capture_weights(module, inputs, outputs, idx=idx):
                if len(outputs) < 2 or outputs[1] is None:
                    raise RuntimeError('Attention weights unavailable; eager attention is required')
                weights[idx] = outputs[1][0].detach().cpu()
            hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(capture_input))
            hooks.append(layer.self_attn.register_forward_hook(capture_weights))
        with torch.no_grad():
            language_model(inputs_embeds=embeddings, attention_mask=attention_mask, output_attentions=True, use_cache=False)
        if len(values) != len(layers) or len(weights) != len(layers):
            raise RuntimeError('Missing attention layers')
        return [weights[i] for i in range(len(layers))], [values[i] for i in range(len(layers))]
    finally:
        language_model.config._attn_implementation = original
        for hook in hooks:
            hook.remove()


def projected_head_contributions(attention, concatenated_heads):
    n_heads, head_dim = head_geometry(attention)
    values = concatenated_heads.reshape(*concatenated_heads.shape[:-1], n_heads, head_dim)
    weight = attention.o_proj.weight.reshape(attention.o_proj.out_features, n_heads, head_dim)
    return torch.einsum('...hd,ohd->...ho', values, weight)
