import torch

full_weight = torch.load('./small.pt')

encoder_weight = {
    'dims': full_weight['dims'],
    'model_state_dict': {}
}

for key in full_weight['model_state_dict'].keys():
    if key.startswith('encoder'):
        if "positional_embedding" in key:
            continue
        encoder_weight['model_state_dict'][key.replace('encoder.', '')] = full_weight['model_state_dict'][key]

torch.save(encoder_weight, './small_encoder.pt')


