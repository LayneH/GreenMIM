# Written by Lang Huang (laynehuang@outlook.com)
# All rights reserved.
# --------------------------------------------------------
import pickle as pkl
import sys
import torch

def load_pretrained(ckpt_path, save_path, model_type='swin'):
    print(f">>>>>>>>>> Load from {ckpt_path} ..........")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    checkpoint_model = checkpoint['model']
    
    if any([True if 'encoder.' in k else False for k in checkpoint_model.keys()]):
        checkpoint_model = {k.replace('encoder.', ''): v for k, v in checkpoint_model.items() if k.startswith('encoder.')}
        print('Detect pre-trained model, remove [encoder.] prefix.')
    else:
        print('Detect non-pre-trained model, pass without doing anything.')

    if model_type == 'swin':
        print(f">>>>>>>>>> Remapping pre-trained keys for SWIN ..........")
        checkpoint = remap_pretrained_keys_swin(checkpoint_model)
    elif model_type == 'vit':
        raise NotImplementedError()
        print(f">>>>>>>>>> Remapping pre-trained keys for VIT ..........")
        checkpoint = remap_pretrained_keys_vit(checkpoint_model)
    else:
        raise NotImplementedError
    
    with open(save_path, "wb") as f:
        torch.save(checkpoint_model, f)
    
    del checkpoint
    torch.cuda.empty_cache()
    print(f">>>>>>>>>> loaded successfully '{ckpt_path}'")
    

def remap_pretrained_keys_swin(checkpoint_model):
    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [k for k in checkpoint_model.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del checkpoint_model[k]

    # delete relative_coords_table since we always re-init it
    relative_coords_table_keys = [k for k in checkpoint_model.keys() if "relative_coords_table" in k]
    for k in relative_coords_table_keys:
        del checkpoint_model[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in checkpoint_model.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del checkpoint_model[k]

    return checkpoint_model

if __name__ == '__main__':
    input = sys.argv[1]
    output = sys.argv[2]
    load_pretrained(input, output)