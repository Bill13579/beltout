# Manual utility script, modify before use!

from pathlib import Path
from safetensors.torch import save_file, load_file

import glob, os, torch

CHECKPOINT_DIR = "checkpoints"

if __name__ == "__main__":
    # full_state_dict = load_file(Path(CHECKPOINT_DIR) / "s3gen-orig.safetensors")
    # print("--- All Keys in the Original Model ---")
    # for key in full_state_dict.keys():
    #     print(key)
    
    # component_state_dict = {
    #     key: tensor for key, tensor in full_state_dict.items() 
    #     if not key.startswith("flow.encoder.") and not key.startswith("flow.input_embedding.") and not key.startswith("flow.decoder.") and key.startswith("flow.")
    # }
    # cleaned_component_state_dict = {
    #     key.removeprefix("flow."): tensor 
    #     for key, tensor in component_state_dict.items()
    # }
    # save_file(cleaned_component_state_dict, "flow.safetensors")
    
    checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, "*.pth"))
    if not checkpoints:
        print("No checkpoints found in directory!")
    
    for checkpoint_path in checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

        # print("--- All Keys in the Original Checkpoint ---")
        # print("--- Decoder ---")
        # for key in checkpoint['decoder_state_dict'].keys():
        #     print(key)
        # print("--- F0 Projector ---")
        # for key in checkpoint['f0_projector_state_dict'].keys():
        #     print(key)
        print(checkpoint_path)
        
        # save_file(checkpoint['decoder_state_dict'], os.path.basename(checkpoint_path).replace("training_step_", "cfm_step_").replace(".pth", ".safetensors"))
        # save_file(checkpoint['f0_projector_state_dict'], os.path.basename(checkpoint_path).replace("training_step_", "pitchmvmt_step_").replace(".pth", ".safetensors"))



