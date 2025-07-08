# When training, a good learning rate schedule is to start at a learning rate of 1e-4, decaying 0.9999 per step until it reaches 1e-7.
# At that point, switch to a fixed rate of 1e-7 for polish.
# To switch to the new static rate, make sure to not load the state dict for the optimizer, to not call "scheduler.step()", and do:
"""
new_lr = 1e-7 # The new learning rate you want to use.

print(f"Old learning rate: {optimizer.param_groups[0]['lr']}")

for param_group in optimizer.param_groups:
    param_group['lr'] = new_lr

print(f"New learning rate: {optimizer.param_groups[0]['lr']}")
"""

# To set up a kernel:

# FIRST INSTALL MINIFORGE3

# THEN
#conda install ipykernel
#python -m ipykernel install --user --name=miniforge_py312 --display-name "Python (Miniforge3 Python 3.12)"

# RUN THIS IN A SHELL
#git clone https://github.com/Bill13579/beltout
#cd beltout && pip install -e .

#pip install ipywidgets widgetsnbextension pandas-profiling  #for TQDM in notebooks

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
import torchaudio
import torchcrepe
from tqdm import tqdm
import os
import glob
import signal

from beltout import BeltOutTTM

from arro3_dataset_streamer import Arro3StreamingDataset

# --- Graceful Exit Handler ---
# This global flag will be set by the signal handler.
interrupted = False
def signal_handler(sig, frame):
    global interrupted
    print("\nCtrl+C detected! Finishing current step and saving checkpoint...")
    interrupted = True

# Register the handler for Ctrl+C (SIGINT)
signal.signal(signal.SIGINT, signal_handler)

# Key values or column names for the components to use inside of each batch.
COMPONENTS = ["english1-1", "english1-2", "english1-3", "english1-4", "english2-1", "english2-2", "english2-3", "other-1", "other-2", "other-3", "musical-1", "musical-2", "musical-3", "musical-4", "musical-5", "musical-6", "musical-7", "musical-8", "japanese-1", "japanese-2", "japanese-3", "persian-1", "greek-1", "esd-1", "esd-2", "esd-3", "chinese-1", "chinese-2", "romance-1", "romance-2", "romance-3", "romance-4"]
class AudioDataProcessor:
    def __init__(self, segment_len_s=4):
        self.target_sr = 24000
        self.segment_len_24k = segment_len_s * self.target_sr

    def transform_example(self, example):
        try:
            waveforms = []

            for component in COMPONENTS:
                audio_data = example[component]
                waveform = torch.from_numpy(audio_data['array']).unsqueeze(0).float()
                sr = audio_data['sampling_rate']

                # Resample to the target 24kHz first. This shouldn't be necessary if the data was prepared with `bake`, but we keep it anyways for when it was not.
                if sr != self.target_sr:
                    resampler_24k = torchaudio.transforms.Resample(sr, self.target_sr)
                    waveform_24k = resampler_24k(waveform)
                else:
                    waveform_24k = waveform

                # --- Handle short and long samples gracefully ---
                current_len = waveform_24k.shape[1]
                if current_len > self.segment_len_24k:
                    # If longer than our segment, crop a random chunk. Each audio file tends to be longer than 4s, so doing this immediately gives up a massive boost in data variety for zero cost.
                    start = torch.randint(0, current_len - self.segment_len_24k, (1,)).item()
                    final_waveform = waveform_24k[:, start:start + self.segment_len_24k]
                else:
                    # If shorter or equal, use the whole thing. Padding will be handled later.
                    final_waveform = waveform_24k
                # ---------------------------------------------------------
            
                waveforms.append(final_waveform.squeeze(0))
            
            # Pad sequences to the length of the longest sequence in the batch
            return {"waveform_24k_batch": torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True, padding_value=0.0)}
        except Exception as e:
            # This is still good practice to catch corrupted files etc.
            print(f"Skipping a problematic sample. Error: {e}")
            return None

# --- Checkpoint Management Functions ---
def save_checkpoint(step, loss, decoder, pitchmvmt, optimizer, scheduler, checkpoint_dir="checkpoints_training"):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Prune old checkpoints, keeping none.
    for old_ckpt in glob.glob(os.path.join(checkpoint_dir, "*.pth")):
        print(f"Removing old checkpoint: {old_ckpt}")
        os.remove(old_ckpt)

    save_path = os.path.join(checkpoint_dir, f"training_step_{step}.pth")
    
    torch.save({
        'step': step,
        'loss': loss,
        'decoder_state_dict': decoder.state_dict(),
        'pitchmvmt_state_dict': pitchmvmt.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, save_path)
    print(f"Saved checkpoint to {save_path} with loss {loss:.4f}")

def load_latest_checkpoint(checkpoint_dir, decoder, pitchmvmt, optimizer, scheduler):
    if not os.path.exists(checkpoint_dir):
        print("No checkpoint directory found. Starting from scratch.")
        return 0, None # Return None for last_loss

    checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    if not checkpoints:
        print("No checkpoints found in directory. Starting from scratch.")
        return 0, None # Return None for last_loss

    latest_ckpt_path = max(checkpoints, key=os.path.getctime)
    print(f"Loading latest checkpoint by last modified time: {latest_ckpt_path}")
    
    checkpoint = torch.load(latest_ckpt_path, map_location=torch.device('cpu'))
    
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    pitchmvmt.load_state_dict(checkpoint['pitchmvmt_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    decoder.to(DEVICE)
    pitchmvmt.to(DEVICE)
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(DEVICE)

    start_step = checkpoint.get('step', 0) + 1 # Use .get for backward compatibility
    last_loss = checkpoint.get('loss', None) # Use .get for backward compatibility
    
    # --- MODIFICATION: Print the loaded loss ---
    if last_loss is not None:
        print(f"Resuming training from step {start_step}. Last saved loss: {last_loss:.4f}")
    else:
        print(f"Resuming training from step {start_step}.")
        
    return start_step, last_loss

import questionary

def select_file_from_menu(folder_path: str, prefix: str) -> str | None:
    """
    Scans a folder for files with a specific prefix, displays an interactive
    menu, and returns the user's selection.

    Args:
        folder_path: The path to the folder to search.
        prefix: The file prefix to match (e.g., 'cfm_step_').

    Returns:
        The full path of the selected file, or None if no file was selected
        or no matching files were found.
    """
    try:
        # Find all files matching the prefix.
        all_files = os.listdir(folder_path)
        matching_files = [
            f for f in all_files 
            if f.startswith(prefix) and os.path.isfile(os.path.join(folder_path, f))
        ]

        if not matching_files:
            print(f"No checkpoints found in '{folder_path}' with prefix '{prefix}'! Make sure to have at least one checkpoint downloaded for each model.")
            return None

        # Sort to find the "greatest name" and make it the default. `questionary` automatically places the cursor on the first item in the list.
        # Holding Enter during selection will quickly choose the latest checkpoints available for every model.
        def key(name):
            try:
                return int(os.path.splitext(name)[0].replace(prefix, ""))
            except ValueError:
                return 0
        matching_files.sort(reverse=True, key=key)

        # Show the CLI UI and get the user's choice.
        selected_file_name = questionary.select(
            "Checkpoints:",
            choices=matching_files,
            use_indicator=True  # Adds a nice '>' indicator.
        ).ask() # .ask() returns the selection or None if the user cancels (e.g., Ctrl+C)

        if selected_file_name:
            return os.path.join(folder_path, selected_file_name)
        else:
            # User cancelled the selection.
            return None

    except FileNotFoundError:
        print(f"Error: The folder '{folder_path}' does not exist.")
        return None

def select_checkpoint_from_menu(model_name: str) -> str:
    print("Searching inside path './checkpoints' for available checkpoints...")
    ckpt_path = select_file_from_menu("./checkpoints", model_name + "_step_")
    if ckpt_path is None:
        print("No checkpoints found for model '" + model_name + "'!")
        exit()
    return ckpt_path


# --- Main Training Function ---
def main():
    global interrupted # Allow modification of the global flag
    
    # --- Configuration ---
    global DEVICE # Make DEVICE accessible to the checkpoint loader
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Batch size is determined by the length of COMPONENTS, when the data is being built in `transform_example`.
    # BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    MAX_STEPS = 1000000
    SAVE_EVERY_N_STEPS = 1000
    CHECKPOINT_DIR = "checkpoints_training"
    # --- Set the dropout probability ---
    F0_DROPOUT_PROB = 0.2 # 20% of the samples in each batch will be guaranteed to have their pitch-conditioning vectors (from `PitchMvmtEncoder`) eliminated.

    # --- Load Main Model for Training ---
    print("Pick starting checkpoints.")
    decoder_ckpt_path = select_checkpoint_from_menu("cfm")
    pitchmvmt_ckpt_path = select_checkpoint_from_menu("pitchmvmt")
    encoder_ckpt_path = select_checkpoint_from_menu("encoder")
    flow_ckpt_path = select_checkpoint_from_menu("flow")
    mel2wav_ckpt_path = select_checkpoint_from_menu("mel2wav")
    speaker_encoder_ckpt_path = select_checkpoint_from_menu("speaker_encoder")
    tokenizer_ckpt_path = select_checkpoint_from_menu("tokenizer")
    model = BeltOutTTM.from_local(decoder_ckpt_path,
                                     pitchmvmt_ckpt_path,
                                     encoder_ckpt_path,
                                     flow_ckpt_path,
                                     mel2wav_ckpt_path,
                                     speaker_encoder_ckpt_path,
                                     tokenizer_ckpt_path, device=DEVICE, eval=False)
    print(f"Model loaded on {DEVICE}.")

    # --- Optimizer and Scheduler ---
    params_to_train = list(model.decoder.parameters()) + list(model.pitchmvmt.parameters())
    optimizer = AdamW(params_to_train, lr=LEARNING_RATE)
    scheduler = ExponentialLR(optimizer, gamma=0.9999) # Slightly slower decay
    
    # --- Load Checkpoint if Available ---
    start_step, last_loss = load_latest_checkpoint(CHECKPOINT_DIR, model.decoder, model.pitchmvmt, optimizer, scheduler)

    # --- Setup Data Pipeline ---
    # data_processor = AudioDataProcessor(segment_len_s=4)
    # print("Loading dataset...")
    # dataset = load_dataset("Bill13579/combined-set-1", split='train', streaming=True)
    # # NO .filter() NEEDED ANYMORE!
    # shuffled_dataset = dataset.shuffle(buffer_size=32, seed=42)
    # transformed_dataset = shuffled_dataset.map(data_processor.transform_example)
    # # dataloader = DataLoader(transformed_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn_simple, num_workers=0)
    # dataloader = DataLoader(transformed_dataset, batch_size=None, num_workers=2, prefetch_factor=96)

    data_processor = AudioDataProcessor(segment_len_s=4)
    print("Loading dataset...")
    dataset = Arro3StreamingDataset(
        "Bill13579/combined-set-1",
        transform_fn=data_processor.transform_example,
        shuffle=True,
        shuffle_buffer_size=32,
        prefetch_files=2
    )
    dataloader = DataLoader(dataset, batch_size=None, num_workers=2, prefetch_factor=32)

    # --- Resamplers for the main loop (on GPU) ---
    resampler_16k = torchaudio.transforms.Resample(24000, 16000).to(DEVICE)
    crepe_sr = 16000

    # --- Training Loop ---
    print("Starting...")
    step = start_step
    postfix_dict = {"loss": f"{last_loss:.4f}" if last_loss is not None else "N/A", "lr": f"{scheduler.get_last_lr()[0]:.1e}"}
    training_progress = tqdm(total=MAX_STEPS, initial=start_step)
    training_progress.set_postfix(postfix_dict)
    
    data_iterator = iter(dataloader)

    epoch = 0

    while step < MAX_STEPS:
        if interrupted:
            break
        
        try:
            batch = next(data_iterator)
        except StopIteration:
            print(f"Epoch {epoch} completed. Restarting dataloader!")
            epoch += 1
            # Dataloader is exhausted, restart it for the next epoch
            data_iterator = iter(dataloader)
            batch = next(data_iterator)

        if batch is None: continue
        
        waveform_24k_batch = batch['waveform_24k_batch'].to(DEVICE)
        
        # --- DATA PREPARATION ---
        with torch.no_grad():
            gt_mel = model.mel_extractor(waveform_24k_batch)
            mel_len = gt_mel.shape[2]

            waveform_16k_batch = resampler_16k(waveform_24k_batch)
            s3_tokens, _ = model.tokenizer(waveform_16k_batch)
            
            x_vectors = torch.cat([
                model.embed_ref_x_vector(wf_24k.unsqueeze(0), 24000, device=DEVICE)
                for wf_24k in waveform_24k_batch
            ], dim=0)
            speaker_embedding = model.flow.spk_embed_affine_layer(x_vectors)

            # Calculate MEL and CREPE interactions.
            mel_frames_per_second = 50 # mel spectrogram settings: sampling_rate=24000, hop_size=480
            crepe_frames_per_second = 100 # 100 mel frames per second
            crepe_hop_length = int(crepe_sr / float(crepe_frames_per_second)) # 10ms hop
            n_crepe_frame_in_mel_frame = int(crepe_frames_per_second / mel_frames_per_second)

            # Pad the 16k waveform (doing it to a clone) to the exact required length if necessary.
            crepe_samples_needed = mel_len * n_crepe_frame_in_mel_frame * crepe_hop_length
            padded_waveform_16k = waveform_16k_batch
            pad_amount = crepe_samples_needed - padded_waveform_16k.shape[1]
            if pad_amount > 0:
                padded_waveform_16k = F.pad(torch.clone(padded_waveform_16k), (0, pad_amount))

            # Get CREPE embeddings by iterating over the batch.
            crepe_embeddings_list = []
            for i in range(padded_waveform_16k.shape[0]):
                # Get a single audio signal, but keep it 2D: (1, T)
                single_waveform = padded_waveform_16k[i:i+1, :]
                
                embedding = torchcrepe.embed(
                    single_waveform,
                    crepe_sr,
                    hop_length=crepe_hop_length,
                    model='tiny',
                    device=DEVICE,
                ) # Shape is guaranteed now to be (1, 1 + mel_len * 2 (or whatever n_crepe_frame_in_mel_frame it is), 32, crepe_embedding_size // 32)
                # We get rid of the extra embedding in the time dimension.
                crepe_embeddings_list.append(embedding[:, :mel_len*2, :, :])

            # Stack the list of (1, T, 32, D // 32) tensors into a single (B, T, 32, D // 32) tensor
            crepe_embedding = torch.cat(crepe_embeddings_list, dim=0)

            mel_lengths = torch.tensor([mel_len] * gt_mel.shape[0], device=DEVICE)
            mask = (torch.arange(mel_len, device=DEVICE).unsqueeze(0) < mel_lengths.unsqueeze(1)).unsqueeze(1)

            token_embeddings = model.flow.input_embedding(s3_tokens)
            token_len = torch.tensor([token_embeddings.shape[1]] * token_embeddings.shape[0], device=DEVICE)
            h, _ = model.encoder(token_embeddings, token_len)
            encoded_tokens = model.flow.encoder_proj(h)
            mu = encoded_tokens.transpose(1, 2)

            B = crepe_embedding.shape[0]
            projector_input = crepe_embedding.view(-1, n_crepe_frame_in_mel_frame, 256)
            pitch_mvmt_encode_flat = model.pitchmvmt(projector_input)
            pitch_mvmt_encode = pitch_mvmt_encode_flat.view(B, -1, 80).transpose(1, 2)
        # --- TRAINING STEP ---
        optimizer.zero_grad()

        # Get the current batch size
        current_batch_size = pitch_mvmt_encode.shape[0]

        ### --- THE RANDOM PERMUTATION DROPOUT LOGIC --- ###
        # 1. Calculate how many samples to drop out
        num_to_drop = int(current_batch_size * F0_DROPOUT_PROB)
        
        # 2. Create a base mask with the correct number of True (drop) and False (keep) values
        base_mask_false = torch.zeros(current_batch_size - num_to_drop, dtype=torch.bool)
        base_mask_true = torch.ones(num_to_drop, dtype=torch.bool)
        base_mask = torch.cat([base_mask_false, base_mask_true], dim=0)
        
        # 3. Shuffle the mask randomly. This is the key step.
        # We generate a random permutation of indices and use it to reorder the mask.
        shuffled_indices = torch.randperm(current_batch_size)
        dropout_mask = base_mask[shuffled_indices].to(DEVICE)
        
        # 4. Expand and apply the mask as before
        expanded_mask = dropout_mask.view(-1, 1, 1)
        final_pitch_mvmt_encode = pitch_mvmt_encode.masked_fill(expanded_mask, 0.0)
        ### ---------------------------------------------------- ###

        loss, _ = model.decoder.compute_loss(
            x1=gt_mel, mask=mask, mu=mu,
            spks=speaker_embedding, cond=final_pitch_mvmt_encode
        )
        loss.backward()
        optimizer.step()
        scheduler.step()

        # --- LOGGING AND SAVING ---
        current_loss = loss.item() # Get the current loss as a float
        postfix_dict["loss"] = f"{current_loss:.4f}"
        postfix_dict["lr"] = f"{scheduler.get_last_lr()[0]:.1e}"
        training_progress.set_postfix(postfix_dict)
        training_progress.update(1)

        if (step + 1) % SAVE_EVERY_N_STEPS == 0:
            save_checkpoint(step, current_loss, model.decoder, model.pitchmvmt, optimizer, scheduler, CHECKPOINT_DIR)

        step += 1
            
    training_progress.close()
    
    # Final save, whether finished or interrupted
    if start_step < step:
        print("Performing final save...")
        # Get the very last loss before saving
        final_loss = loss.item() if 'loss' in locals() else last_loss
        save_checkpoint(step - 1, final_loss, model.decoder, model.pitchmvmt, optimizer, scheduler, CHECKPOINT_DIR)

    print("Training finished.")


if __name__ == "__main__":
    main()

