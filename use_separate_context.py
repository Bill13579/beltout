# A version of the run script which allows for extracting prosodic/phonetic context and pitch context from two separate audio files of the same length (Experimental).

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
import librosa
import time
import os
from beltout import BeltOutTTM
import torchaudio
import torchcrepe
import soundfile

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

# --- Model and Checkpoint Loading ---
print("Loading model...")
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
if torch.backends.mps.is_available():
    device = 'mps'

# Load the pre-trained model
# try:
#     model = BeltOutTTM.from_pretrained_hf(local_dir="./checkpoints", device=device)
#     print(f"Model loaded on {device}.")
# except Exception as e:
#     print(f"Could not load pretrained model from HF: {e}.")

try:
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
                                  tokenizer_ckpt_path, device=device)
    # Set model to evaluation mode
    model.eval()
    print(f"Model loaded from local './checkpoints' directory on {device}.")
except Exception as e_local:
    print(f"Could not load model from local directory: {e_local}")
    exit()

# --- Utility Functions ---
def get_x_vector_from_wav_chunk(wav_chunk):
    """Helper to get a single x-vector from a raw audio chunk."""
    ref_tensor = torch.from_numpy(wav_chunk).float().to(device).unsqueeze(0)
    with torch.inference_mode():
        return model.embed_ref_x_vector(ref_tensor, model.sr, device=device).detach().cpu().numpy().flatten()

def smart_split(wav, wav2, sr, min_chunk_duration_s=40.0, top_db=35):
    split_indices = librosa.effects.split(wav, top_db=top_db, frame_length=1024, hop_length=256)
    if len(split_indices) < 1: return [wav]
    min_chunk_samples = int(min_chunk_duration_s * sr)
    merged_chunks = []
    merged_chunks_2 = []
    current_chunk_start = 0
    for i in range(len(split_indices)):
        next_chunk_start = split_indices[i+1][0] if i + 1 < len(split_indices) else len(wav)
        if (next_chunk_start - current_chunk_start) >= min_chunk_samples and current_chunk_start != next_chunk_start:
            merged_chunks.append(wav[current_chunk_start:next_chunk_start])
            merged_chunks_2.append(wav2[current_chunk_start:next_chunk_start])
            current_chunk_start = next_chunk_start
    if current_chunk_start < len(wav):
        last_piece = wav[current_chunk_start:]
        last_piece_2 = wav2[current_chunk_start:]
        if merged_chunks and len(last_piece) < min_chunk_samples / 2:
            merged_chunks[-1] = np.concatenate([merged_chunks[-1], last_piece])
            merged_chunks_2[-1] = np.concatenate([merged_chunks_2[-1], last_piece_2])
        elif len(last_piece) > 0:
            merged_chunks.append(last_piece)
            merged_chunks_2.append(last_piece_2)
    return merged_chunks, merged_chunks_2

def get_vector_stats(vector):
    if vector is None: return "Vector Stats: N/A"
    return f"Vector Norm: {np.linalg.norm(vector):.4f} | Mean: {np.mean(vector):.4f}"

# --- Gradio Backend Logic ---
def update_chunk_slider(audio_file):
    """
    When a user uploads an audio file to a blender slot, this function
    updates the corresponding chunk size slider to match the audio's duration.
    """
    if audio_file is None:
        # If audio is cleared, reset and hide the slider
        return gr.update(value=0, maximum=120, visible=False)
    
    try:
        # gr.Audio(type="filepath") provides the path to a temporary file
        duration = librosa.get_duration(path=audio_file)
        # Make the slider visible and set its max value to the audio duration
        return gr.update(maximum=int(duration), visible=True)
    except Exception as e:
        print(f"Could not get audio duration: {e}")
        # If there's an error, just return a default state
        return gr.update(value=0, maximum=120, visible=False)

def set_source_audio_s3(state, source_audio):
    if source_audio is None: return state, None
    state["source_audio_path_s3"] = source_audio
    state["is_source_set_s3"] = True
    print("Source audio loaded for prosodic and phonetic context.")
    return state, (model.sr, np.zeros(1))

def set_source_audio_pitchmvmt(state, source_audio):
    if source_audio is None: return state, None
    state["source_audio_path_pitchmvmt"] = source_audio
    state["is_source_set_pitchmvmt"] = True
    print("Source audio loaded for pitch context.")
    return state, (model.sr, np.zeros(1))

# --- Main Inference Function ---
def run_conversion(state, mode, progress=gr.Progress(track_tqdm=True)):
    if (not state.get("is_source_set_s3")) or (not state.get("is_source_set_pitchmvmt")) or ("source_audio_path_s3" not in state) or ("source_audio_path_pitchmvmt" not in state):
        gr.Warning("Please upload source audio files first.")
        return (model.sr, np.zeros(1)), None
    
    source_path_s3 = state["source_audio_path_s3"]
    source_path_pitchmvmt = state["source_audio_path_pitchmvmt"]
    active_x_vector = state.get("current_x_vector")
    if active_x_vector is None:
        gr.Warning("No active x-vector. Please set or create one before running the conversion.")
        return (model.sr, np.zeros(1)), None

    yield None, None
    
    # --- HELPER FUNCTION FOR A SINGLE CHUNK ---
    # This avoids code duplication between HQ and Streaming modes
    def process_chunk(wav_chunk_s3, wav_chunk_pitchmvmt):
        with torch.inference_mode():
            # --- STEP 1: PREPARE ALL RAW INPUTS FIRST ---
            waveform_24k_tensor_s3 = torch.from_numpy(wav_chunk_s3).float().to(device).unsqueeze(0)
            waveform_16k_tensor_s3 = torchaudio.transforms.Resample(model.sr, 16000).to(device)(waveform_24k_tensor_s3)
            waveform_24k_tensor_pitchmvmt = torch.from_numpy(wav_chunk_pitchmvmt).float().to(device).unsqueeze(0)
            waveform_16k_tensor_pitchmvmt = torchaudio.transforms.Resample(model.sr, 16000).to(device)(waveform_24k_tensor_pitchmvmt)
            
            # Get S3 tokens and speaker embedding
            s3_tokens, _ = model.tokenizer(waveform_16k_tensor_s3)
            x_vector_tensor = torch.from_numpy(active_x_vector).float().to(device).unsqueeze(0)
            speaker_embedding = model.flow.spk_embed_affine_layer(x_vector_tensor)
            
            # --- STEP 3: PREPARE CONDITIONING SIGNALS TO MATCH THE TARGET MEL LENGTH ---
            
            # 3a. Prepare token embeddings ('mu')
            token_embeddings = model.flow.input_embedding(s3_tokens)
            token_len = torch.tensor([token_embeddings.shape[1]], device=device)
            h, _ = model.encoder(token_embeddings, token_len)
            encoded_tokens = model.flow.encoder_proj(h)
            
            mu = encoded_tokens.transpose(1, 2)
            mel_len = mu.shape[2]

            # 3b. Prepare pitch embeddings ('pitchmvmt')
            pitch_mvmt_encode = None
            crepe_sr = 16000
            crepe_frames_per_second = 100 # 100 mel frames per second
            crepe_hop_length = int(crepe_sr / float(crepe_frames_per_second)) # 10ms hop
            n_crepe_frame_in_mel_frame = 2
            
            crepe_samples_needed = mel_len * n_crepe_frame_in_mel_frame * crepe_hop_length
            padded_waveform_16k = waveform_16k_tensor_pitchmvmt
            pad_amount = crepe_samples_needed - padded_waveform_16k.shape[1]
            if pad_amount > 0:
                padded_waveform_16k = F.pad(torch.clone(padded_waveform_16k), (0, pad_amount))

            crepe_embedding = torchcrepe.embed(
                padded_waveform_16k,
                crepe_sr,
                hop_length=crepe_hop_length,
                model='tiny',
                device=device,
            )

            crepe_embedding = crepe_embedding[:, :mel_len*2, :, :]

            projector_input = crepe_embedding.view(-1, n_crepe_frame_in_mel_frame, 256)
            pitch_mvmt_encode_flat = model.pitchmvmt(projector_input)
            pitch_mvmt_encode = pitch_mvmt_encode_flat.view(1, -1, 80).transpose(1, 2)

            # --- STEP 4: GENERATE THE MEL-SPECTROGRAM ---
            mask = torch.ones(1, 1, mu.shape[2], device=device, dtype=torch.bool)
            output_mels, _ = model.decoder(
                mu=mu, mask=mask, spks=speaker_embedding, cond=pitch_mvmt_encode, n_timesteps=10
            )
            
            # --- STEP 5: VOCODE ---
            output_wav_tensor, _ = model.mel2wav.inference(speech_feat=output_mels)
            return output_wav_tensor.squeeze(0).cpu().numpy()

    # --- MODE SWITCH LOGIC ---
    if mode == "⭐ High Quality (Single Pass)":
        progress(0, desc="Starting high-quality conversion...")
        source_wav_s3, sr = librosa.load(source_path_s3, sr=model.sr, mono=True)
        source_wav_pitchmvmt, sr = librosa.load(source_path_pitchmvmt, sr=model.sr, mono=True)
        output_wav_np = process_chunk(source_wav_s3, source_wav_pitchmvmt)
        progress(1, desc="Conversion complete!")

        filename = f"audio_{int(time.time())}.wav"
        soundfile.write(filename, output_wav_np, model.sr)

        yield (model.sr, output_wav_np), gr.File(value=filename, label="Saved Audio File")
    else: # "⚡ Fast Preview (Streaming)"
        wav_s3, sr = librosa.load(source_path_s3, sr=None, mono=True)
        if sr != model.sr:
            wav_s3 = librosa.resample(wav_s3, orig_sr=sr, target_sr=model.sr)
        wav_pitchmvmt, sr = librosa.load(source_path_pitchmvmt, sr=None, mono=True)
        if sr != model.sr:
            wav_pitchmvmt = librosa.resample(wav_pitchmvmt, orig_sr=sr, target_sr=model.sr)
        
        source_chunks_s3, source_chunks_pitchmvmt = smart_split(wav_s3, wav_pitchmvmt, sr=model.sr)

        full_np = np.zeros((0,))
        
        for i, (chunk_wav_s3, chunk_wav_pitchmvmt) in enumerate(zip(source_chunks_s3, source_chunks_pitchmvmt)):
            print(f"Streaming chunk {i+1}/{len(source_chunks_s3)}...")
            output_chunk_np = process_chunk(chunk_wav_s3, chunk_wav_pitchmvmt)
            full_np = np.concatenate([full_np, output_chunk_np], axis=0)
            if i+1 == len(source_chunks_s3):
                filename = f"audio_{int(time.time())}.wav"
                soundfile.write(filename, full_np, model.sr)

                yield (model.sr, output_chunk_np), gr.File(value=filename, label="Saved Audio File")
            else:
                yield (model.sr, output_chunk_np), None

def synth_style_blender(state, *all_inputs, progress=gr.Progress()):
    audio_tasks, npy_tasks = [], []
    for i in range(0, 8 * 3, 3):
        audio, weight, chunk_size = all_inputs[i:i+3]
        if audio is not None:
            audio_tasks.append({'audio_path': audio, 'weight': weight, 'chunk_size_s': chunk_size, 'label': f"Voice {chr(65 + i//3)}"})
    npy_start_index = 8 * 3
    for i in range(0, 8 * 2, 2):
        npy_file, weight = all_inputs[npy_start_index + i : npy_start_index + i + 2]
        if npy_file is not None:
            npy_tasks.append({'npy_path': npy_file.name, 'weight': weight, 'label': f"Vector {i//2 + 1}"})
    if not audio_tasks and not npy_tasks:
        gr.Warning("Please upload at least one voice/vector.")
        active_vector = state.get("current_x_vector")
        return state, get_vector_stats(active_vector)
    
    # --- Phase 1 & 2: Combined Processing with Progress ---
    all_vectors, all_weights = [], []
    
    for task in audio_tasks:
        wav, sr = librosa.load(task['audio_path'], sr=None, mono=True)
        if sr != model.sr: wav = librosa.resample(wav, orig_sr=sr, target_sr=model.sr)
        
        chunk_size_s = task['chunk_size_s']
        partial_vectors = []
        
        if chunk_size_s > 0:
            chunk_samples = int(chunk_size_s * model.sr)
            if len(wav) < chunk_samples:
                progress(0, desc=f"Processing {task['label']} (clip shorter than chunk size)")
                partial_vectors.append(get_x_vector_from_wav_chunk(wav))
            else:
                num_chunks = (len(wav) - chunk_samples) // chunk_samples + 1
                if len(wav) % chunk_samples != 0: num_chunks += 1
                
                for i, start_idx in enumerate(range(0, len(wav) - chunk_samples + 1, chunk_samples)):
                    progress(i / num_chunks, desc=f"Processing {task['label']}, Chunk {i+1}/{num_chunks}")
                    chunk = wav[start_idx:start_idx+chunk_samples]
                    partial_vectors.append(get_x_vector_from_wav_chunk(chunk))
                if len(wav) % chunk_samples != 0:
                    progress((num_chunks-1) / num_chunks, desc=f"Processing {task['label']}, Chunk {num_chunks}/{num_chunks} (final)")
                    last_chunk = wav[-chunk_samples:]
                    partial_vectors.append(get_x_vector_from_wav_chunk(last_chunk))
        else:
            progress(0, desc=f"Processing {task['label']} (full clip)")
            partial_vectors.append(get_x_vector_from_wav_chunk(wav))
            
        if partial_vectors:
            avg_vector = np.mean(partial_vectors, axis=0)
            all_vectors.append(avg_vector)
            all_weights.append(task['weight'])

    for task in npy_tasks:
        progress(0, desc=f"Loading {task['label']}...")
        try:
            loaded_vector = np.load(task['npy_path'])
            if loaded_vector.shape == (192,):
                all_vectors.append(loaded_vector); all_weights.append(task['weight'])
            else: gr.Warning(f"Skipping {task['label']}: invalid shape {loaded_vector.shape}")
        except Exception as e: gr.Warning(f"Skipping {task['label']}: could not load file. Error: {e}")

    # --- Phase 3: Final Blending ---
    if not all_vectors:
        gr.Warning("Failed to process any voices/vectors.");
        active_vector = state.get("current_x_vector")
        return state, get_vector_stats(active_vector)

    progress(0.99, desc="Blending final vectors...")
    all_vectors, all_weights = np.array(all_vectors), np.array(all_weights).reshape(-1, 1)
    blended_vec = np.sum(all_vectors * all_weights, axis=0)
    
    state["current_x_vector"] = blended_vec
    
    progress(1.0, desc="Blend complete!")
    gr.Info("Blended successfully!")
    return state, get_vector_stats(blended_vec)

def randomize_vector(state, strength):
    random_vector = np.random.randn(192).astype(np.float32)
    final_vector = (random_vector / np.linalg.norm(random_vector)) * strength
    state["current_x_vector"] = final_vector
    gr.Info("Random vector generated! Check the stats for information on it.")
    return state, get_vector_stats(final_vector)

def reset_vector(state):
    state["current_x_vector"] = None
    gr.Info("Cleared.")
    return state, get_vector_stats(state.get("current_x_vector"))

def save_vector(state):
    active_vector = state.get("current_x_vector")
    if active_vector is None:
        gr.Warning("No active x-vector to save."); return None
    filename = f"vec_{int(time.time())}.npy"
    np.save(filename, active_vector)
    gr.Info(f"Saved as {filename}.")
    return gr.File(value=filename, label="Saved `.npy` File")

def load_vector(state, vector_file):
    if vector_file is None:
        gr.Warning("Please upload a vector file.")
        return state, get_vector_stats(state.get("current_x_vector"))
    try:
        loaded_vector = np.load(vector_file.name)
    except Exception as e:
        gr.Warning(f"Failed to load vector file: {e}"); return state, get_vector_stats(state.get("current_x_vector"))
    if loaded_vector.shape != (192,):
        gr.Warning(f"Invalid vector file. Shape must be (192,), got {loaded_vector.shape}"); return state, get_vector_stats(state.get("current_x_vector"))
    active_vector = loaded_vector
    state["current_x_vector"] = loaded_vector
    gr.Info("Vector loaded!")
    return state, get_vector_stats(active_vector)

# --- Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    state = gr.State({})
    gr.Markdown("# BeltOut Timbre Workshop!!")
    with gr.Tabs():
        with gr.TabItem("Vectors"):
            gr.Markdown("""
                        The model represents timbre as a list of 192 numbers, which is called the **x-vector** by the originating literature. Taking this in along with your voice recording, the model produces a new audio file with the timbre applied.

                        You can:
                        - Load these numbers directly from pre-saved **npy (numpy)** files,
                        - Or calculate an average timbre vector from example audio files,
                        - Or mix multiple vectors into a new one,
                        - Or generate one randomly.
                        """)
            vector_stats_display = gr.Textbox(label="Active Vector Stats", interactive=False)
            with gr.Tabs():
                with gr.TabItem("🧬 Synth-Style Blender"):
                    gr.Markdown("""
                                Blend up to 8 voices and 8 pre-saved vector files. Only slots with uploaded files will be used.

                                Weights are not normalized. If a weight is set to 2, then the vector will be mixed in with twice the magnitude as expected.

                                When a sample audio file is uploaded, a new field called "Chunk Size (s)" will appear. While the vector statistics model can theoretically handle any length, the maximum amount of audio it can look at at once is still limited by how much VRAM you have. For such a scenario, you can set this chunk size to something less than the full thing, which will split the audio file into chunks of that size, process those chunks one-by-one, and then merge the vectors back at the end.
                                """)
                    all_blend_inputs = []
                    with gr.Row():
                        for i in range(2):
                            with gr.Column():
                                audio = gr.Audio(type="filepath", label=f"Voice {chr(65+i)}")
                                weight = gr.Slider(-5, 5, value=1.0, label=f"Weight {chr(65+i)}")
                                chunk_size = gr.Slider(0, 120, value=0, label="Chunk Size (s)", step=1, visible=False)
                                audio.upload(fn=update_chunk_slider, inputs=[audio], outputs=[chunk_size])
                                audio.clear(fn=update_chunk_slider, inputs=[audio], outputs=[chunk_size])
                                all_blend_inputs.extend([audio, weight, chunk_size])
                    with gr.Accordion("➕ More Voices", open=False):
                        gr.Markdown("### Audio Voices (C-H)")
                        for i in range(2, 8, 2):
                             with gr.Row():
                                for j in range(2):
                                    with gr.Column():
                                        audio = gr.Audio(type="filepath", label=f"Voice {chr(65+i+j)}")
                                        weight = gr.Slider(-5, 5, value=0.0, label=f"Weight {chr(65+i+j)}")
                                        chunk_size = gr.Slider(0, 120, value=0, label=f"Chunk Size (s)", step=1, visible=False)
                                        audio.upload(fn=update_chunk_slider, inputs=[audio], outputs=[chunk_size])
                                        audio.clear(fn=update_chunk_slider, inputs=[audio], outputs=[chunk_size])
                                        all_blend_inputs.extend([audio, weight, chunk_size])
                    with gr.Accordion("➕ Vectors", open=False):
                        gr.Markdown("### Saved Vector Files (.npy)")
                        for i in range(0, 8, 2):
                            with gr.Row():
                                for j in range(2):
                                    with gr.Column():
                                        npy = gr.File(label=f"Vector {i+j+1}", file_types=[".npy"])
                                        weight = gr.Slider(-5, 5, value=0.0, label=f"Weight {i+j+1}")
                                        all_blend_inputs.extend([npy, weight])
                    blend_button = gr.Button("Blend Voices")
                with gr.TabItem("🔀 Voice Randomizer"):
                    gr.Markdown("Generate a new, random timbre vector. 'Strength' controls the magnitude (norm) of the random x-vector.")
                    random_strength_input = gr.Number(value=12.0, label="Randomization Strength (Vector Norm)", minimum=0.01)
                    randomize_button = gr.Button("✨ Generate Random Voice")
                with gr.TabItem("💾 Save / Load Voice"):
                    gr.Markdown("Save the current active x-vector as a `.npy` file or load a previously saved one.")
                    with gr.Row():
                        save_button = gr.Button("Save Active Vector")
                        load_vector_file = gr.File(label="Load Vector File (.npy)")
                    saved_file_output = gr.File(label="Saved Vector File", interactive=False)
        with gr.TabItem("Main Conversion"):
            gr.Markdown("#### THE TWO AUDIO FILES MUST BE OF THE EXACT SAME LENGTH IN SAMPLES. THIS IS THE ADVANCED VERSION OF THE APP, USE WITH CARE.")
            source_audio_input_s3 = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Upload or Record Source Audio for Prosodic and Phonetic Context")
            source_audio_input_pitchmvmt = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Upload or Record Source Audio for Pitch Context")
            with gr.Accordion("⚙️ Generation Settings", open=False):
                mode_switch = gr.Radio(["⭐ High Quality (Single Pass)", "⚡ Fast Preview (Streaming)"], value="⭐ High Quality (Single Pass)", label="Conversion Mode")
            start_button = gr.Button("Run", variant="primary")
            gr.Markdown("### Output")
            output_audio = gr.Audio(label="Conversion Result", streaming=True, autoplay=False) # streaming=True works with generators
            saved_audio_file_output = gr.File(label="Saved Audio File", interactive=False)

    # --- Event Handlers ---
    source_audio_input_s3.upload(fn=set_source_audio_s3, inputs=[state, source_audio_input_s3], outputs=[state, output_audio])
    source_audio_input_pitchmvmt.upload(fn=set_source_audio_pitchmvmt, inputs=[state, source_audio_input_pitchmvmt], outputs=[state, output_audio])
    source_audio_input_s3.stop_recording(fn=set_source_audio_s3, inputs=[state, source_audio_input_s3], outputs=[state, output_audio])
    source_audio_input_pitchmvmt.stop_recording(fn=set_source_audio_pitchmvmt, inputs=[state, source_audio_input_pitchmvmt], outputs=[state, output_audio])
    start_button.click(fn=run_conversion, inputs=[state, mode_switch], outputs=[output_audio, saved_audio_file_output])
    randomize_button.click(fn=randomize_vector, inputs=[state, random_strength_input], outputs=[state, vector_stats_display])
    blend_button.click(fn=synth_style_blender, inputs=[state, *all_blend_inputs], outputs=[state, vector_stats_display])
    save_button.click(fn=save_vector, inputs=[state], outputs=[saved_file_output])
    load_vector_file.upload(fn=load_vector, inputs=[state, load_vector_file], outputs=[state, vector_stats_display])

demo.queue().launch(debug=True, share=False)