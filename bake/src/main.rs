//! The main baking script. See "config.rs" for more information on how to use.
//! 
//! A single run can use up to 860GB of RAM and at least 400 or so GB of storage space. 
//! It is recommended that you rent out a server for an hour to do this. The run is fairly quick, 
//! completing in less than 20 minutes on a 50 core server.

use std::{collections::HashMap, fs::File, io::Cursor, sync::Arc, time::Instant};

use anyhow::{bail, Context, Result};
use arrow::{array::{Array, ArrayRef, BinaryArray, LargeBinaryBuilder, LargeStringBuilder, RecordBatch, StructArray}, datatypes::{Field, Schema}};
use hound::{SampleFormat, WavSpec, WavWriter};
use indicatif::{MultiProgress, ParallelProgressIterator, ProgressBar, ProgressStyle};
use parquet::arrow::{arrow_reader::ParquetRecordBatchReaderBuilder, ArrowWriter};
use rand::{rng, seq::SliceRandom};
use rayon::{iter::{IntoParallelIterator, ParallelBridge, ParallelIterator}, ThreadPoolBuilder};
use rubato::{Resampler, SincFixedIn, SincInterpolationParameters};
use symphonia::core::{audio::SampleBuffer, codecs::{DecoderOptions, CODEC_TYPE_NULL}, formats::FormatOptions, io::MediaSourceStream, meta::MetadataOptions, probe::Hint};

use crate::config::{BAKE_MERGE_NUM_THREADS, BATCH_COMPOSITION, DATASET_PATHS, GROUP_CONFIGS, PARQUET_CHUNK_SIZE, TARGET_SR};

mod config;

// A type alias for our processed audio: an in-memory WAV file.
type WavPool = Vec<Vec<u8>>;

/// Resamples a single audio array to the target sample rate.
fn resample_audio_to_target(audio_array: Vec<f32>, original_sr: u32) -> Result<Vec<f32>> {
    if original_sr == TARGET_SR {
        return Ok(audio_array);
    }

    // Highest-quality settings.
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: rubato::SincInterpolationType::Cubic,
        oversampling_factor: 256,
        window: rubato::WindowFunction::BlackmanHarris2,
    };

    let mut resampler = SincFixedIn::<f32>::new(
        TARGET_SR as f64 / original_sr as f64,
        2.0,
        params,
        audio_array.len(),
        1, // Number of channels.
    )?;

    // Rubato expects a Vec<Vec<f32>>.
    let waves_in = vec![audio_array];
    let waves_out = resampler.process(&waves_in, None)?;

    Ok(waves_out.into_iter().next().unwrap()) // Ok since we always should have some output.
}

/// Decodes, resamples, and re-encodes an audio file from its raw bytes.
fn process_audio_bytes(encoded_bytes: &[u8]) -> Result<Vec<u8>> {
    let cursor = Cursor::new(encoded_bytes.to_vec());
    let mss = MediaSourceStream::new(Box::new(cursor), Default::default());

    let hint = Hint::new();
    let meta_opts: MetadataOptions = Default::default();
    let fmt_opts: FormatOptions = Default::default();
    let probed = symphonia::default::get_probe().format(&hint, mss, &fmt_opts, &meta_opts)?;

    let mut format = probed.format;
    let track = format.tracks().iter().find(|t| t.codec_params.codec != CODEC_TYPE_NULL).context("No supported audio track!")?;

    let original_sr = track.codec_params.sample_rate.context("Missing sample rate!")?;
    let decode_options = DecoderOptions { verify: true };
    let mut decoder = symphonia::default::get_codecs().make(&track.codec_params, &decode_options)?;

    let mut pcm_f32 = Vec::new();

    loop {
        match format.next_packet() {
            Ok(packet) => {
                let decoded = decoder.decode(&packet)?;
                let channel_count = decoded.spec().channels.count();
                let mut sample_buf = SampleBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec());
                sample_buf.copy_interleaved_ref(decoded);
                
                // Convert to mono if necessary and extend our buffer
                if channel_count > 1 {
                    for i in 0..sample_buf.len() / channel_count {
                        let frame = &sample_buf.samples()[i*channel_count..(i+1)*channel_count];
                        let mono_sample = frame.iter().sum::<f32>() / frame.len() as f32;
                        pcm_f32.push(mono_sample);
                    }
                } else {
                    pcm_f32.extend_from_slice(sample_buf.samples());
                }
            }
            Err(symphonia::core::errors::Error::IoError(_)) => break, // End of stream
            Err(e) => bail!("Error during decoding: {}", e),
        }
    }
    let resampled_pcm = resample_audio_to_target(pcm_f32, original_sr)?;
    encode_to_wav_bytes(&resampled_pcm)
}

/// Loads, decodes, resamples, and re-encodes all Parquet files for a given glob pattern.
fn load_and_process_component(path_glob: &str, m: &MultiProgress, name: &str) -> Result<WavPool> {
    let paths: Vec<_> = glob::glob(path_glob)?.collect::<Result<_, _>>()?;
    if paths.is_empty() {
        return Err(anyhow::anyhow!("No files found for glob: {}", path_glob));
    }

    let pb = m.add(ProgressBar::new(paths.len() as u64));
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} {prefix:15.bold.dim} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")?
        .progress_chars("#>-"));
    pb.set_prefix(name.to_string());

    // Process each Parquet file in parallel.
    let all_wav_data: Vec<WavPool> = paths
        .into_par_iter()
        .progress_with(pb)
        .map(|path| -> Result<WavPool> {
            let file = File::open(&path)?;
            let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
            let mut reader = builder.build()?;
            let mut processed_in_file = Vec::new();

            while let Some(record_batch) = reader.next() {
                let batch = record_batch?;

                // 1. Get the top-level column "audio", which is a StructArray.
                let audio_struct: &StructArray = batch
                    .column_by_name("audio")
                    .context("Missing top-level 'audio' struct column!")?
                    .as_any()
                    .downcast_ref()
                    .context("Top-level 'audio' column is not a StructArray!")?;

                // 2. Get the "bytes" field from *within* the struct, which is a BinaryArray.
                let audio_bytes_col: &BinaryArray = audio_struct
                    .column_by_name("bytes")
                    .context("Struct 'audio' is missing 'bytes' field!")?
                    .as_any()
                    .downcast_ref()
                    .context("'bytes' field is not a BinaryArray!")?;

                for i in 0..batch.num_rows() {
                    let audio_bytes = audio_bytes_col.value(i);
                    if let Ok(wav) = process_audio_bytes(audio_bytes) {
                        processed_in_file.push(wav);
                    }
                }
            }
            Ok(processed_in_file)
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(all_wav_data.into_iter().flatten().collect())
}

/// Calculates repetition factors based on dataset sizes and probabilities.
fn calculate_repetition_factors(sizes: &[usize], probs: &[f64]) -> Vec<f64> {
    let densities: Vec<f64> = sizes.iter().zip(probs).map(|(&s, &p)| if s > 0 && p > 0.0 { p / s as f64 } else { 0.0 }).collect();
    let min_density = densities.iter().filter(|&&d| d > 0.0).fold(f64::INFINITY, |a, &b| a.min(b));
    if min_density.is_infinite() { return vec![1.0; sizes.len()]; }
    densities.iter().map(|&d| if d > 0.0 { d / min_density } else { 0.0 }).collect()
}

/// Encodes a raw f32 PCM vector into a full WAV file in an in-memory byte buffer.
fn encode_to_wav_bytes(samples: &[f32]) -> Result<Vec<u8>> {
    let spec = WavSpec {
        channels: 1,
        sample_rate: TARGET_SR,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };

    let mut buffer = Cursor::new(Vec::new());
    let mut writer = WavWriter::new(&mut buffer, spec)?;
    for &sample in samples {
        writer.write_sample(sample)?;
    }
    writer.finalize()?;
    Ok(buffer.into_inner())
}

fn main() -> Result<()> {
    let start_time = Instant::now();
    let m = MultiProgress::new();

    // --- Step 1: Process each group to create a resampled, upsampled pool of WAV bytes ---
    let main_pb = m.add(ProgressBar::new_spinner());
    main_pb.set_style(ProgressStyle::default_spinner().template("{spinner:.blue} {msg}")?);
    main_pb.set_message("Step 1: Processing all dataset groups...");

    let processed_groups: HashMap<String, Arc<WavPool>> = GROUP_CONFIGS
        .entries()
        .par_bridge() // Process groups in parallel
        .map(|(&group_name, group_config)| -> Result<(String, Arc<WavPool>)> {
            println!("-> Starting group: {}", group_name);

            // Load components and get their sizes.
            let components: Vec<(String, WavPool)> = group_config
                .components
                .iter()
                .map(|&key| {
                    let path = DATASET_PATHS.get(key).unwrap();
                    println!("  -> Loading & Resampling component: {} for group {}...", key, group_name);
                    let wav_pool = load_and_process_component(path, &m, key)?;
                    Ok((key.to_string(), wav_pool))
                })
                .collect::<Result<Vec<_>>>()?;

            let sizes: Vec<usize> = components.iter().map(|(_, pool)| pool.len()).collect();
            let factors = calculate_repetition_factors(&sizes, group_config.probabilities);

            let mut final_group_pool: WavPool = Vec::new();
            for ((key, pool), factor) in components.into_iter().zip(factors) {
                if factor == 0.0 || pool.is_empty() { continue; }

                let num_full_repeats = factor.floor() as usize;
                let num_extra_samples = (factor.fract() * pool.len() as f64).round() as usize;

                for _ in 0..num_full_repeats {
                    final_group_pool.extend_from_slice(&pool);
                }
                if num_extra_samples > 0 {
                    final_group_pool.extend_from_slice(&pool[..num_extra_samples]);
                }
                println!("    - Upsampled {}: factor {:.2}, added {} samples", key, factor, num_full_repeats * pool.len() + num_extra_samples);
            }

            // Shuffle the final pool for this group.
            let mut rng = rng();
            final_group_pool.shuffle(&mut rng);

            println!("<- Finished group: {}. Total samples: {}", group_name, final_group_pool.len());
            Ok((group_name.to_string(), Arc::new(final_group_pool)))
        })
        .collect::<Result<HashMap<_, _>>>()?;
    
    main_pb.finish_with_message("Step 1: All groups processed.");

    // --- Step 2: Assemble the final "baked" dataset in parallel ---
    let main_pb = m.add(ProgressBar::new_spinner());
    main_pb.set_style(ProgressStyle::default_spinner().template("{spinner:.blue} {msg}")?);
    main_pb.set_message("Step 2: Assembling final delicious baked batches...");

    let (largest_group_name, num_batches) = BATCH_COMPOSITION
        .entries()
        .filter_map(|(&name, &count)| {
            processed_groups.get(name).map(|pool| (name, pool.len() / count + 1))
        })
        .max_by_key(|&(_, num_batches)| num_batches)
        .context("could not determine largest group or no data was processed.")?;
    println!("Largest group is '{}', generating {} batches.", largest_group_name, num_batches);

    let total_columns_to_build = BATCH_COMPOSITION.values().sum::<usize>();
    let pb = m.add(ProgressBar::new(total_columns_to_build as u64));
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} {prefix:20.bold.dim} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")?
        .progress_chars("#>-"));
    pb.set_prefix("Building final columns");

    let mut columns_to_build = Vec::new();
    for (group_idx, (&group_name, &count)) in BATCH_COMPOSITION.entries().enumerate() {
        for i in 0..count {
            columns_to_build.push((group_idx, i, group_name, count));
        }
    }

    // The next part is memory intensive, adjust in order to prevent OOM.
    let pool = ThreadPoolBuilder::new()
        .num_threads(BAKE_MERGE_NUM_THREADS)
        .build()?;

    let mut final_ordered_columns: Vec<((usize, usize), (Field, ArrayRef))> = pool.install(|| {
        columns_to_build
            .into_par_iter()
            .progress_with(pb)
            .map(|(group_idx, i, group_name, count)| -> Result<((usize, usize), (Field, ArrayRef))> {
                // This logic is now executed by a maximum of BAKE_MERGE_NUM_THREADS threads at a time.
                let group_pool = processed_groups.get(group_name).unwrap();
                let samples_to_take = count * num_batches;

                // Cycle through the pool to get enough samples.
                let mut collected_samples: WavPool = group_pool.iter().cloned().cycle().take(samples_to_take).collect();
                let mut rng = rng();
                collected_samples.shuffle(&mut rng);

                let col_name = format!("{}-{}", group_name.to_lowercase(), i + 1);
                let start = i * num_batches;
                let end = start + num_batches;
                let column_data: &[Vec<u8>] = &collected_samples[start..end];

                // Build the `bytes` and `path` arrays.
                let mut bytes_builder = LargeBinaryBuilder::new();
                let mut path_builder = LargeStringBuilder::new();

                for (idx, wav_bytes) in column_data.iter().enumerate() {
                    bytes_builder.append_value(wav_bytes);
                    // The path is often a placeholder, but let's make it unique.
                    path_builder.append_value(format!("{}-{}-{}.wav", group_name, i, idx));
                }
                let bytes_array = bytes_builder.finish();
                let path_array = path_builder.finish();

                // Create the final StructArray for this column.
                let struct_array = StructArray::from(vec![
                    (Arc::new(Field::new("bytes", bytes_array.data_type().clone(), true)), Arc::new(bytes_array) as ArrayRef),
                    (Arc::new(Field::new("path", path_array.data_type().clone(), true)), Arc::new(path_array) as ArrayRef),
                ]);
                let field = Field::new(&col_name, struct_array.data_type().clone(), false);
                let column = Arc::new(struct_array) as ArrayRef;

                Ok(((group_idx, i), (field, column)))
            })
            .collect::<Result<Vec<_>>>()
    })?;
    // Sort the results to maintain the original order of the columns.
    final_ordered_columns.sort_by_key(|(k, _)| *k);

    let (final_fields, final_columns): (Vec<_>, Vec<_>) = final_ordered_columns.into_iter().map(|(_, v)| v).unzip();

    let schema = Arc::new(Schema::new(final_fields));
    let final_batch = RecordBatch::try_new(schema.clone(), final_columns)?;
    
    main_pb.finish_with_message("Step 2: Final assembly complete.");

    // --- Step 3: Write the final dataset to multiple Parquet files ---
    let main_pb = m.add(ProgressBar::new_spinner());
    main_pb.set_style(ProgressStyle::default_spinner().template("{spinner:.blue} {msg}")?);
    main_pb.set_message("Step 3: Writing final Parquet files...");

    let total_rows = final_batch.num_rows();
    let chunk_size = PARQUET_CHUNK_SIZE;
    let num_chunks = (total_rows as f64 / chunk_size as f64).ceil() as usize;

    println!("Total rows: {}, Chunk size: {}, Number of files: {}", total_rows, chunk_size, num_chunks);

    let pb = m.add(ProgressBar::new(num_chunks as u64));
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} {prefix:20.bold.dim} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")?
        .progress_chars("#>-"));
    pb.set_prefix("Writing Parquet chunks");

    // Create an output directory for the split files.
    let output_dir = "baked_dataset_split";
    std::fs::create_dir_all(output_dir)?;

    // We can use the global thread pool here, as writing is I/O bound and less memory-intensive per-task.
    (0..num_chunks).into_par_iter().progress_with(pb).try_for_each(|i| -> Result<()> {
        let offset = i * chunk_size;
        let length = std::cmp::min(chunk_size, total_rows - offset);

        // Take a zero-copy slice of the full RecordBatch.
        let chunk_batch = final_batch.slice(offset, length);

        // e.g., baked_dataset_split/00000.parquet
        let output_path = format!("{}/train-{:05}-of-{:05}.parquet", output_dir, i, num_chunks);
        let file = File::create(output_path)?;
        
        // Each thread creates its own writer.
        let mut writer = ArrowWriter::try_new(file, schema.clone(), None)?;
        writer.write(&chunk_batch)?;
        writer.close()?;
        Ok(())
    })?;

    main_pb.finish_with_message("Step 3: All Parquet chunks written.");
    m.clear()?;

    println!("\n✅ Bake complete! Output written to directory '{}'", output_dir);
    println!("Total time elapsed: {:?}", start_time.elapsed());
    println!("Final dataset shape: ({}, {}) across {} files", final_batch.num_rows(), final_batch.num_columns(), num_chunks);

    Ok(())
}

