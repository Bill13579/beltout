use anyhow::Result;
use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
use arrow::array::RecordBatchReader;
use indicatif::{ProgressBar, ProgressStyle};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use std::fs::{self, File};
use std::path::Path;

// --- Configuration ---
// Point this to the directory containing the large Parquet files.
const INPUT_DIR: &str = "./baked_dataset_split";
// The directory where the new, smaller files will be created. If it exists already, it will be removed first. Thus, do not point it to the same directory as INPUT_DIR, it will not do what you expect.
const OUTPUT_DIR: &str = "./final_baked_dataset";
// With a batch size of 32 samples, 62 such batches come out to about ~1GB, give or take some variance due to differing sample lengths.
// We round that up to 64.
// In the end, each of our split parquet files will contain this number of rows.
const CHUNK_SIZE: usize = 64;

fn main() -> Result<()> {
    println!("--- Starting Re-sharding Process ---");

    // 1. Setup input and output paths.
    let input_glob = format!("{}/**/*.parquet", INPUT_DIR);
    let output_path = Path::new(OUTPUT_DIR);
    if output_path.exists() {
        fs::remove_dir_all(output_path)?;
    }
    fs::create_dir_all(output_path)?;

    let input_files: Vec<_> = glob::glob(&input_glob)?
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .filter(|p| p.is_file())
        .collect();

    if input_files.is_empty() {
        anyhow::bail!("No Parquet files found in '{}'!", INPUT_DIR);
    }
    println!("Found {} input Parquet files to process.", input_files.len());

    // 2. Initialize variables for the streaming process.
    let mut schema: Option<SchemaRef> = None;
    let mut current_chunk_rows: Vec<RecordBatch> = Vec::new();
    let mut rows_in_chunk = 0;
    let mut output_file_index = 0;

    let pb = ProgressBar::new(input_files.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} Processing Files [{bar:40.cyan/blue}] {pos}/{len} ({eta})")?
            .progress_chars("#>-"),
    );

    // 3. The main streaming loop.
    for input_file_path in input_files {
        let file = File::open(&input_file_path)?;
        let mut reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;

        // On the first file, grab the schema.
        if schema.is_none() {
            schema = Some(reader.schema().clone());
        }

        // Iterate through the RecordBatches within this Parquet file.
        while let Some(batch_result) = reader.next() {
            let mut batch = batch_result?;

            // This loop handles cases where a single input batch is larger than our CHUNK_SIZE.
            while batch.num_rows() > 0 {
                let needed = CHUNK_SIZE - rows_in_chunk;
                
                if batch.num_rows() >= needed {
                    // We have enough rows to complete a chunk.
                    let slice_to_complete_chunk = batch.slice(0, needed);
                    current_chunk_rows.push(slice_to_complete_chunk);

                    // Write the completed chunk to a new file.
                    write_chunk(
                        &current_chunk_rows,
                        schema.as_ref().unwrap(),
                        output_path,
                        output_file_index,
                    )?;
                    output_file_index += 1;

                    // Reset for the next chunk.
                    current_chunk_rows.clear();
                    rows_in_chunk = 0;

                    // The remainder of the batch becomes the new batch to process.
                    batch = batch.slice(needed, batch.num_rows() - needed);
                } else {
                    // Not enough rows to complete a chunk, so just add what we have.
                    rows_in_chunk += batch.num_rows();
                    current_chunk_rows.push(batch);
                    break; // Move to the next batch from the reader.
                }
            }
        }
        pb.inc(1);
    }

    // 4. Write any remaining data in the last partial chunk.
    if !current_chunk_rows.is_empty() {
        println!("Writing final partial chunk...");
        write_chunk(
            &current_chunk_rows,
            schema.as_ref().unwrap(),
            output_path,
            output_file_index,
        )?;
        output_file_index += 1;
    }
    
    pb.finish_with_message("All files processed.");

    println!("\n✅ Re-sharding complete!");
    println!("  - Wrote {} smaller files to '{}'", output_file_index, OUTPUT_DIR);
    println!("  - You can now upload the contents of this directory to the Hub.");
    println!("  - Suggested renaming command for upload:");
    println!("    (cd {} && i=0; for f in *.parquet; do mv \"$f\" \"train-$(printf \"%05d\" $i)-of-$(printf \"%05d\" {}).parquet\"; i=$((i+1)); done)", OUTPUT_DIR, output_file_index);


    Ok(())
}

/// Helper function to write a collection of RecordBatches to a single Parquet file.
fn write_chunk(
    batches: &[RecordBatch],
    schema: &SchemaRef,
    output_dir: &Path,
    file_index: usize,
) -> Result<()> {
    if batches.is_empty() {
        return Ok(());
    }

    // Use a temporary name first to avoid issues with partial writes.
    let temp_path = output_dir.join(format!("chunk_{}.parquet.tmp", file_index));
    let final_path = output_dir.join(format!("{:05}.parquet", file_index));

    let file = File::create(&temp_path)?;
    let mut writer = ArrowWriter::try_new(file, schema.clone(), None)?;

    for batch in batches {
        writer.write(batch)?;
    }
    writer.close()?;

    // Rename the file to its final name upon successful completion.
    fs::rename(temp_path, final_path)?;

    Ok(())
}

