# bake

`bake` is the tool use to convert many parquet datasets with distinct general speech characteristics into a single, pre-batched, fixed composition dataset for actual training.

To use, modify `src/config.rs` to fit your own training dataset design. See the comment at the top of `src/main.rs` and `src/config.rs` for more information.

Pre-batching datasets allow for training on lower-spec machines (but with a decent graphics card) that would otherwise suffer greatly from having to shuffle, randomize, do DSP, and more with traditional training approaches that need to do so on-the-fly. Instead, it can just chug along through each row, stream from a repository new slices as it goes, reaching blistering secs/step.

### Resharding

If the final size per parquet file is too large, use `src/bin/reshard.rs` to re-adjust it. Its configurations can likewise be modified at the top of the file.

