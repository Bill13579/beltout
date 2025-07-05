//! Configuration file for how to bake the dataset.
//! 
//! The final dataset is baked in two steps.
//! 
//! Each source dataset first gets grouped into groups based on desirable training properties. 
//! Within these groups, each dataset is repeated >=1.0 times until the composition of the dataset 
//! matches the desired one defined in "probabilities". The >=1.0 times ensures that we achieve the target 
//! composition without having to miss out on any training samples.
//! 
//! Next up, each group contributes a predefined number of samples to each batch. Detailed 
//! information on why we do not use randomization but instead make each group contribute 
//! a fixed number of samples is described in the technical report; the most important idea 
//! is that we hope the model will learn how a human's vocalization works based on their timbre/vocal cords. 
//! Unfortunately, we do not have clean data on doing exactly that, especially for singing. We *can* however approximate 
//! such a dataset by having every batch consist of every single variety of vocalizations we can 
//! think of, where there are natural differences based on language patterns and singing styles (for the few singing examples we do have). 
//! The average, we then hope by the Law of Large Numbers, will be the general vocalization techniques 
//! applicable to all humans.

use phf::phf_map;

pub const TARGET_SR: u32 = 24000;

// How many rows per parquet file.
pub const PARQUET_CHUNK_SIZE: usize = 64;
// pub const PARQUET_CHUNK_SIZE: usize = 938;

// How many threads to use during step 2, when the resampled data is built into the final dataset. 
// This process is incredibly memory intensive, and each thread that participates will increase 
// the memory usage even further. With memory in abundance, setting this to 5 allows for the process 
// to complete much faster.
pub const BAKE_MERGE_NUM_THREADS: usize = 5;

// A unique, clean name for each dataset component.
pub type DatasetKey = &'static str;

/// Maps our clean name to the glob pattern for its Parquet files.
/// 
/// Make sure to update this with your own paths before running! You can also choose to use 
/// different training data as well.
pub const DATASET_PATHS: phf::Map<DatasetKey, &'static str> = phf_map! {
    // Note: We use glob patterns to handle multi-file datasets.
    "acapella_song1" => "data/acapella/song1-*.parquet",
    "acapella_song2" => "data/acapella/song2-*.parquet",
    "acapella_song3" => "data/acapella/song3-*.parquet",
    "acapella_song4" => "data/acapella/song4-*.parquet",
    "acapella_song5" => "data/acapella/song5-*.parquet",
    "acapella_song6" => "data/acapella/song6-*.parquet",
    "aesdd" => "data/aesdd-*.parquet",
    "bel_canto" => "data/bel_canto/*.parquet",
    "berst_test" => "data/berst/test/*.parquet",
    "berst_train" => "data/berst/train/*.parquet",
    "berst_validation" => "data/berst/validation/*.parquet",
    "cafe" => "data/cafe/*.parquet",
    "casia" => "data/casia-*.parquet",
    "crema_d" => "data/crema_d/*.parquet",
    "emodb" => "data/emodb/*.parquet",
    "emouerj_sed_test" => "data/emouerj_sed/test-*.parquet",
    "emouerj_sed_train" => "data/emouerj_sed/train-*.parquet",
    "emov_db" => "data/emov_db/train-*.parquet",
    "emozionalmente" => "data/emozionalmente/*.parquet",
    "esd" => "data/esd/*.parquet",
    "jl_corpus" => "data/jl_corpus/*.parquet",
    "jvnv" => "data/jvnv/test-*.parquet",
    "mesd" => "data/mesd/*.parquet",
    "nemo" => "data/nemo/*.parquet",
    "oreau" => "data/oreau/*.parquet",
    "ravdess" => "data/ravdess/*.parquet",
    "resd" => "data/resd/*.parquet",
    "savee" => "data/savee-*.parquet",
    "shemo" => "data/shemo/train-*.parquet",
    "vocalset" => "data/vocalset/train-*.parquet",
    "yuemotion_test" => "data/yuemotion/test/*.parquet",
    "yuemotion_train" => "data/yuemotion/train/*.parquet",
    "yuemotion_validation" => "data/yuemotion/validation/*.parquet",
};

// Defines the composition and probabilities for each final group.
pub struct GroupConfig {
    pub components: &'static [DatasetKey],
    pub probabilities: &'static [f64],
}

pub const GROUP_CONFIGS: phf::Map<&'static str, GroupConfig> = phf_map! {
    "Musical" => GroupConfig {
        components: &["vocalset"],
        probabilities: &[1.0],
    },
    "ESD" => GroupConfig {
        components: &["esd"],
        probabilities: &[1.0],
    },
    "Japanese" => GroupConfig {
        components: &["jvnv"],
        probabilities: &[1.0],
    },
    "English1" => GroupConfig {
        components: &["crema_d", "berst_train", "berst_validation", "berst_test"],
        probabilities: &[0.5, 0.5 * (3503. / 4523.), 0.5 * (488. / 4523.), 0.5 * (532. / 4523.)],
    },
    "English2" => GroupConfig {
        components: &["jl_corpus", "ravdess", "savee", "emov_db"],
        probabilities: &[0.2, 0.2, 0.2, 0.4],
    },
    "Chinese" => GroupConfig {
        components: &[
            "casia", "acapella_song1", "acapella_song2", "acapella_song3",
            "acapella_song4", "acapella_song5", "acapella_song6", "bel_canto",
            "yuemotion_train", "yuemotion_validation", "yuemotion_test"
        ],
        probabilities: &[
            0.55, 0.15 * (1./6.), 0.15 * (1./6.), 0.15 * (1./6.), 0.15 * (1./6.),
            0.15 * (1./6.), 0.15 * (1./6.), 0.15, 0.15 * (540./1080.),
            0.15 * (162./1080.), 0.15 * (378./1080.)
        ],
    },
    "Greek" => GroupConfig {
        components: &["aesdd"],
        probabilities: &[1.0],
    },
    "Persian" => GroupConfig {
        components: &["shemo"],
        probabilities: &[1.0],
    },
    "Romance" => GroupConfig {
        components: &[
            "cafe", "oreau", "emozionalmente", "mesd",
            "emouerj_sed_train", "emouerj_sed_test"
        ],
        probabilities: &[0.125, 0.125, 0.25, 0.25, 0.25 * (199./285.), 0.25 * (86./285.)],
    },
    "Other" => GroupConfig {
        components: &["emodb", "nemo", "resd"],
        probabilities: &[0.3, 0.5, 0.2],
    },
};

// Defines the final batch structure.
pub const BATCH_COMPOSITION: phf::Map<&'static str, usize> = phf_map! {
    // Musical training data is hard to come by. By mixing at least 8 samples 
    // from VocalSet, we can stretch what we have. Combined with the randomization 
    // of the other datasets, we can effectively get the effect of a musical 
    // dataset many times larger with just a training optimization.
    "Musical" => 8,
    "ESD" => 3,
    // The Japanese (JVNV) dataset as well as the Greek and Persian datasets are particularly 
    // emotive, along with being clean. They serve as further help for the model to learn 
    // more about what our vocal cords are capable of. Thus, we ensure they always get spots in each batch. 
    // In general, the balance of these numbers is carefully adjusted to maximize the model's exposure to a 
    // variety of vocal "techniques" per batch. The averaging that happens to the gradients then points toward a general 
    // "this is how people's voices should sound like based on their vocal cords (x-vectors) when they are trying to vocalize `x`" direction instead of 
    // learning any specific type of voice.
    "Japanese" => 3,
    "English1" => 4,
    "English2" => 3,
    "Chinese" => 2,
    "Greek" => 1,
    "Persian" => 1,
    "Romance" => 4,
    "Other" => 3,
};

