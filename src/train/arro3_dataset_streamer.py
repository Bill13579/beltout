# Streams pre-batched parquet datasets.

import os
import queue
import random
import threading
import logging
import io
from typing import Callable, Dict, Any, Optional

from torch.utils.data import IterableDataset, get_worker_info
from huggingface_hub import hf_hub_download, list_repo_files

# --- Dependencies ---
#pyproject  or  `pip install arro3-core arro3-io huggingface_hub torch scipy`
import arro3.io
# We use arro3 since pyarrow currently has errors 
# ("OSError: Couldn't deserialize thrift: No more data to read." after running for a couple hundred steps)
# loading datasets created with arrow-rs with massive rows.
from scipy.io import wavfile

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')

class Arro3StreamingDataset(IterableDataset):
    """
    A streaming dataset that reads Parquet files from a Hugging Face repository
    using arro3.

    Parquet parts will be downloaded into the specified cache directory.
    """
    def __init__(
        self,
        repo_id: str,
        transform_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
        shuffle: bool = True,
        shuffle_buffer_size: int = 1000,
        seed: int = 42,
        prefetch_files: int = 2,
        repo_type: str = "dataset",
        revision: Optional[str] = None,
        local_dir: str = "./cache",
    ):
        super().__init__()
        self.repo_id = repo_id
        self.transform_fn = transform_fn
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.prefetch_files = max(1, prefetch_files)
        self.repo_type = repo_type
        self.revision = revision
        self.local_dir = local_dir

        self.parquet_files = sorted([
            f for f in list_repo_files(self.repo_id, repo_type=self.repo_type, revision=self.revision)
            if f.endswith('.parquet')
        ])
        
        if not self.parquet_files:
            raise FileNotFoundError(f"No .parquet files found in repo '{self.repo_id}'!")

    def _downloader_thread_target(
        self,
        worker_id: int,
        url_queue: queue.Queue,
        path_queue: queue.Queue,
        file_semaphore: threading.Semaphore,
        stop_event: threading.Event
    ):
        """
        Pulls filenames from a work queue, downloads them, and puts the local 
        path on a results queue.
        """
        while not stop_event.is_set():
            try:
                # --- Non-blocking get ---
                filename_to_download = url_queue.get(block=False)
            except queue.Empty:
                # The work queue is empty, so this epoch is done.
                break

            # Acquire a permit before downloading.
            file_semaphore.acquire()

            try:
                logging.info(f"[W{worker_id}] Permit acquired. Downloading: {filename_to_download}")
                local_path = hf_hub_download(
                    self.repo_id, filename=filename_to_download, repo_type=self.repo_type, revision=self.revision, local_dir=self.local_dir
                )
                path_queue.put(local_path)
            except Exception as e:
                logging.error(f"[W{worker_id}] Failed to download {filename_to_download}: {e}. Skipping.")
                path_queue.put(None) # Note down this failure so that the main thread isn't blocked forever waiting for it.
                file_semaphore.release() # Release permit on failure
                continue
        
        logging.info(f"[W{worker_id}] Downloader has finished its work queue.")

    def _iter_single_parquet_file(self, local_path: str):
        reader = arro3.io.read_parquet(local_path)
        for batch in reader:
            column_names = batch.schema.names
            columns = [batch.column(name) for name in column_names]
            for i in range(batch.num_rows):
                yield self._decode_audio_columns({name: col[i].as_py() for name, col in zip(column_names, columns)})

    def _shuffled_iterator(self, iterator):
        buffer = []
        rng = random.Random(self.seed)
        try:
            for _ in range(self.shuffle_buffer_size):
                buffer.append(next(iterator))
        except StopIteration:
            pass
        rng.shuffle(buffer)
        for item in iterator:
            idx = rng.randint(0, len(buffer) - 1)
            yield buffer[idx]
            buffer[idx] = item
        while buffer:
            yield buffer.pop(rng.randint(0, len(buffer) - 1))

    def _decode_audio_columns(self, example: Dict[str, Any]):
        processed_example = {}
        for key, value in example.items():
            if isinstance(value, dict) and 'bytes' in value and isinstance(value['bytes'], bytes):
                try:
                    sr, audio_data = wavfile.read(io.BytesIO(value['bytes']))
                    processed_example[key] = {'array': audio_data.copy(), 'sampling_rate': sr} # To prevent "UserWarning: The given NumPy array is not writable" later on.
                except Exception as e:
                    logging.warning(f"Audio decode failed for key '{key}': {e}. Passing as is.")
                    processed_example[key] = value
            else:
                processed_example[key] = value
        return processed_example

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        
        files_for_worker = self.parquet_files[worker_id::num_workers]

        # Create worker-specific resources
        url_queue = queue.Queue()
        path_queue = queue.Queue()
        file_semaphore = threading.Semaphore(self.prefetch_files)
        stop_event = threading.Event()
        
        if self.shuffle:
            rng = random.Random(self.seed + worker_id)
            rng.shuffle(files_for_worker)

        for filename in files_for_worker:
            url_queue.put(filename)

        downloader_thread = threading.Thread(
            target=self._downloader_thread_target,
            args=(worker_id, url_queue, path_queue, file_semaphore, stop_event),
            name=f"Downloader-W{worker_id}",
            daemon=True
        )
        downloader_thread.start()

        fail_count = 0

        try:
            # --- The main processing loop ---
            # Loop exactly for the number of files in the epoch.
            for _ in range(len(files_for_worker)):
                try:
                    # A long timeout, e.g., 30 minutes, to catch hangs but not slow downloads
                    local_path = path_queue.get(timeout=1800) 
                except queue.Empty:
                    logging.error(f"[W{worker_id}] Timeout: Downloader thread appears to be stuck. Terminating worker.")
                    # Breaking the loop will trigger the finally block for cleanup
                    break
                if local_path is None: # Sentinel from downloader indicating failure to obtain a parquet file. We acknowledge it, but do nothing about it.
                    fail_count += 1
                    continue

                # Create a generator for all examples in the current file
                iterator_to_process = self._iter_single_parquet_file(local_path)
                if self.shuffle:
                    iterator_to_process = self._shuffled_iterator(iterator_to_process)
                
                for raw_example in iterator_to_process:
                    yield self.transform_fn(raw_example)
                
                # Cleanup and release semaphore AFTER file is fully processed
                try:
                    os.remove(local_path)
                    logging.info(f"[W{worker_id}] Cleaned up file: {os.path.basename(local_path)}")
                except OSError as e:
                    logging.warning(f"[W{worker_id}] Could not clean up file {local_path}: {e}")
                file_semaphore.release()

        finally:
            # Crucial cleanup logic
            logging.info(f"[W{worker_id}] Iterator is finishing. Cleaning up resources.")
            logging.info(f"[W{worker_id}] {fail_count} sets were skipped.")
            stop_event.set()
            file_semaphore.release() # For workers currently blocked on the semaphore. `stop_event` is set first so that once the worker unblocks it is guaranteed to see the stop.
            if downloader_thread.is_alive():
                downloader_thread.join(timeout=5)
            logging.info(f"[W{worker_id}] Cleanup complete.")

