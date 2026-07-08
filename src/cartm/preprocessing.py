from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import queue
import re
import threading
from collections import Counter, deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from typing import Any, TypeAlias

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from jax import Array
from nltk.corpus import stopwords as default_stopwords


Batch: TypeAlias = tuple[Array, Array, Array]
DocumentSource: TypeAlias = Iterable[str] | Callable[[], Iterable[str]]

_TOKEN_RE = re.compile(r"[a-zA-Z]+")

_SENTINEL: Any = object()


class _PrefetchError:
    __slots__ = ("exc",)

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc


def _prefetch(source: Iterator[Batch], size: int) -> Iterator[Batch]:
    q: queue.Queue[Any] = queue.Queue(maxsize=size)
    stop = threading.Event()

    def put(item: Any) -> bool:
        while not stop.is_set():
            try:
                q.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def produce() -> None:
        try:
            for item in source:
                if not put(item):
                    return
            put(_SENTINEL)
        except BaseException as exc:  # noqa: BLE001 - forwarded to consumer
            put(_PrefetchError(exc))

    worker = threading.Thread(
        target=produce, name="CorpusDataLoader-prefetch", daemon=True
    )
    worker.start()

    try:
        while True:
            item = q.get()
            if item is _SENTINEL:
                break
            if isinstance(item, _PrefetchError):
                raise item.exc
            yield item
    finally:
        stop.set()
        worker.join()


class _EncoderConfig:
    """Self-contained, picklable snapshot of all preprocessing settings."""

    __slots__ = (
        "preprocessor", "tokenizer", "token_normalizer",
        "stopwords", "lower", "min_len", "max_len", "vocab",
    )

    def __init__(
        self,
        preprocessor: Callable[[str], str] | None,
        tokenizer: Callable[[str], Iterable[str]] | None,
        token_normalizer: Callable[[str], str] | None,
        stopwords: frozenset[str],
        lower: bool,
        min_len: int,
        max_len: int,
        vocab: dict[str, int] | None,
    ) -> None:
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.token_normalizer = token_normalizer
        self.stopwords = stopwords
        self.lower = lower
        self.min_len = min_len
        self.max_len = max_len
        self.vocab = vocab

    def __getstate__(self):
        return {s: getattr(self, s) for s in self.__slots__}

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)

    def tokenize(self, text: str) -> list[str]:
        if self.tokenizer is not None:
            tokens: Iterable[str] = self.tokenizer(text)
        else:
            tokens = _TOKEN_RE.findall(text)

        if self.lower:
            tokens = [t.lower() for t in tokens]

        stop = self.stopwords
        lo, hi = self.min_len, self.max_len
        normalize = self.token_normalizer

        if normalize is None:
            # Fast path (min_len >= 1 also drops empty tokens).
            return [t for t in tokens if t not in stop and lo <= len(t) <= hi]

        cleaned: list[str] = []
        append = cleaned.append
        for token in tokens:
            if token in stop:
                continue
            token = normalize(token)
            if token in stop:
                continue
            if lo <= len(token) <= hi:
                append(token)
        return cleaned

    def process(self, doc: str) -> list[str]:
        if self.preprocessor is not None:
            doc = self.preprocessor(doc)
        return self.tokenize(doc)

    def encode(self, doc: str) -> np.ndarray:
        get = self.vocab.get  # type: ignore[union-attr]
        return np.asarray(
            [i for t in self.process(doc) if (i := get(t)) is not None],
            dtype=np.int32,
        )


_WORKER_CFG: _EncoderConfig | None = None


def _init_worker(cfg: _EncoderConfig) -> None:
    global _WORKER_CFG
    _WORKER_CFG = cfg


def _encode_chunk_worker(docs: list[str]) -> list[np.ndarray]:
    cfg = _WORKER_CFG
    assert cfg is not None and cfg.vocab is not None
    encode = cfg.encode
    return [encode(d) for d in docs]


def _df_chunk_worker(docs: list[str]) -> tuple[int, Counter[str]]:
    cfg = _WORKER_CFG
    assert cfg is not None
    df: Counter[str] = Counter()
    update = df.update
    process = cfg.process
    for d in docs:
        update(set(process(d)))
    return len(docs), df


_CACHE_TOKENS = "tokens.int32.bin"
_CACHE_OFFSETS = "offsets.int64.bin"
_CACHE_META = "meta.json"


class _EncodedCacheWriter:
    """Streams encoded docs to disk; only valid once `finalize()` completes."""

    def __init__(self, cache_dir: str, fingerprint: str) -> None:
        os.makedirs(cache_dir, exist_ok=True)
        self._dir = cache_dir
        self._fingerprint = fingerprint
        self._tokens_tmp = os.path.join(cache_dir, _CACHE_TOKENS + ".tmp")
        self._offsets_tmp = os.path.join(cache_dir, _CACHE_OFFSETS + ".tmp")
        self._tokens_f = open(self._tokens_tmp, "wb")
        self._offsets_f = open(self._offsets_tmp, "wb")
        self._offsets_f.write(np.int64(0).tobytes())
        self._n_tokens = 0
        self._n_docs = 0
        self._finalized = False

    def append(self, arr: np.ndarray) -> None:
        if arr.size:
            self._tokens_f.write(
                np.ascontiguousarray(arr, dtype=np.int32).tobytes()
            )
            self._n_tokens += int(arr.size)
        self._n_docs += 1
        self._offsets_f.write(np.int64(self._n_tokens).tobytes())

    def finalize(self) -> None:
        self._tokens_f.close()
        self._offsets_f.close()
        os.replace(self._tokens_tmp, os.path.join(self._dir, _CACHE_TOKENS))
        os.replace(self._offsets_tmp, os.path.join(self._dir, _CACHE_OFFSETS))
        meta = {
            "fingerprint": self._fingerprint,
            "n_docs": self._n_docs,
            "n_tokens": self._n_tokens,
        }
        meta_tmp = os.path.join(self._dir, _CACHE_META + ".tmp")
        with open(meta_tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f)
        os.replace(meta_tmp, os.path.join(self._dir, _CACHE_META))
        self._finalized = True

    def close(self) -> None:
        if self._finalized:
            return
        for f in (self._tokens_f, self._offsets_f):
            try:
                f.close()
            except OSError:
                pass
        for path in (self._tokens_tmp, self._offsets_tmp):
            try:
                os.remove(path)
            except OSError:
                pass


class CorpusDataLoader:
    """
    Lazy text corpus loader. It handles all preprocessing logic:
    1. preprocess raw documents;
    2. tokenize and normalize tokens;
    3. map tokens to vocabulary ids;
    4. pack token ids into fixed-size batches;
    5. emit JAX arrays suitable for `ModelBase.fit`.

    Note that the loader stores fitted vocabulary.

    Returns batch tuples:

        data_batch:       (batch_size,)
        doc_bounds_batch: (batch_size,)
        valid_mask:       (batch_size,)

    `valid_mask == False` marks padding tokens.
    """

    def __init__(  # noqa: C901
        self,
        data: DocumentSource,
        *,
        batch_size: int = 10000,
        split_documents: bool = False,
        lower: bool = True,
        vocabulary: dict[str, int] | None = None,
        preprocessor: Callable[[str], str] | None = None,
        tokenizer: Callable[[str], Iterable[str]] | None = None,
        token_normalizer: Callable[[str], str] | None = None,
        stopwords: Iterable[str] | None = None,
        pad_token_id: int = -1,
        min_token_len: int = 2,
        max_token_len: int = 20,
        min_df: int | float = 1,
        max_df: int | float = 1.0,
        prefetch_batches: int = 2,
        num_workers: int = 0,
        worker_chunk_size: int = 256,
        cache_dir: str | os.PathLike[str] | None = None,
    ):
        """
        Args:
            data: re-iterable source which iterates over documents.
            batch_size: size of a single batch.
            split_documents: if True, each document with
                length > batch_size will be split into two or more batches.
                If false, an error will be raised if such document
                will be encountered in data.
            lower: convert all characters to lowercase per token after tokenization.
            vocabulary: mapping (e.g., a dict) where keys are terms and values
                are unique integers from 0 to len(vocabulary). If not given,
                a vocabulary is inferred from the input documents.
            preprocessor: override the preprocessing stage.
            tokenizer: override the tokenizer stage.
            token_normalizer: normalizer applied to each token
                in splitted text, typically a stemmer or a lemmatizer.
            stopwords: terms to be ignored in tokenized data.
                If None, uses default english stopwords from nltk module.
            pad_token_id: token used for padding.
            min_token_len: if length of a normalized token is less than
                min_token_len, it is ignored.
            max_token_len: if length of a normalized token is more than
                max_token_len, it is ignored.
            min_df: if document frequency of a normalized token is less than
                min_df, it is ignored. If float in range [0.0, 1.0],
                the parameter represents a proportion of documents,
                integer absolute counts.
            max_df: if document frequency of a normalized token is more than
                max_df, it is ignored. If float in range [0.0, 1.0],
                the parameter represents a proportion of documents,
                integer absolute counts.
            prefetch_batches: number of prefetched batches.
            num_workers: number of processes used for preprocessing.
            worker_chunk_size: preprocessing size per each process.
            cache_dir: directory where preprocessed data is cached. If None, caching is disabled.
        """
        self._data = data

        if not callable(data):
            try:
                data_iter = iter(data)
            except TypeError as exc:
                raise TypeError(
                    "Data must be iterable or a callable returning an iterable"
                ) from exc

            if data_iter is data:
                raise TypeError(
                    "CorpusDataLoader must be re-iterable. "
                    "You passed a one-shot iterator/generator. "
                    "Pass a sequence or a callable that returns a fresh iterator instead."
                )

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if prefetch_batches < 0:
            raise ValueError("prefetch_batches must be >= 0")
        if num_workers < 0:
            raise ValueError("num_workers must be >= 0")
        if worker_chunk_size <= 0:
            raise ValueError("worker_chunk_size must be positive")

        self.batch_size = batch_size
        self.pad_token_id = pad_token_id
        self.split_documents = split_documents
        self.prefetch_batches = prefetch_batches

        self._num_workers = num_workers
        self._worker_chunk_size = worker_chunk_size
        self._cache_dir = os.fspath(cache_dir) if cache_dir is not None else None

        self._lower = lower
        self._vocab = dict(vocabulary) if vocabulary is not None else None
        self._preprocessor = preprocessor
        self._tokenizer = tokenizer
        self._token_normalizer = token_normalizer
        self._min_token_len = min_token_len
        self._max_token_len = max_token_len
        self._min_df = min_df
        self._max_df = max_df

        self._vocab_version = 0
        self._cfg: _EncoderConfig | None = None
        self._cfg_version = -1
        self._executor: ProcessPoolExecutor | None = None
        self._executor_version = -1
        self._fingerprint: str | None = None
        self._fingerprint_version = -1
        self._cache_write_active = False

        self.n_docs_: int | None = None
        self.doc_freq_: Counter[str] | None = None

        if self._vocab is not None:
            self._validate_vocabulary(self._vocab)

        if preprocessor is not None and not callable(preprocessor):
            raise TypeError(f"preprocessor must be callable, got {type(preprocessor)}")
        if tokenizer is not None and not callable(tokenizer):
            raise TypeError(f"tokenizer must be callable, got {type(tokenizer)}")
        if token_normalizer is not None and not callable(token_normalizer):
            raise TypeError(
                f"token_normalizer must be callable, got {type(token_normalizer)}"
            )

        if stopwords is None:
            stopwords = default_stopwords.words("english")

        if self._lower:
            self._stopwords = frozenset(w.lower() for w in stopwords)
        else:
            self._stopwords = frozenset(stopwords)

        if min_token_len < 1:
            raise ValueError("min_token_len must be >= 1")
        if max_token_len < min_token_len:
            raise ValueError("max_token_len must be >= min_token_len")

        self._validate_df_threshold("min_df", min_df)
        self._validate_df_threshold("max_df", max_df)

    @staticmethod
    def _validate_vocabulary(vocabulary: dict[str, int]) -> None:
        if not all(isinstance(k, str) for k in vocabulary):
            raise TypeError("Vocabulary keys must be strings")
        if not all(isinstance(v, int) for v in vocabulary.values()):
            raise TypeError("Vocabulary values must be integer ids")
        ids = list(vocabulary.values())
        if sorted(ids) != list(range(len(ids))):
            raise ValueError(
                "Vocabulary should contain contiguous token ids starting at 0"
            )

    @staticmethod
    def _validate_df_threshold(name: str, value: int | float) -> None:
        if isinstance(value, float):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} as float must be in [0.0, 1.0]")
        elif value < 1:
            raise ValueError(f"{name} as int must be >= 1")

    def _config(self) -> _EncoderConfig:
        if self._cfg is None or self._cfg_version != self._vocab_version:
            self._cfg = _EncoderConfig(
                preprocessor=self._preprocessor,
                tokenizer=self._tokenizer,
                token_normalizer=self._token_normalizer,
                stopwords=self._stopwords,
                lower=self._lower,
                min_len=self._min_token_len,
                max_len=self._max_token_len,
                vocab=self._vocab,
            )
            self._cfg_version = self._vocab_version
        return self._cfg

    def _get_executor(self) -> ProcessPoolExecutor:
        if (
            self._executor is not None
            and self._executor_version == self._vocab_version
        ):
            return self._executor

        self._shutdown_executor()
        cfg = self._config()
        try:
            pickle.dumps(cfg)
        except Exception as exc:
            raise ValueError(
                "num_workers > 0 requires preprocessor, tokenizer, "
                "token_normalizer and stopwords to be picklable "
                "(define custom callables at module level, not as lambdas "
                "or closures). Use num_workers=0 to disable multiprocessing."
            ) from exc

        self._executor = ProcessPoolExecutor(
            max_workers=self._num_workers,
            initializer=_init_worker,
            initargs=(cfg,),
        )
        self._executor_version = self._vocab_version
        return self._executor

    def _shutdown_executor(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
            self._executor_version = -1

    def close(self) -> None:
        """Release the worker pool (optional; also happens on GC)."""
        self._shutdown_executor()

    def __del__(self) -> None:
        try:
            self._shutdown_executor()
        except Exception:  # noqa: BLE001 - interpreter may be shutting down
            pass

    def _iter_documents(self) -> Iterator[str]:
        docs = self._data() if callable(self._data) else self._data
        return iter(docs)

    def _iter_doc_chunks(self) -> Iterator[list[str]]:
        size = self._worker_chunk_size
        chunk: list[str] = []
        for doc in self._iter_documents():
            chunk.append(doc)
            if len(chunk) >= size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

    def _parallel_chunks(self, worker_fn: Callable[[list[str]], Any]) -> Iterator[Any]:
        """Ordered parallel map over document chunks with bounded look-ahead."""
        executor = self._get_executor()
        max_pending = self._num_workers * 2 + 2
        pending: deque[Any] = deque()
        try:
            for chunk in self._iter_doc_chunks():
                pending.append(executor.submit(worker_fn, chunk))
                if len(pending) >= max_pending:
                    yield pending.popleft().result()
            while pending:
                yield pending.popleft().result()
        finally:
            while pending:
                pending.popleft().cancel()

    def _preprocess_text(self, text: str) -> str:
        if self._preprocessor is not None:
            text = self._preprocessor(text)
        return text

    def _tokenize(self, text: str) -> list[str]:
        return self._config().tokenize(text)

    def process_doc(self, doc: str) -> list[str]:
        text = self._preprocess_text(doc)
        return self._tokenize(text)

    def _resolve_df_thresholds(self, n_docs: int) -> tuple[int, int]:
        if isinstance(self._min_df, float):
            min_df = math.ceil(self._min_df * n_docs)
        else:
            min_df = self._min_df

        if isinstance(self._max_df, float):
            max_df = math.floor(self._max_df * n_docs)
        else:
            max_df = self._max_df

        if min_df > max_df:
            raise ValueError("min_df must be <= max_df")

        return min_df, max_df

    def fit(self, *, force: bool = False) -> CorpusDataLoader:
        """Learn vocabulary by streaming over the raw corpus once."""
        if self._vocab is not None and not force:
            return self

        n_docs = 0
        doc_freq: Counter[str] = Counter()

        if self._num_workers > 0:
            for n, df in self._parallel_chunks(_df_chunk_worker):
                n_docs += n
                doc_freq.update(df)
        else:
            process_doc = self.process_doc
            update = doc_freq.update
            for doc in self._iter_documents():
                n_docs += 1
                update(set(process_doc(doc)))

        if n_docs == 0:
            raise ValueError("Cannot fit vocabulary on an empty corpus")

        min_df, max_df = self._resolve_df_thresholds(n_docs)

        vocab_words = [
            word for word, df in doc_freq.items() if min_df <= df <= max_df
        ]
        vocab_words.sort()

        if len(vocab_words) == 0:
            raise ValueError(
                "Vocabulary is empty after applying min_df/max_df/token filters"
            )

        self._vocab = {word: i for i, word in enumerate(vocab_words)}
        self._vocab_version += 1  # invalidates config, pool and cache key
        self.n_docs_ = n_docs
        self.doc_freq_ = doc_freq

        return self

    def _require_vocabulary(self) -> dict[str, int]:
        if self._vocab is None:
            raise ValueError("Vocabulary is not fitted. Call `fit()` first.")
        return self._vocab

    def _cache_fingerprint(self) -> str:
        if self._fingerprint_version == self._vocab_version:
            assert self._fingerprint is not None
            return self._fingerprint

        vocab = self._require_vocabulary()
        h = hashlib.sha1()
        h.update(
            repr((self._lower, self._min_token_len, self._max_token_len)).encode()
        )
        for word in sorted(vocab):
            h.update(word.encode("utf-8", "replace"))
            h.update(b"\x00")
            h.update(str(vocab[word]).encode())
        self._fingerprint = h.hexdigest()
        self._fingerprint_version = self._vocab_version
        return self._fingerprint

    def _iter_cached_encoded(self) -> Iterator[np.ndarray] | None:
        if self._cache_dir is None:
            return None

        meta_path = os.path.join(self._cache_dir, _CACHE_META)
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, ValueError):
            return None

        if meta.get("fingerprint") != self._cache_fingerprint():
            return None

        n_docs = int(meta["n_docs"])
        n_tokens = int(meta["n_tokens"])
        tokens_path = os.path.join(self._cache_dir, _CACHE_TOKENS)
        offsets_path = os.path.join(self._cache_dir, _CACHE_OFFSETS)

        try:
            offsets = np.memmap(
                offsets_path, dtype=np.int64, mode="r", shape=(n_docs + 1,)
            )
            if n_tokens > 0:
                tokens = np.memmap(
                    tokens_path, dtype=np.int32, mode="r", shape=(n_tokens,)
                )
            else:
                tokens = np.empty(0, dtype=np.int32)
        except (OSError, ValueError):
            return None

        def gen() -> Iterator[np.ndarray]:
            for i in range(n_docs):
                yield tokens[offsets[i]:offsets[i + 1]]

        return gen()

    def _iter_encoded_serial(self) -> Iterator[np.ndarray]:
        vocab = self._require_vocabulary()
        get = vocab.get
        process_doc = self.process_doc
        for doc in self._iter_documents():
            yield np.asarray(
                [i for t in process_doc(doc) if (i := get(t)) is not None],
                dtype=np.int32,
            )

    def _iter_encoded_arrays(self) -> Iterator[np.ndarray]:
        self._require_vocabulary()

        cached = self._iter_cached_encoded()
        if cached is not None:
            yield from cached
            return

        if self._num_workers > 0:
            source: Iterator[np.ndarray] = (
                arr
                for chunk in self._parallel_chunks(_encode_chunk_worker)
                for arr in chunk
            )
        else:
            source = self._iter_encoded_serial()

        writer: _EncodedCacheWriter | None = None
        if self._cache_dir is not None and not self._cache_write_active:
            writer = _EncodedCacheWriter(self._cache_dir, self._cache_fingerprint())
            self._cache_write_active = True

        if writer is None:
            yield from source
            return

        try:
            for arr in source:
                writer.append(arr)
                yield arr
            writer.finalize()
        finally:
            writer.close()  # no-op if finalized, aborts otherwise
            self._cache_write_active = False

    def iter_encoded_docs(self) -> Iterator[list[int]]:
        for arr in self._iter_encoded_arrays():
            yield arr.tolist()

    def _assemble_batch(
        self,
        chunks: list[np.ndarray],
        bound_positions: list[int],
        real_len: int,
    ) -> Batch:
        if real_len == 0:
            raise ValueError("Cannot create an empty batch")

        if real_len > self.batch_size:
            raise ValueError(
                f"Internal error: batch length {real_len} exceeds "
                f"batch_size={self.batch_size}"
            )

        data = np.full(self.batch_size, self.pad_token_id, dtype=np.int32)
        pos = 0
        for chunk in chunks:
            n = chunk.shape[0]
            data[pos:pos + n] = chunk
            pos += n

        bounds = np.zeros(self.batch_size, dtype=bool)
        if bound_positions:
            bounds[bound_positions] = True
        if real_len < self.batch_size:
            bounds[real_len] = True

        valid_mask = np.zeros(self.batch_size, dtype=bool)
        valid_mask[:real_len] = True

        return (
            jnp.asarray(data, dtype=jnp.int32),
            jnp.asarray(bounds, dtype=jnp.bool_),
            jnp.asarray(valid_mask, dtype=jnp.bool_),
        )

    def _iter_batches(self) -> Iterator[Batch]:  # noqa: C901
        batch_size = self.batch_size
        split_documents = self.split_documents

        chunks: list[np.ndarray] = []
        bound_positions: list[int] = []
        total = 0

        def flush() -> Batch:
            nonlocal total
            batch = self._assemble_batch(chunks, bound_positions, total)
            chunks.clear()
            bound_positions.clear()
            total = 0
            return batch

        def append_segment(segment: np.ndarray) -> None:
            nonlocal total
            n = segment.shape[0]
            if n == 0:
                return
            if total > 0:
                bound_positions.append(total)
            chunks.append(segment)
            total += n

        for doc_ids in self._iter_encoded_arrays():
            n = doc_ids.shape[0]
            if n == 0:
                continue

            if not split_documents:
                if n > batch_size:
                    raise ValueError(
                        f"Found encoded document of length {n}, "
                        f"but batch_size={batch_size}. "
                        "Either increase batch_size or pass split_documents=True."
                    )

                if total > 0 and total + n > batch_size:
                    yield flush()

                append_segment(doc_ids)
                continue

            offset = 0
            while offset < n:
                if total == batch_size:
                    yield flush()

                take = min(batch_size - total, n - offset)
                append_segment(doc_ids[offset:offset + take])
                offset += take

                if total == batch_size:
                    yield flush()

        if total > 0:
            yield flush()

    def __iter__(self) -> Iterator[Batch]:
        _ = self._require_vocabulary()

        if self.prefetch_batches <= 0:
            return self._iter_batches()

        return _prefetch(self._iter_batches(), self.prefetch_batches)

    @property
    def vocabulary(self) -> dict[str, int] | None:
        """Token -> id mapping."""
        if self._vocab is None:
            return None
        return dict(self._vocab)

    @property
    def vocab_size(self) -> int:
        return len(self._require_vocabulary())


def build_bow_from_loader(loader: CorpusDataLoader) -> sp.csr_matrix:
    indices_chunks: list[np.ndarray] = []
    data_chunks: list[np.ndarray] = []
    indptr: list[int] = [0]
    nnz = 0

    for doc_ids in loader._iter_encoded_arrays():
        if doc_ids.size:
            token_ids, counts = np.unique(doc_ids, return_counts=True)
            indices_chunks.append(token_ids.astype(np.int32, copy=False))
            data_chunks.append(counts.astype(np.uint32, copy=False))
            nnz += token_ids.size
        indptr.append(nnz)

    n_docs = len(indptr) - 1

    if nnz == 0:
        return sp.csr_matrix((n_docs, loader.vocab_size), dtype=np.uint32)

    return sp.csr_matrix(
        (
            np.concatenate(data_chunks),
            np.concatenate(indices_chunks),
            np.asarray(indptr, dtype=np.int64),
        ),
        shape=(n_docs, loader.vocab_size),
        dtype=np.uint32,
    )
