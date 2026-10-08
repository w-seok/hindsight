import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hindsight_api.engine.embeddings import LocalSTEmbeddings


async def test_local_embeddings_use_sentence_transformer_input_types():
    model = MagicMock()
    model.encode.return_value = np.array([[1.0]])
    model.encode_query.return_value = np.array([[2.0]])
    model.encode_document.return_value = np.array([[3.0]])

    embeddings = LocalSTEmbeddings()
    embeddings._model = model
    embeddings._device_type = "mps"

    with patch("hindsight_api.engine.embeddings.release_local_inference_memory") as release:
        assert await embeddings.encode(["text"]) == [[1.0]]
        assert await embeddings.encode_query(["query"]) == [[2.0]]
        assert await embeddings.encode_documents(["document"]) == [[3.0]]

    model.encode.assert_called_once_with(["text"], convert_to_numpy=True, show_progress_bar=False)
    model.encode_query.assert_called_once_with(["query"], convert_to_numpy=True, show_progress_bar=False)
    model.encode_document.assert_called_once_with(["document"], convert_to_numpy=True, show_progress_bar=False)
    assert release.call_count == 3


def test_sentence_transformer_exposes_input_type_entry_points():
    """Guard the >=5.0 floor in local-ml.

    The test above drives a MagicMock, so it passes on any sentence-transformers
    version while the real one raises AttributeError at the first encode. These
    entry points only exist from 5.0 onwards; assert against the real class so a
    loosened pin fails here instead of at recall time.
    """
    sentence_transformers = pytest.importorskip("sentence_transformers")

    assert hasattr(sentence_transformers.SentenceTransformer, "encode_query")
    assert hasattr(sentence_transformers.SentenceTransformer, "encode_document")


def _fake_sentence_transformer(*, new_name: bool, old_name: bool, dim: int = 64):
    """A stand-in model exposing one or both of the dimension methods.

    sentence-transformers 5.4+ has both and the old one warns on every call; 5.0-5.3
    have only the old one. ``calls`` records which one initialize() reached.
    """
    torch = pytest.importorskip("torch")

    class _FakeSentenceTransformer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(dim, dim))
            self.calls: list[str] = []

        def encode(self, texts, **_kwargs):
            return np.zeros((len(texts), dim), dtype=np.float32)

    fake = _FakeSentenceTransformer()

    if new_name:

        def get_embedding_dimension() -> int:
            fake.calls.append("get_embedding_dimension")
            return dim

        fake.get_embedding_dimension = get_embedding_dimension

    if old_name:

        def get_sentence_embedding_dimension() -> int:
            fake.calls.append("get_sentence_embedding_dimension")
            if new_name:
                warnings.warn(
                    "The `get_sentence_embedding_dimension` method has been renamed to `get_embedding_dimension`.",
                    FutureWarning,
                    stacklevel=2,
                )
            return dim

        fake.get_sentence_embedding_dimension = get_sentence_embedding_dimension

    return fake


async def test_dimension_uses_the_renamed_method_without_a_future_warning():
    pytest.importorskip("sentence_transformers")
    fake = _fake_sentence_transformer(new_name=True, old_name=True)
    embeddings = LocalSTEmbeddings(model_name="test-model", force_cpu=True)

    with patch("sentence_transformers.SentenceTransformer", return_value=fake), warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        await embeddings.initialize()

    assert embeddings.dimension == 64
    assert fake.calls == ["get_embedding_dimension"]


async def test_dimension_falls_back_to_the_old_name_before_sentence_transformers_5_4():
    pytest.importorskip("sentence_transformers")
    fake = _fake_sentence_transformer(new_name=False, old_name=True)
    embeddings = LocalSTEmbeddings(model_name="test-model", force_cpu=True)

    with patch("sentence_transformers.SentenceTransformer", return_value=fake):
        await embeddings.initialize()

    assert embeddings.dimension == 64
    assert fake.calls == ["get_sentence_embedding_dimension"]
