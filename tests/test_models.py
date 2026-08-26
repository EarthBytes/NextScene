from app.models import EMBEDDING_DIM, Interaction, InteractionType, Item, ItemEmbedding, User


def test_models_registered():
    assert User.__tablename__ == "users"
    assert Item.__tablename__ == "items"
    assert Interaction.__tablename__ == "interactions"
    assert ItemEmbedding.__tablename__ == "item_embeddings"


def test_interaction_type_values():
    assert InteractionType.RATING.value == "rating"
    assert InteractionType.TAG.value == "tag"


def test_embedding_dim():
    assert EMBEDDING_DIM == 512
