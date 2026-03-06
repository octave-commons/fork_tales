from __future__ import annotations

from code.world_web.vecstore_layer_policy import (
    VecstoreLayerContext,
    resolve_vecstore_collection_name,
    resolve_vecstore_layer_token,
)


def test_resolve_vecstore_collection_name_supports_signature_mode() -> None:
    collection = resolve_vecstore_collection_name(
        VecstoreLayerContext(
            base_collection="eta_mu_nexus_v1",
            mode="signature",
            space_id="space:text",
            space_signature="abcDEF1234567890",
            model_name="nomic-embed-text",
        )
    )
    assert collection == "eta_mu_nexus_v1__abcdef123456"


def test_resolve_vecstore_collection_name_supports_alias_modes() -> None:
    token_one = resolve_vecstore_layer_token(
        mode="space-signature",
        space_id="space:image",
        space_signature="sig_ABCDEFGHIJ12345",
        model_name="clip-vit",
    )
    token_two = resolve_vecstore_layer_token(
        mode="space_model",
        space_id="space:image",
        space_signature="sig_ABCDEFGHIJ12345",
        model_name="clip-vit",
    )
    assert token_one == "space_image_sig_abcdef"
    assert token_two == "space_image_clip-vit"


def test_resolve_vecstore_collection_name_handles_disabled_and_custom_modes() -> None:
    disabled = resolve_vecstore_collection_name(
        VecstoreLayerContext(
            base_collection="eta_mu_nexus_v1",
            mode="off",
            space_id="space:text",
            space_signature="sig",
            model_name="model",
        )
    )
    custom = resolve_vecstore_collection_name(
        VecstoreLayerContext(
            base_collection="eta_mu_nexus_v1",
            mode="CUSTOM MODE",
            space_id="space:text",
            space_signature="sig",
            model_name="model",
        )
    )
    assert disabled == "eta_mu_nexus_v1"
    assert custom == "eta_mu_nexus_v1__custom_mode"
