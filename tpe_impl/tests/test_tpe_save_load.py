"""Round-trip test: save a TPEPathModuleList, load it into a fresh instance,
verify the tensors are allclose and that the metadata asserts catch a
mismatched vocabulary.
"""
import sys, os, tempfile, shutil, json
import _bootstrap  # noqa: F401  (sys.path setup)

import torch
from tpe_impl.module.tpe_path_module import TPEPathModuleList
from tpe_impl.module.save_load import save_tpe_state, load_tpe_state


def main():
    # --- Setup: a module list with non-zero weights
    num_layers, vocab_size, dim = 30, 100, 8
    ml = TPEPathModuleList(num_layers=num_layers, path_vocab_size=vocab_size, low_rank_dim=dim)
    with torch.no_grad():
        for m in ml:
            m.emb_top_path_K.weight.normal_(0, 0.1)
            m.emb_top_path_V.weight.normal_(0, 0.1)
            m.emb_left_path_K.weight.normal_(0, 0.1)
            m.emb_left_path_V.weight.normal_(0, 0.1)

    # Build a tiny fake vocab file for the save
    tmp_vocab = {"pad_id": 0, "unk_id": 1,
                 "vocab": [f"tok{i}" for i in range(vocab_size)],
                 "token_to_id": {f"tok{i}": i for i in range(vocab_size)}}

    # Minimal "model" wrapper + minimal TPEConfig for save_tpe_state signature
    class FakeModel:
        def __init__(self, ml):
            self.tpe_path_module = ml

    from tpe_impl.config import TPEConfig

    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_path = os.path.join(tmpdir, "vocab.json")
        with open(vocab_path, "w") as f:
            json.dump(tmp_vocab, f)

        cfg = TPEConfig(use_tpe=True, path_low_rank_dim=dim, path_vocab_path=vocab_path)
        cfg.load_vocab()  # ensure loaded

        # Save
        print("--- Save ---")
        fake = FakeModel(ml)
        save_tpe_state(fake, tmpdir, cfg)
        files = sorted(os.listdir(tmpdir))
        print(f"  files in tmpdir: {files}")
        assert "tpe_modules.safetensors" in files
        assert "tpe_path_vocab.json" in files
        assert "tpe_metadata.json" in files

        # --- Fresh ModuleList + load ---
        print("\n--- Load into fresh ModuleList ---")
        ml2 = TPEPathModuleList(num_layers=num_layers, path_vocab_size=vocab_size, low_rank_dim=dim)
        # Check it starts zero
        assert torch.all(ml2[0].emb_top_path_K.weight == 0)

        fake2 = FakeModel(ml2)
        load_tpe_state(fake2, tmpdir, strict=True)

        # --- Compare all tensors ---
        for i in range(num_layers):
            for attr in ("emb_top_path_K", "emb_top_path_V", "emb_left_path_K", "emb_left_path_V"):
                w1 = getattr(ml[i], attr).weight
                w2 = getattr(ml2[i], attr).weight
                assert torch.allclose(w1, w2, atol=0, rtol=0), \
                    f"layer {i} {attr} differs after save/load"
        print("  ✓ all 120 tensors match exactly (layers × 4 embeddings)")

        # --- Metadata mismatch asserts ---
        print("\n--- Size-mismatch assert test ---")
        # Tamper with metadata file to pretend different vocab_size
        meta_path = os.path.join(tmpdir, "tpe_metadata.json")
        meta = json.loads(open(meta_path).read())
        meta["path_vocab_size"] = vocab_size + 999
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        ml3 = TPEPathModuleList(num_layers=num_layers, path_vocab_size=vocab_size, low_rank_dim=dim)
        fake3 = FakeModel(ml3)
        got_error = False
        try:
            load_tpe_state(fake3, tmpdir, strict=True)
        except AssertionError as e:
            got_error = True
            print(f"  ✓ tampered metadata raises AssertionError: {e}")
        assert got_error, "metadata tampering should be caught"

    print("\n✓ TPE save/load round-trip passed.")


if __name__ == "__main__":
    main()
