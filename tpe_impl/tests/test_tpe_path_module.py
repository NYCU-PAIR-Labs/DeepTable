"""Unit test for TPEPathModule: instantiate it and exercise pool() on the
edge cases (empty path, partial path, full path, all-pad, all-valid)."""
import sys
import _bootstrap  # noqa: F401  (sys.path setup)

import torch
from tpe_impl.module.tpe_path_module import TPEPathModule, TPEPathModuleList, PATH_PAD


def main():
    print(f"PATH_PAD = {PATH_PAD}")
    V = 20   # path vocab size (small for testing)
    D = 4    # low_rank_dim
    hm = TPEPathModule(path_vocab_size=V, low_rank_dim=D)

    # --- 1. Zero-init sanity
    for name, param in hm.named_parameters():
        assert torch.all(param == 0), f"{name} should be zero-init"
    print("✓ zero-init check passed (all 4 embeddings are 0)")

    # --- 2. pool() on all-PAD input → zero output (not NaN)
    B, L, Dmax = 2, 5, 3
    path_ids = torch.full((B, L, Dmax), PATH_PAD, dtype=torch.long)
    pooled = TPEPathModule.pool(path_ids, hm.emb_top_path_K)
    assert pooled.shape == (B, L, D), f"shape {pooled.shape}"
    assert torch.all(pooled == 0), "all-PAD pool should return zeros"
    assert not torch.any(torch.isnan(pooled)), "no NaN when all pad"
    print("✓ all-PAD pool returns zero vectors (no NaN)")

    # --- 3. pool() with learnable (non-zero) embedding
    torch.manual_seed(0)
    hm2 = TPEPathModule(path_vocab_size=V, low_rank_dim=D)
    with torch.no_grad():
        hm2.emb_top_path_K.weight.copy_(torch.arange(V * D, dtype=torch.float).reshape(V, D))

    # Sample with 3 valid ids [1, 2, 3] at one token, all -1 at another token,
    # mixed [4, -1, -1] at another
    path_ids = torch.tensor([
        [
            [1, 2, 3],       # mean of rows 1,2,3
            [-1, -1, -1],    # all pad → zero
            [4, -1, -1],     # just row 4
            [0, -1, -1],     # just row 0 (which is PAD_PATH but we treat any >=0 as valid)
            [5, 6, -1],      # mean of rows 5,6
        ]
    ], dtype=torch.long)
    pooled = TPEPathModule.pool(path_ids, hm2.emb_top_path_K)

    # Manually compute expected
    W = hm2.emb_top_path_K.weight  # [20, 4]
    exp = torch.stack([
        (W[1] + W[2] + W[3]) / 3,     # all 3 valid
        torch.zeros(D),                # all pad
        W[4],                           # 1 valid
        W[0],                           # id 0 treated as valid
        (W[5] + W[6]) / 2,             # 2 valid
    ])
    exp = exp.unsqueeze(0)  # [1, 5, 4]

    assert torch.allclose(pooled, exp, atol=1e-6), \
        f"pool output mismatch:\n got:\n{pooled}\n exp:\n{exp}"
    print("✓ pool() correctly averages valid ids, ignores PAD, no div-by-zero")

    # --- 4. Shape / dtype sanity on full forward()
    top = torch.tensor([[[1, 2, -1], [3, -1, -1]]], dtype=torch.long)  # [1, 2, 3]
    left = torch.tensor([[[7, -1, -1], [-1, -1, -1]]], dtype=torch.long)
    top_K, top_V, left_K, left_V = hm2(top, left)
    for name, t in [("top_K", top_K), ("top_V", top_V), ("left_K", left_K), ("left_V", left_V)]:
        assert t.shape == (1, 2, D), f"{name} shape {t.shape}"
    # left_V should be zero since emb_left_path_V is zero-init (only emb_top_path_K was overwritten)
    assert torch.all(left_V == 0), "left_V should be 0 (emb_left_path_V untouched)"
    print("✓ forward() returns 4 tensors with correct shapes")

    # --- 5. TPEPathModuleList wrapper
    ml = TPEPathModuleList(num_layers=30, path_vocab_size=V, low_rank_dim=D)
    assert len(ml) == 30
    assert isinstance(ml[0], TPEPathModule)
    state = ml.state_dict()
    keys = list(state.keys())
    # Every emb should be listed with a predictable name pattern
    expected_per_layer = {"emb_top_path_K.weight", "emb_top_path_V.weight",
                          "emb_left_path_K.weight", "emb_left_path_V.weight"}
    for i in range(30):
        for nm in expected_per_layer:
            assert f"{i}.{nm}" in state, f"missing key {i}.{nm}"
    print(f"✓ ModuleList has 30 layers × 4 embeddings, state_dict has {len(keys)} entries")

    # --- 6. Parameter count
    total_params = sum(p.numel() for p in ml.parameters())
    expected = 30 * 4 * V * D
    assert total_params == expected, f"param count {total_params} != expected {expected}"
    print(f"✓ total params = 30 × 4 × {V} × {D} = {total_params:,}")

    print("\n✓ TPEPathModule smoke test passed.")


if __name__ == "__main__":
    main()
