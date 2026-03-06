# inspect_h5.py
import h5py
import json
from pathlib import Path

files = [
    r"D:\Major project\saved models\best_model.h5",   # EfficientNetB3 file
    r"D:\Major project\saved models\brain_tumor_inceptionv3.h5"     # Inception file
]

for fpath in files:
    print("\n" + "="*80)
    print("Inspecting:", fpath)
    p = Path(fpath)
    if not p.exists():
        print("File not found:", fpath)
        continue
    with h5py.File(fpath, "r") as f:
        print("Top-level keys:", list(f.keys()))
        # model_config group
        if "model_config" in f:
            try:
                raw = f["model_config"][()]
                try:
                    cfg = json.loads(raw)
                    print("model_config keys:", list(cfg.keys()))
                except Exception:
                    print("model_config present but could not parse JSON (binary).")
            except Exception as e:
                print("Could not read model_config:", e)

        # keras_metadata in attrs?
        if "keras_metadata" in f.attrs:
            print("keras_metadata present in file attributes.")

        # model_weights
        if "model_weights" in f:
            mw = f["model_weights"]
            keys = list(mw.keys())
            print("model_weights groups (count):", len(keys))
            print("Some layer/group names (first 40):")
            for name in keys[:40]:
                print("  ", name)
            # show sample dataset shapes for some suspect layer groups
            sample_layers = ["stem_conv", "conv1/conv", "conv2_block1_0_conv", "conv5_block32_2_conv"]
            for sl in sample_layers:
                if sl in mw:
                    print(f"\nLayer group found: {sl}")
                    grp = mw[sl]
                    for k in list(grp.keys())[:20]:
                        obj = grp[k]
                        if isinstance(obj, h5py.Dataset):
                            print(f"  {k} -> shape {obj.shape}")
                        elif isinstance(obj, h5py.Group):
                            for subk in list(obj.keys())[:10]:
                                ds = obj[subk]
                                if isinstance(ds, h5py.Dataset):
                                    print(f"  {k}/{subk} -> shape {ds.shape}")
        else:
            print("No 'model_weights' group found. Keys available:", list(f.keys()))
    print("="*80)
