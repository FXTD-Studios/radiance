# Contributing to Radiance

Thank you for contributing to Radiance! This guide helps maintain consistency across the codebase.

## Quick Start

```bash
# Clone and install test dependencies
cd custom_nodes/radiance
pip install -e ".[test]"

# Syntax-check all modules
python -m py_compile $(find . -name "*.py" -not -path "./tests/*")

# Run the test suite
pytest tests/ --timeout=30 -q
```

## Project Structure

```
radiance/
├── __init__.py              # Node registration hub
├── nodes/                   # ComfyUI node classes — organized sub-package
│   ├── ai/                  # AI assist · knowledge base · LLM driver
│   ├── color/               # CDL · colorscience · curves · grade · OCIO · QC
│   ├── generate/            # engine · HDR LoRA · loader · prompt · sampler
│   ├── hdr/                 # ACES2 · colorspace · delivery · encoder · smart · uplift
│   ├── io/                  # EXR I/O · sequence · unified read/write · video I/O
│   ├── monitor/             # Radiance Viewer · realtime preview
│   ├── pipeline/            # audio_cut · DNA · layout · DCC bridge · workspace
│   ├── upscale/             # Tiler · image/video upscale · face restore · router
│   ├── vfx/                 # 3D · depth · motion · optics · overlay · multipass
│   └── video/               # character · DiT adapter · T2V pipeline · HDR video
├── hdr/                     # Core HDR library (vae, color, analysis, io, tonemap)
├── color/                   # LUT · GPU matrix · OCIO view/CDL (9 nodes)
├── sampler_utils.py         # Sampling math (sigma schedules, noise, model detection)
├── color_utils.py           # Pure color-science functions (no ComfyUI deps)
├── gpu_utils.py             # GPU-accelerated helpers (Laplacian blend, local contrast)
├── path_utils.py            # Safe path construction (traversal prevention)
├── tensor_contract.py       # ensure_4d / ensure_5d tensor shape contracts
├── exceptions.py            # RadianceError hierarchy + decorators
├── tests/                   # Test suite (pytest)
├── .github/workflows/       # CI (pytest matrix + smoke test + lint-config)
├── requirements.txt         # Full production deps
├── requirements_windows.txt
├── requirements_linux.txt
├── requirements_mac_silicon.txt
├── pyproject.toml           # Package metadata, test config, coverage config
└── CONTRIBUTING.md          # This file
```

## Adding a New Node

### 1. Choose the right module

Place new nodes in the appropriate `nodes/<group>/` subdirectory:

| Domain | Location |
|---|---|
| Color operations | `nodes/color/` |
| Model loading | `nodes/generate/` |
| HDR / OCIO | `nodes/hdr/` |
| IO / Video | `nodes/io/` |
| VFX effects | `nodes/vfx/` |
| AI / LLM | `nodes/ai/` |

### 2. Follow the node template

```python
class RadianceYourNode:
    """
    One-line description for the class registry.
    """

    CATEGORY = "FXTD STUDIOS/Radiance/◎ YourCategory"
    FUNCTION = "process"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    DESCRIPTION = "Tooltip for the node itself."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "strength": ("FLOAT", {"default": 1.0, "tooltip": "Descriptions are mandatory"}),
            }
        }

    def process(self, image, strength=1.0):
        # Implementation
        return (result,)
```

### 3. Register the node

Add to the module's `NODE_CLASS_MAPPINGS`. Ensure the display name uses the `◎` icon:

```python
NODE_CLASS_MAPPINGS = {
    "RadianceYourNode": RadianceYourNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceYourNode": "◎ Your Node Name",  # ✅ Icon prefix required
}
```

### 4. Write a test

Add `tests/test_your_node.py` covering:
- Node registration (key in `NODE_CLASS_MAPPINGS`)
- `INPUT_TYPES` structure (required keys present)
- Core math / transform correctness (with real numpy/torch data)

```python
from radiance.nodes.your_group.your_module import RadianceYourNode

def test_registration():
    assert "RadianceYourNode" in NODE_CLASS_MAPPINGS

def test_input_types():
    it = RadianceYourNode.INPUT_TYPES()
    assert "image" in it["required"]
```

## Dependencies & Tests

- **No real torch needed for unit tests** — `conftest.py` installs a stub automatically.
- **GPU tests** — mark with `@pytest.mark.gpu` and skip via `-m "not gpu"` in CI.
- **Integration tests** — mark with `@pytest.mark.integration`; they require ComfyUI + weights.
- **Nuke Bridge** — if modifying `nodes_mcp_bridge.py`, verify connectivity with Nuke 13+.
- **Video IO** — if touching `nodes/io/`, test ffmpeg availability.
- **Viewer** — changes to `radiance_viewer.js` require a hard browser refresh.

## Pull Request Process

1. **Branch Naming**:
   - `feat/new-node`
   - `fix/bug-description`
2. **Checklist**:
   - [ ] Code matches the conventions of the module it lives in
   - [ ] Added tooltips to all widgets
   - [ ] Used `◎` icon in display name
   - [ ] Added / updated test file in `tests/`
   - [ ] Verified `pytest tests/ -q` passes locally
3. **Submit PR**: include a detailed description of changes.

## 4.0 feature freeze

4.0 has been feature-frozen since 16 November 2026 (release candidate 1, tag
`v4.0.0-rc1`). Until 4.0.0 is released:

- **`release/4.0` takes fixes only:** bug fixes, crash and security fixes,
  documentation, and fixes to tests and CI. Branch them as `fix/...` from
  `release/4.0` and open the pull request against it.
- **Not on `release/4.0`:** new nodes, new inputs or outputs, new workflows,
  new models or model versions, new Viewer or Radiance-page features, and
  changes to defaults or file names. Those wait for 4.1.
- A fix that changes behaviour someone can see gets a line in the 4.0.0
  section of the CHANGELOG.
- Each fix on `release/4.0` is merged forward into the 4.1 work so it is not
  lost.
- If a fix needs more than a fix (a new input, say), it goes to 4.1 and
  KNOWN_ISSUES.md says so for 4.0.

When the fixes are done and the checks are green, `release/4.0` is tagged
`v4.0.0`, merged into `main`, and the CHANGELOG heading gets its date.

## Version Numbering

- **v4.0.x**: Patch fixes (single source of truth: `pyproject.toml` → read via `importlib.metadata`)
- **v4.1.0**: New features (non-breaking)
- **v5.0.0**: Breaking changes

## Questions?

Open an issue or contact the FXTD Studios team.
