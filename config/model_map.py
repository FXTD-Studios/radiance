"""Model URL map and checkpoint presets — single source of truth for model resources.

Every model URL, preset configuration, and architecture heuristic lives here
so loader_utils and other modules never hardcode URLs or heuristics.
"""
from __future__ import annotations

# 3.5.0: every entry is pinned. "url" names a repository commit, not "main", and
# "sha256" / "size" are that file's LFS digest and length, so Read Models
# downloads exactly this file and installs it only if the digest matches
# (radiance.core.model_fetch). "gated": True marks a repository whose licence
# must be accepted on Hugging Face first; the download then uses HF_TOKEN or
# the huggingface-cli login. Re-pin with tools/pin_models.py when a new model
# is added.

RADIANCE_MODEL_MAP: dict = {
    # Kijai names the file flux1-schnell-fp8-e4m3fn; it is saved under the
    # name the loader lists. (The old URL, .../flux1-schnell-fp8.safetensors,
    # does not exist in that repository.)
    "flux1-schnell-fp8.safetensors": {
        "url": "https://huggingface.co/Kijai/flux-fp8/resolve/2ef6f85f9f4b94634f995ba0bfa84fd3ab865c1d/flux1-schnell-fp8-e4m3fn.safetensors",
        "sha256": "bbdfba27fed8ff3be237523fb37b83821a6c4bbaa1db43ef9288767d0e4042fb",
        "size": 11891329784,
        "type": "diffusion_models",
    },
    "flux1-dev-fp8.safetensors": {
        "url": "https://huggingface.co/Kijai/flux-fp8/resolve/2ef6f85f9f4b94634f995ba0bfa84fd3ab865c1d/flux1-dev-fp8.safetensors",
        "sha256": "1be961341be8f5307ef26c787199f80bf4e0de3c1c0b4617095aa6ee5550dfce",
        "size": 11901525888,
        "type": "diffusion_models",
    },
    "sd_xl_base_1.0.safetensors": {
        "url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/462165984030d82259a11f4367a4eed129e94a7b/sd_xl_base_1.0.safetensors",
        "sha256": "31e35c80fc4829d14f90153f4c74cd59c90b779f6afe05a74cd6120b893f7e5b",
        "size": 6938078334,
        "type": "diffusion_models",
    },
    "t5xxl_fp8_e4m3fn.safetensors": {
        "url": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/6af2a98e3f615bdfa612fbd85da93d1ed5f69ef5/t5xxl_fp8_e4m3fn.safetensors",
        "sha256": "7d330da4816157540d6bb7838bf63a0f02f573fc48ca4d8de34bb0cbfd514f09",
        "size": 4893934904,
        "type": "text_encoders",
    },
    "clip_l.safetensors": {
        "url": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/6af2a98e3f615bdfa612fbd85da93d1ed5f69ef5/clip_l.safetensors",
        "sha256": "660c6f5b1abae9dc498ac2d21e1347d2abdb0cf6c0c0c8576cd796491d9a6cdd",
        "size": 246144152,
        "type": "text_encoders",
    },
    # The FLUX.1 VAE. Same file, byte for byte (same sha256), as
    # black-forest-labs/FLUX.1-dev/ae.safetensors, from Comfy-Org's ungated
    # repackage so it downloads without a Hugging Face login.
    "ae.safetensors": {
        "url": "https://huggingface.co/Comfy-Org/Lumina_Image_2.0_Repackaged/resolve/5b072540ef86570fecb8249c505f23d5bdeb88cd/split_files/vae/ae.safetensors",
        "sha256": "afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38",
        "size": 335304388,
        "type": "vae",
    },
    # Flux.2 Dev (huggingface.co/black-forest-labs/FLUX.2-dev, gated repo, request
    # access first). fp8mixed is Comfy-Org's split/quantized repackaging, ungated.
    "flux2-dev.safetensors": {
        "url": "https://huggingface.co/black-forest-labs/FLUX.2-dev/resolve/26afe3a78bb242c0a8bb181dcc8937bb16e5c66c/flux2-dev.safetensors",
        "sha256": "6159a3f19f829c8e84ba6e9996b7afaf7c0a5f3428677f5b37445778a320d275",
        "size": 64446596128,
        "gated": True,
        "type": "diffusion_models",
    },
    "flux2_dev_fp8mixed.safetensors": {
        "url": "https://huggingface.co/Comfy-Org/flux2-dev/resolve/ed33133cd56476eac818c0943b6f9419b3e4a3a1/split_files/diffusion_models/flux2_dev_fp8mixed.safetensors",
        "sha256": "863a82e4ff950a42a6b0e80bea824828f129eb1a8fbbdbd9e8cb29859127b486",
        "size": 35455599592,
        "type": "diffusion_models",
    },
    "mistral_3_small_flux2_bf16.safetensors": {
        "url": "https://huggingface.co/Comfy-Org/flux2-dev/resolve/ed33133cd56476eac818c0943b6f9419b3e4a3a1/split_files/text_encoders/mistral_3_small_flux2_bf16.safetensors",
        "sha256": "7d79902f60b1aeb3a6de2cfad02f4367b5e300a1387de3d03ac717cfa3df117c",
        "size": 35584897447,
        "type": "text_encoders",
    },
    "mistral_3_small_flux2_fp8.safetensors": {
        "url": "https://huggingface.co/Comfy-Org/flux2-dev/resolve/ed33133cd56476eac818c0943b6f9419b3e4a3a1/split_files/text_encoders/mistral_3_small_flux2_fp8.safetensors",
        "sha256": "e3467b7d912a234fb929cdf215dc08efdb011810b44bc21081c4234cc75b370e",
        "size": 18034640095,
        "type": "text_encoders",
    },
    "mistral_3_small_flux2_fp4_mixed.safetensors": {
        "url": "https://huggingface.co/Comfy-Org/flux2-dev/resolve/ed33133cd56476eac818c0943b6f9419b3e4a3a1/split_files/text_encoders/mistral_3_small_flux2_fp4_mixed.safetensors",
        "sha256": "1ee1ff334d78228d73049ef0ee4fcd21c1700536b5a45c06547af057f92463a7",
        "size": 12275678071,
        "type": "text_encoders",
    },
    "flux2-vae.safetensors": {
        "url": "https://huggingface.co/Comfy-Org/flux2-dev/resolve/ed33133cd56476eac818c0943b6f9419b3e4a3a1/split_files/vae/flux2-vae.safetensors",
        "sha256": "d64f3a68e1cc4f9f4e29b6e0da38a0204fe9a49f2d4053f0ec1fa1ca02f9c4b5",
        "size": 336213556,
        "type": "vae",
    },
    # Flux.2 Klein 9B (huggingface.co/black-forest-labs, full-precision repos
    # gated, -fp8 repos ungated). Base = distillation-free variant, more
    # flexible CFG/steps, same shape as Dev vs Klein.
    "flux-2-klein-9b.safetensors": {
        "url": "https://huggingface.co/black-forest-labs/FLUX.2-klein-9B/resolve/92196c8e11f7b6cf2b7493e037d8c5345c559216/flux-2-klein-9b.safetensors",
        "sha256": "0975d6b77b5f510b99547d6724a208e36527df654e8f6134f59ece3f9f30da58",
        "size": 18157185168,
        "gated": True,
        "type": "diffusion_models",
    },
    "flux-2-klein-9b-fp8.safetensors": {
        "url": "https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8/resolve/902d9d510b51533e07729f19211414a3648b77d2/flux-2-klein-9b-fp8.safetensors",
        "sha256": "865ba09f5b4c3cbd3468a4bd3acb9fcb2f8740c54317482f0bcd4ed1d3655cee",
        "size": 9433061528,
        "gated": True,
        "type": "diffusion_models",
    },
    "flux-2-klein-base-9b.safetensors": {
        "url": "https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B/resolve/32773329fbe7e81a90ef971740e8ba4b0364ecf3/flux-2-klein-base-9b.safetensors",
        "sha256": "4a54fad7f5f741b99eee217198daac20b8d8e515e2a1f5b064fd51cf074f95bd",
        "size": 18157185168,
        "gated": True,
        "type": "diffusion_models",
    },
    "flux-2-klein-base-9b-fp8.safetensors": {
        "url": "https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9b-fp8/resolve/9ecf2143d71542449960c5584340269c6d401449/flux-2-klein-base-9b-fp8.safetensors",
        "sha256": "a9f5028c24a7a96f4f45beb883aad287d9bccc246227a6803edc898ddda42cf4",
        "size": 9567278472,
        "gated": True,
        "type": "diffusion_models",
    },
    "qwen_3_8b.safetensors": {
        "url": "https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/resolve/3f62d9d8ae1fec33c6e91453d5c712855b096b55/split_files/text_encoders/qwen_3_8b.safetensors",
        "sha256": "f0ff9239d56269ca1d05e5f86da6a79fac111af464955681f11c7ab0ec5ef6c1",
        "size": 16381517176,
        "type": "text_encoders",
    },
    "qwen_3_8b_fp8mixed.safetensors": {
        "url": "https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/resolve/3f62d9d8ae1fec33c6e91453d5c712855b096b55/split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors",
        "sha256": "abad16806e0cbabc54e0325d6565847443fe396d5f0be38bb3cd3fe75a1201d6",
        "size": 8664848742,
        "type": "text_encoders",
    },
    "qwen_3_8b_fp4mixed.safetensors": {
        "url": "https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/resolve/3f62d9d8ae1fec33c6e91453d5c712855b096b55/split_files/text_encoders/qwen_3_8b_fp4mixed.safetensors",
        "sha256": "bbf16f981d98e16d080c566134814c4e9f6aadd0d0e1383c60bc44ba939d760d",
        "size": 6802593327,
        "type": "text_encoders",
    },
    # Flux.2 Klein 4B: full-precision UNET and both text encoders share one
    # Comfy-Org repo (ungated), unlike the 9B tier above; -fp8 UNET variants
    # are on their own black-forest-labs repos, also ungated here.
    "flux-2-klein-4b.safetensors": {
        "url": "https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-4b/resolve/5f526678002e43af5551dadb73ce2e8c91b43afe/split_files/diffusion_models/flux-2-klein-4b.safetensors",
        "sha256": "ec3d4e733a771f61c052fb4856c48b336c55eaf2c65487c2a1faeb9bbda7a343",
        "size": 7751105712,
        "type": "diffusion_models",
    },
    "flux-2-klein-4b-fp8.safetensors": {
        "url": "https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-fp8/resolve/5b4408e59397a4a37ccb46afe426d8ed86379441/flux-2-klein-4b-fp8.safetensors",
        "sha256": "97ed34fe0567e436200f2faee3939b88f2b5d99f8af2a4dc16532c4245c0ccb6",
        "size": 4070624520,
        "type": "diffusion_models",
    },
    "flux-2-klein-base-4b.safetensors": {
        "url": "https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-4b/resolve/5f526678002e43af5551dadb73ce2e8c91b43afe/split_files/diffusion_models/flux-2-klein-base-4b.safetensors",
        "sha256": "9c5fed22b76baea749d88fc2abe3ad53245e7b21a0d353a762665eea00043b92",
        "size": 7751105712,
        "type": "diffusion_models",
    },
    "flux-2-klein-base-4b-fp8.safetensors": {
        "url": "https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4b-fp8/resolve/103db268c10d4d3921101b46057671f9ac460da6/flux-2-klein-base-4b-fp8.safetensors",
        "sha256": "44bab3a86fe98b85d21dd2a4729ebdc3ae51fb8a39f76e457e18c724219e6840",
        "size": 4089498488,
        "type": "diffusion_models",
    },
    "qwen_3_4b.safetensors": {
        "url": "https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-4b/resolve/5f526678002e43af5551dadb73ce2e8c91b43afe/split_files/text_encoders/qwen_3_4b.safetensors",
        "sha256": "6c671498573ac2f7a5501502ccce8d2b08ea6ca2f661c458e708f36b36edfc5a",
        "size": 8044982048,
        "type": "text_encoders",
    },
    "qwen_3_4b_fp4_flux2.safetensors": {
        "url": "https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-4b/resolve/5f526678002e43af5551dadb73ce2e8c91b43afe/split_files/text_encoders/qwen_3_4b_fp4_flux2.safetensors",
        "sha256": "3eab03a77adb0ee5304a4e677d5c10ac22f9049c1d7c894adca4f8bb39206ca8",
        "size": 3848213998,
        "type": "text_encoders",
    },
    # LTX-2.3 transformer (46 GB) — only needed for generation, NOT for decoder training.
    "ltx-2.3-22b-dev.safetensors": {
        "url": "https://huggingface.co/Lightricks/LTX-2.3/resolve/5948be4ced3a4493d1f836df64378ff136ddb770/ltx-2.3-22b-dev.safetensors",
        "sha256": "7ab7225325bc403448ea84b6db2269811a880e5118cd2ee2b6282a93d585016f",
        "size": 46149344974,
        "type": "diffusion_models",
    },
    "ltx-2.3-22b-distilled-1.1.safetensors": {  # ALBABIT-FIX: v1.1 — better audio + aesthetics over original distilled
        "url": "https://huggingface.co/Lightricks/LTX-2.3/resolve/5948be4ced3a4493d1f836df64378ff136ddb770/ltx-2.3-22b-distilled-1.1.safetensors",
        "sha256": "b33b7fe4bbfe084f484be4aaf90b0f1d95dca20d403ac4c0e037eb8c4f0af7cc",
        "size": 46149345334,
        "type": "diffusion_models",
    },
    # Superseded by -1.1 above, kept only because it's still a real file and
    # still a hint fallback for whoever downloaded it before -1.1 shipped.
    "ltx-2.3-22b-distilled.safetensors": {
        "url": "https://huggingface.co/Lightricks/LTX-2.3/resolve/5948be4ced3a4493d1f836df64378ff136ddb770/ltx-2.3-22b-distilled.safetensors",
        "sha256": "14409a4d1337a8ded02fa87fb895b17a91ab2c6588f7cc3352e624ff18a689bf",
        "size": 46149345038,
        "type": "diffusion_models",
    },
    # fp8 tier lives on a separate official Lightricks repo, not a subfolder
    # of the main LTX-2.3 one.
    "ltx-2.3-22b-dev-fp8.safetensors": {
        "url": "https://huggingface.co/Lightricks/LTX-2.3-fp8/resolve/1d756cd27fa11c0896c4dfee093cd1bf36c7f7a1/ltx-2.3-22b-dev-fp8.safetensors",
        "sha256": "28606c5b5a06ce56f896d4dfcb20f212739e07a68fbe48e53638188449d26450",
        "size": 29145431166,
        "type": "diffusion_models",
    },
    "ltx-2.3-spatial-upscaler-x2-1.1.safetensors": {
        "url": "https://huggingface.co/Lightricks/LTX-2.3/resolve/5948be4ced3a4493d1f836df64378ff136ddb770/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        "sha256": "5f416311fa8172b65af67530758964708d29a317b830d689a51143b7f91913ed",
        "size": 995743560,
        "type": "latent_upscale_models",
    },
    # ltx-2.3-spatial-upscaler-x2-1.0.safetensors (still a real unet_hints
    # fallback for whoever already has it on disk) was removed from the
    # Lightricks repo, confirmed by Albabit; its old URL 404s now, so it is
    # deliberately not catalogued here for auto-download.
    "gemma_3_12B_it.safetensors": {
        "url": "https://huggingface.co/Comfy-Org/ltx-2/resolve/ccde4ba417d7900669fd56dd292a883cee11ff37/split_files/text_encoders/gemma_3_12B_it.safetensors",
        "sha256": "56eaa964a0d9325d2dc9ecaf7759bfaf0fac78ae36c789bed6e03e275a3729ec",
        "size": 24379468890,
        "type": "text_encoders",
    },
    "gemma_3_12B_it_fp4_mixed.safetensors": {
        "url": "https://huggingface.co/Comfy-Org/ltx-2/resolve/ccde4ba417d7900669fd56dd292a883cee11ff37/split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors",
        "sha256": "aaca463d11e6d8d2a4bdb0d6299214c15ef78a3f73e0ef8113d5a9d0219b3f6d",
        "size": 9447702218,
        "type": "text_encoders",
    },
    # LTX23_video_vae_bf16 / LTX23_audio_vae_bf16 / text_projection: community
    # repackaging (Kijai/LTX2.3_comfy), not an official Lightricks/Comfy-Org
    # repo. Confirmed acceptable to catalogue as a source by Albabit.
    "LTX23_video_vae_bf16.safetensors": {
        "url": "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/6d980fde0d330f2fed6ff8dfdfddb06d88a004e5/vae/LTX23_video_vae_bf16.safetensors",
        "sha256": "01ea62d09bc139f95c5dee7b5c062ad6a3e6cd8be910a1983ac02e7eb5b8ee3b",
        "size": 1452258578,
        "type": "vae",
    },
    "LTX23_audio_vae_bf16.safetensors": {
        "url": "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/6d980fde0d330f2fed6ff8dfdfddb06d88a004e5/vae/LTX23_audio_vae_bf16.safetensors",
        "sha256": "5bc10fa4adecf99dda132d916e23048cbd56797702c5fa50eb5d2079048a38c3",
        "size": 364855188,
        "type": "vae",
    },
    "ltx-2.3_text_projection_bf16.safetensors": {
        "url": "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/6d980fde0d330f2fed6ff8dfdfddb06d88a004e5/text_encoders/ltx-2.3_text_projection_bf16.safetensors",
        "sha256": "911d59bb4cb7708179c9a0045ea0fe41212ecfb77aed3a02702b7c0a8274911f",
        "size": 2312149072,
        "type": "text_encoders",
    },
    # LTX-2 Video-VAE (AutoencoderKLLTX2Video, 32x/8x/128ch, 2.44 GB) — shared by LTX-2
    # and LTX-2.3. This is all the RUDRA decoder needs to encode HDR -> latents.
    "ltx2_vae_config.json": {
        "url": "https://huggingface.co/Lightricks/LTX-2/resolve/dfcc2108383fe1aaa0584bdf55d368a4bdadd90c/vae/config.json",
        "sha256": "1d402aa299d7c58bf6a30232ee1dcbb0aa41f61237fcb569ef27c2135bfba1cc",
        "size": 1324,
        "type": "vae",
    },
    "ltx2_vae.safetensors": {
        "url": "https://huggingface.co/Lightricks/LTX-2/resolve/dfcc2108383fe1aaa0584bdf55d368a4bdadd90c/vae/diffusion_pytorch_model.safetensors",
        "sha256": "107cc359e3c4bce18c53d98686f4b3fe10c4207b6665d89b38b0741270514bfb",
        "size": 2444982370,
        "type": "vae",
    },
    # LTX-2.5 (huggingface.co/Lightricks/LTX-2.5, gated repo -- request access
    # first). Dev = full/flexible-CFG; Distilled = fixed 8-step, CFG=1 required.
    "ltx-2.5-22b-dev-transformer-bf16.safetensors": {
        "url": "https://huggingface.co/Lightricks/LTX-2.5/resolve/5e6e71018ee1756ed329b697a7b4aedc934dfce9/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors",
        "sha256": "792a2bad501ca03262c0bc2ce7a2949e85b142ce18e30894aad5bc849c8e7584",
        "size": 42018190584,
        "gated": True,
        "type": "diffusion_models",
    },
    "ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors": {
        "url": "https://huggingface.co/Lightricks/LTX-2.5/resolve/5e6e71018ee1756ed329b697a7b4aedc934dfce9/diffusion_models/ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors",
        "sha256": "2edbdb4465cd6c3b532cd67a31ddb38a63e97dcad20be3729675e2a4e8caf92b",
        "size": 21504034224,
        "gated": True,
        "type": "diffusion_models",
    },
    "ltx-2.5-22b-distilled-transformer-bf16.safetensors": {
        "url": "https://huggingface.co/Lightricks/LTX-2.5/resolve/5e6e71018ee1756ed329b697a7b4aedc934dfce9/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
        "sha256": "31eb3cad89b9e54e99dd3baf286f70825ac4f6c660a70d9184d895be76d7bff4",
        "size": 42018190584,
        "gated": True,
        "type": "diffusion_models",
    },
    "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors": {
        "url": "https://huggingface.co/Lightricks/LTX-2.5/resolve/5e6e71018ee1756ed329b697a7b4aedc934dfce9/diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
        "sha256": "c4279eeff115cbeaca494bd2183e7d768c38fe85a184dc6afbb7159157c44334",
        "size": 21504034224,
        "gated": True,
        "type": "diffusion_models",
    },
    # ALBABIT-FIX: needs Blackwell GPU + ltx-kernels pipeline -- not usable via
    # plain ComfyUI today, listed for completeness only.
    "ltx-2.5-22b-distilled-transformer-nvfp4.safetensors": {
        "url": "https://huggingface.co/Lightricks/LTX-2.5/resolve/5e6e71018ee1756ed329b697a7b4aedc934dfce9/diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors",
        "sha256": "4b94231e734c1950f8f6826cb8bd8715d94be5b3e04f8256ee060c5bc3886c30",
        "size": 18721548408,
        "gated": True,
        "type": "diffusion_models",
    },
    # Text encoder -- projection layer is baked in (single CLIP slot), unlike 2.3.
    "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors": {
        "url": "https://huggingface.co/Lightricks/LTX-2.5/resolve/5e6e71018ee1756ed329b697a7b4aedc934dfce9/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
        "sha256": "ef7243612fdae7a75cb4d5cee9433e81380675fb6c213bd98ae74a9cd16561d1",
        "size": 26263858182,
        "gated": True,
        "type": "text_encoders",
    },
    "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors": {
        "url": "https://huggingface.co/Lightricks/LTX-2.5/resolve/5e6e71018ee1756ed329b697a7b4aedc934dfce9/text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
        "sha256": "6ce688a0aa98a5fa36a9f1e6c3f42152a498cc2b53ee8c15674c64244f91487f",
        "size": 15372969374,
        "gated": True,
        "type": "text_encoders",
    },
    "ltx-2.5-video-vae-bf16.safetensors": {
        "url": "https://huggingface.co/Lightricks/LTX-2.5/resolve/5e6e71018ee1756ed329b697a7b4aedc934dfce9/vae/ltx-2.5-video-vae-bf16.safetensors",
        "sha256": "847e14ca7f3355debca0cea4eaa24ac0fbcdf0061da054ac89ca638a869ddba3",
        "size": 1472223346,
        "gated": True,
        "type": "vae",
    },
    # ALBABIT-FIX: "ltx-2.5-video-vae-conv-bf16.safetensors" deliberately NOT
    # catalogued -- different (16x/4x) VAE compression Radiance can't detect
    # per-file, would silently halve output resolution if auto-recommended.
    "ltx-2.5-audio-vae-bf16.safetensors": {
        "url": "https://huggingface.co/Lightricks/LTX-2.5/resolve/5e6e71018ee1756ed329b697a7b4aedc934dfce9/vae/ltx-2.5-audio-vae-bf16.safetensors",
        "sha256": "c52733d37f6a7fb7949c3dc0fb468c6cb2169e4d836983a73babb9f0d54837a5",
        "size": 364866540,
        "gated": True,
        "type": "vae",
    },
    "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors": {
        "url": "https://huggingface.co/Lightricks/LTX-2.5/resolve/5e6e71018ee1756ed329b697a7b4aedc934dfce9/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
        "sha256": "eb5a71fe4068ee87ccdb1c3aa635e547ca76bd2d30ae20ae889f2c325c0677e8",
        "size": 995778752,
        "gated": True,
        "type": "latent_upscale_models",
    },
    # ALBABIT-FIX: 2x temporal/duration upscale at the latent level -- genuinely
    # new vs 2.3, not wired into any Radiance node yet.
    "ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors": {
        "url": "https://huggingface.co/Lightricks/LTX-2.5/resolve/5e6e71018ee1756ed329b697a7b4aedc934dfce9/latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors",
        "sha256": "2bc3300f2b3c3c1834d72164fbf13a3b9fd73e5a741e8a2c3f4035f89a75c3fe",
        "size": 261944000,
        "gated": True,
        "type": "latent_upscale_models",
    },
    # ALBABIT-FIX: powers "Auto Duration" (frame-count prediction from the
    # prompt), loaded via ModelPatchLoader + LTXVDurationPredictor (core
    # ComfyUI). Not wired into any Radiance node yet.
    "ltx-2.5-duration-head-bf16.safetensors": {
        "url": "https://huggingface.co/Lightricks/LTX-2.5/resolve/5e6e71018ee1756ed329b697a7b4aedc934dfce9/model_patches/ltx-2.5-duration-head-bf16.safetensors",
        "sha256": "2ec71e4206ed365d015f00c05a48caccfb0ee862986809d06ae376c09f5d9190",
        "size": 3843690,
        "gated": True,
        "type": "model_patches",
    },
    # Official 2.5 equivalent of ltx-2.3-22b-distilled-1.1's distillation LoRA.
    "ltx-2.5-22b-distilled-lora-450-bf16.safetensors": {
        "url": "https://huggingface.co/Lightricks/LTX-2.5/resolve/5e6e71018ee1756ed329b697a7b4aedc934dfce9/loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors",
        "sha256": "86370bbf79a9eb4edaa158907e2b48a5188fe4c5dc8ce30c7eb8f2f131a9bbf5",
        "size": 8899889568,
        "gated": True,
        "type": "loras",
    },
    # MiniMax H3 (huggingface.co/Comfy-Org/MiniMax-H3). fl2va = text/image-to-
    # video, every diffusion_models/text_encoders quantization tier Comfy-Org
    # publishes for it. bf16 is the "MiniMax H3" preset's default UNET
    # (quality-first, same philosophy as the Flux.2 preset): it exceeds a
    # 32GB card's native capacity alongside a text encoder, so ComfyUI's own
    # automatic lowvram partial-load streams the overflow from system RAM,
    # but the real-world cost measured on a 5090 was only about +37%
    # generation time, not a dealbreaker. pruned_int8_convrot (matches the
    # official workflow) is the "MiniMax H3 (Low VRAM)" preset's default
    # instead, for users who want the lighter/faster tier. ref2va (reference-
    # to-video) is a separate checkpoint family, not catalogued. Out of scope
    # for now.
    "minimax_h3_fl2va_bf16.safetensors": {
        "url": "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/bf92c4091e333e69b8ca1998e0a669f15cb0832b/diffusion_models/minimax_h3_fl2va_bf16.safetensors",
        "sha256": "907d4add438438ec1544f5240c3b38532ed934fe6be75677a6bbda2a6fdd6182",
        "size": 66280487368,
        "type": "diffusion_models",
    },
    "minimax_h3_fl2va_pruned_int8_convrot.safetensors": {
        "url": "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/bf92c4091e333e69b8ca1998e0a669f15cb0832b/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "sha256": "e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a",
        "size": 20970379616,
        "type": "diffusion_models",
    },
    "minimax_h3_fl2va_int8_convrot.safetensors": {
        "url": "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/bf92c4091e333e69b8ca1998e0a669f15cb0832b/diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors",
        "sha256": "7ad4c73e6e378b822ffd1629f27f632d3787d95f5e468e3af958f98c58df96a5",
        "size": 34038892334,
        "type": "diffusion_models",
    },
    "minimax_h3_fl2va_pruned_bf16.safetensors": {
        "url": "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/bf92c4091e333e69b8ca1998e0a669f15cb0832b/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors",
        "sha256": "a32572fb90b5508b201ec7c2eddcc184b13ddfd3c6f6d2cf06a0b46535d541b4",
        "size": 40225724176,
        "type": "diffusion_models",
    },
    "minimax_h3_fl2va_pruned_fp8_scaled.safetensors": {
        "url": "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/bf92c4091e333e69b8ca1998e0a669f15cb0832b/diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors",
        "sha256": "12944c1f7791637e7de12208aef04da82bd26b95271b1b47d817364315ade993",
        "size": 20958205608,
        "type": "diffusion_models",
    },
    "minimax_h3_video_vae_fp16.safetensors": {
        "url": "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/bf92c4091e333e69b8ca1998e0a669f15cb0832b/vae/minimax_h3_video_vae_fp16.safetensors",
        "sha256": "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522",
        "size": 5207808496,
        "type": "vae",
    },
    "minimax_h3_audio_vae_fp32.safetensors": {
        "url": "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/bf92c4091e333e69b8ca1998e0a669f15cb0832b/vae/minimax_h3_audio_vae_fp32.safetensors",
        "sha256": "8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48",
        "size": 605254808,
        "type": "vae",
    },
    "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors": {
        "url": "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/bf92c4091e333e69b8ca1998e0a669f15cb0832b/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "sha256": "35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6",
        "size": 15687142551,
        "type": "text_encoders",
    },
    "qwen3vl_32b_minimax_h3_int8_convrot.safetensors": {
        "url": "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/bf92c4091e333e69b8ca1998e0a669f15cb0832b/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
        "sha256": "bc2ced0fbea64757fa9acddccfc0b3f4819d1dcf1da6c124d690d368be283923",
        "size": 27141342152,
        "type": "text_encoders",
    },
    "qwen3vl_32b_minimax_h3_bf16.safetensors": {
        "url": "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/bf92c4091e333e69b8ca1998e0a669f15cb0832b/text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors",
        "sha256": "600d567f6a9629c8574e8e7041b199bdd9c59a986afa7906910a81919610607d",
        "size": 51506295256,
        "type": "text_encoders",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
#  Unified per-model VAE / training configuration  (single source of truth)
# ─────────────────────────────────────────────────────────────────────────────
#
#  Canonical keys match the user-facing names in RADIANCE_MODEL_PRESETS.
#  All training scripts, decoders, and HDR nodes should resolve through
#  resolve_model_vae_config() rather than hard-coding values.

MODEL_VAE_CONFIG: dict[str, dict] = {
    "ltx-video": {
        "latent_channels":     128,
        "scale_factor":        1.0,
        "log_curve":           "Sony S-Log3",
        "compression_ratio":     0.50,
        "norm_center":         3.0,
        "vae_spatial_factor":  32,
        "vae_temporal_factor": 8,
        "noise_schedule":      "flow",
        "text_embed_seq_len":  128,
        "text_embed_hidden":   4096,
        "clip_slots":          ["llm_encoder", "text_projection"],
        "notes":               "LTX-Video / LTX-2.3 Video-VAE (AutoencoderKLLTX2Video): "
                               "spatial_compression_ratio=32, temporal=8, latent_channels=128, "
                               "scaling_factor=1.0. 32x spatial -> log2(32)=5 decoder upsample "
                               "stages (was wrongly 8 -> 3 stages, which broke real-LTX decode).",
    },
    "flux": {
        "latent_channels":     16,
        "scale_factor":        0.18215,
        "log_curve":           "ARRI LogC4",
        "compression_ratio":   0.50,
        "norm_center":         3.0,
        "vae_spatial_factor":  8,
        "vae_temporal_factor": 1,
        "noise_schedule":      "flow",
        "text_embed_seq_len":  256,
        "text_embed_hidden":   4096,
        "clip_slots":          ["clip_l", "t5xxl"],
        "notes":               "Flux.1 image model — validated against Bradford sRGB derivation.",
    },
    # ALBABIT-FIX: Chroma (distilled Flux.1, single T5-XXL encoder)
    "chroma": {
        "latent_channels":     16,
        "scale_factor":        0.18215,
        "log_curve":           "ARRI LogC4",
        "compression_ratio":   0.50,
        "norm_center":         3.0,
        "vae_spatial_factor":  8,
        "vae_temporal_factor": 1,
        "noise_schedule":      "flow",
        "text_embed_seq_len":  256,
        "text_embed_hidden":   4096,
        "clip_slots":          ["t5xxl"],
        "notes":               "Chroma — distilled Flux.1 variant, single T5-XXL encoder (no clip_l), reuses the Flux.1 VAE (16ch).",
    },
    "zimage": {
        "latent_channels":     16,
        "scale_factor":        0.3611,
        "shift_factor":        0.1159,
        "log_curve":           "ARRI LogC4",
        "compression_ratio":   0.50,
        "norm_center":         3.0,
        "vae_spatial_factor":  8,
        "vae_temporal_factor": 1,
        "noise_schedule":      "flow",
        "text_embed_seq_len":  256,
        "text_embed_hidden":   2560,
        "clip_slots":          ["qwen3_4b"],
        "notes":               "Z-Image (Tongyi) — uses the FLUX.1 VAE (16ch). The flux decoder works directly.",
    },
    "qwen": {
        "latent_channels":     16,
        "scale_factor":        0.3611,
        "log_curve":           "ARRI LogC4",
        "compression_ratio":   0.50,
        "norm_center":         3.0,
        "vae_spatial_factor":  8,
        "vae_temporal_factor": 1,
        "noise_schedule":      "flow",
        "text_embed_seq_len":  256,
        "text_embed_hidden":   3584,
        "clip_slots":          ["qwen2.5_vl_7b"],
        "notes":               "Qwen-Image — 16ch VAE (qwen_image_vae.safetensors), 8x. Standard decoder.",
    },
    # ALBABIT-FIX: Flux.2 Dev (Mistral-3 24B encoder) — shares the 128ch VAE with Flux.2 Klein.
    "flux2": {
        "latent_channels":     128,
        "scale_factor":        0.3611,
        "log_curve":           "ARRI LogC4",
        "compression_ratio":   0.50,
        "norm_center":         3.0,
        "vae_spatial_factor":  16,
        "vae_temporal_factor": 1,
        "noise_schedule":      "flow",
        "text_embed_seq_len":  512,
        "text_embed_hidden":   7680,
        "clip_slots":          ["mistral_3_small"],
        "notes":               "Flux.2 Dev — Mistral-3 24B encoder, 128ch latent (post pixel-shuffle), shares the same VAE/DiT conditioning dim as Flux.2 Klein.",
    },
    "flux2-klein": {
        "latent_channels":     128,
        "scale_factor":        0.3611,
        "log_curve":           "ARRI LogC4",
        "compression_ratio":   0.50,
        "norm_center":         3.0,
        "vae_spatial_factor":  16,
        "vae_temporal_factor": 1,
        "noise_schedule":      "flow",
        "text_embed_seq_len":  512,
        "text_embed_hidden":   7680,
        "clip_slots":          ["qwen3_4b"],
        "notes":               "Flux.2 Klein — 128ch latent (post pixel-shuffle), 16x -> 4 upsample stages.",
    },
    # ALBABIT-FIX: Cosmos (16ch, T5XXL-old encoder) and Mochi (12ch, T5XXL encoder)
    "cosmos": {
        "latent_channels":     16,
        "scale_factor":        1.0,
        "log_curve":           "ARRI LogC4",
        "compression_ratio":   0.50,
        "norm_center":         3.0,
        "vae_spatial_factor":  8,
        "vae_temporal_factor": 8,
        "noise_schedule":      "flow",
        "text_embed_seq_len":  512,
        "text_embed_hidden":   1024,
        "clip_slots":          ["t5xxl"],
        "notes":               "NVIDIA Cosmos; 16ch causal video VAE, T5-XXL (old) text encoder.",
    },
    "mochi": {
        "latent_channels":     12,
        "scale_factor":        1.0,
        "log_curve":           "ARRI LogC4",
        "compression_ratio":   0.50,
        "norm_center":         3.0,
        "vae_spatial_factor":  8,
        "vae_temporal_factor": 6,
        "noise_schedule":      "flow",
        "text_embed_seq_len":  256,
        "text_embed_hidden":   4096,
        "clip_slots":          ["t5xxl"],
        "notes":               "Genmo Mochi-1; 12ch latent, T5-XXL text encoder.",
    },
    "cogvideox": {
        "latent_channels":     16,
        "scale_factor":        0.18215,
        "log_curve":           "ARRI LogC4",
        "compression_ratio":   0.45,
        "norm_center":         3.0,
        "vae_spatial_factor":  8,
        "vae_temporal_factor": 4,
        "noise_schedule":      "flow",
        "text_embed_seq_len":  77,
        "text_embed_hidden":   4096,
        "clip_slots":          ["t5xxl"],
        "notes":               "CogVideoX 3D causal VAE; tighter latent distribution → lower ratio.",
    },
    "wan": {
        "latent_channels":     16,
        "scale_factor":        0.18215,
        "log_curve":           "ARRI LogC4",
        "compression_ratio":   0.60,
        "norm_center":         3.5,
        "vae_spatial_factor":  8,
        "vae_temporal_factor": 4,
        "noise_schedule":      "flow",
        "text_embed_seq_len":  128,
        "text_embed_hidden":   4096,
        "clip_slots":          ["t5xxl"],
        "notes":               "Wan 2.1 Flow-VAE; different latent mean target → higher compression.",
    },
    # 3.5: ComfyUI 0.32 families. Decoder conventions follow the VAE each
    # model shares: Krea 2 the Qwen/Wan VAE, HiDream / OmniGen2 / LongCat /
    # Kandinsky 5 image the Flux VAE, Kandinsky 5 video the HunyuanVideo VAE.
    "krea2": {
        "latent_channels":     16,
        "scale_factor":        0.3611,
        "log_curve":           "ARRI LogC4",
        "compression_ratio":   0.50,
        "norm_center":         3.0,
        "vae_spatial_factor":  8,
        "vae_temporal_factor": 1,
        "noise_schedule":      "flow",
        "text_embed_seq_len":  256,
        "text_embed_hidden":   2560,
        "clip_slots":          ["llm_encoder"],
        "notes":               "Krea 2 (K2) — Qwen-Image 16ch VAE, Qwen3-VL 4B encoder.",
    },
    "hidream": {
        "latent_channels":     16,
        "scale_factor":        0.3611,
        "log_curve":           "ARRI LogC4",
        "compression_ratio":   0.50,
        "norm_center":         3.0,
        "vae_spatial_factor":  8,
        "vae_temporal_factor": 1,
        "noise_schedule":      "flow",
        "text_embed_seq_len":  128,
        "text_embed_hidden":   4096,
        "clip_slots":          ["clip_l", "clip_g", "t5xxl", "llm_encoder"],
        "notes":               "HiDream-I1 — Flux 16ch VAE (ae.safetensors), four text encoders.",
    },
    "omnigen2": {
        "latent_channels":     16,
        "scale_factor":        0.3611,
        "log_curve":           "ARRI LogC4",
        "compression_ratio":   0.50,
        "norm_center":         3.0,
        "vae_spatial_factor":  8,
        "vae_temporal_factor": 1,
        "noise_schedule":      "flow",
        "text_embed_seq_len":  256,
        "text_embed_hidden":   2048,
        "clip_slots":          ["llm_encoder"],
        "notes":               "OmniGen2 — Flux 16ch VAE, Qwen2.5-VL 3B encoder.",
    },
    "longcat_image": {
        "latent_channels":     16,
        "scale_factor":        0.3611,
        "log_curve":           "ARRI LogC4",
        "compression_ratio":   0.50,
        "norm_center":         3.0,
        "vae_spatial_factor":  8,
        "vae_temporal_factor": 1,
        "noise_schedule":      "flow",
        "text_embed_seq_len":  256,
        "text_embed_hidden":   3584,
        "clip_slots":          ["llm_encoder"],
        "notes":               "LongCat-Image — Flux 16ch VAE, Qwen2.5-VL 7B encoder, FluxGuidance.",
    },
    "kandinsky5_image": {
        "latent_channels":     16,
        "scale_factor":        0.3611,
        "log_curve":           "ARRI LogC4",
        "compression_ratio":   0.50,
        "norm_center":         3.0,
        "vae_spatial_factor":  8,
        "vae_temporal_factor": 1,
        "noise_schedule":      "flow",
        "text_embed_seq_len":  256,
        "text_embed_hidden":   3584,
        "clip_slots":          ["llm_encoder", "clip_l"],
        "notes":               "Kandinsky 5 Image — Flux 16ch VAE, Qwen2.5-VL 7B + CLIP-L.",
    },
    "kandinsky5": {
        "latent_channels":     16,
        "scale_factor":        0.18215,
        "log_curve":           "ARRI LogC4",
        "compression_ratio":   0.60,
        "norm_center":         3.5,
        "vae_spatial_factor":  8,
        "vae_temporal_factor": 4,
        "noise_schedule":      "flow",
        "text_embed_seq_len":  256,
        "text_embed_hidden":   3584,
        "clip_slots":          ["llm_encoder", "clip_l"],
        "notes":               "Kandinsky 5 Video — HunyuanVideo 16ch VAE, Qwen2.5-VL 7B + CLIP-L.",
    },
    "hunyuan_image": {
        "latent_channels":     64,
        "scale_factor":        0.75289,
        "log_curve":           "ARRI LogC4",
        "compression_ratio":   0.55,
        "norm_center":         3.0,
        "vae_spatial_factor":  32,
        "vae_temporal_factor": 1,
        "noise_schedule":      "flow",
        "text_embed_seq_len":  256,
        "text_embed_hidden":   3584,
        "clip_slots":          ["llm_encoder", "text_projection"],
        "notes":               "HunyuanImage 2.1 — 64ch 32x VAE (comfy HunyuanImage21), Qwen2.5-VL + ByT5.",
    },
    "hunyuan_video_15": {
        "latent_channels":     32,
        "scale_factor":        1.03682,
        "log_curve":           "ARRI LogC4",
        "compression_ratio":   0.60,
        "norm_center":         3.5,
        "vae_spatial_factor":  16,
        "vae_temporal_factor": 4,
        "noise_schedule":      "flow",
        "text_embed_seq_len":  256,
        "text_embed_hidden":   3584,
        "clip_slots":          ["llm_encoder", "text_projection"],
        "notes":               "HunyuanVideo 1.5 — 32ch 16x/4 VAE (comfy HunyuanVideo15), Qwen2.5-VL + ByT5.",
    },
    "hunyuanvideo": {
        "latent_channels":     16,
        "scale_factor":        0.18215,
        "log_curve":           "ARRI LogC4",
        "compression_ratio":   0.60,
        "norm_center":         3.5,
        "vae_spatial_factor":  8,
        "vae_temporal_factor": 4,
        "noise_schedule":      "flow",
        "text_embed_seq_len":  128,
        "text_embed_hidden":   4096,
        "clip_slots":          ["llm_encoder", "clip_l"],
        "notes":               "HunyuanVideo; similar latent conventions to Wan.",
    },
    "sd3": {
        "latent_channels":     16,
        "scale_factor":        1.5305,
        "log_curve":           "ARRI LogC4",
        "compression_ratio":   0.50,
        "norm_center":         3.0,
        "vae_spatial_factor":  8,
        "vae_temporal_factor": 1,
        "noise_schedule":      "flow",
        "text_embed_seq_len":  154,
        "text_embed_hidden":   4096,
        "clip_slots":          ["clip_l", "clip_g", "t5xxl"],
        "notes":               "SD 3 / SD 3.5 — same VAE as Flux, different scale factor.",
    },
    "sdxl": {
        "latent_channels":     4,
        "scale_factor":        0.18215,
        "log_curve":           "ARRI LogC3",
        "compression_ratio":   0.40,
        "norm_center":         2.5,
        "vae_spatial_factor":  8,
        "vae_temporal_factor": 1,
        "noise_schedule":      "ddpm",
        "text_embed_seq_len":  77,
        "text_embed_hidden":   2048,
        "clip_slots":          ["clip_l", "clip_g"],
        "notes":               "SDXL classic 4ch VAE; tighter DR headroom → lower ratio.",
    },
    "sd15": {
        "latent_channels":     4,
        "scale_factor":        0.18215,
        "log_curve":           "ARRI LogC3",
        "compression_ratio":   0.35,
        "norm_center":         2.5,
        "vae_spatial_factor":  8,
        "vae_temporal_factor": 1,
        "noise_schedule":      "ddpm",
        "text_embed_seq_len":  77,
        "text_embed_hidden":   768,
        "clip_slots":          ["clip_l"],
        "notes":               "SD 1.x / 2.x — narrowest latent dynamic range.",
    },
    "lumina2": {
        "latent_channels":     16,
        "scale_factor":        0.18215,
        "log_curve":           "ARRI LogC4",
        "compression_ratio":   0.50,
        "norm_center":         3.0,
        "vae_spatial_factor":  8,
        "vae_temporal_factor": 1,
        "noise_schedule":      "flow",
        "text_embed_seq_len":  77,
        "text_embed_hidden":   4096,
        "clip_slots":          ["t5xxl"],
        "notes":               "Lumina-Next; 16ch latent flow-matching model.",
    },
    "pixart": {
        "latent_channels":     4,
        "scale_factor":        0.18215,
        "log_curve":           "ARRI LogC3",
        "compression_ratio":   0.40,
        "norm_center":         2.5,
        "vae_spatial_factor":  8,
        "vae_temporal_factor": 1,
        "noise_schedule":      "ddpm",
        "text_embed_seq_len":  77,
        "text_embed_hidden":   4096,
        "clip_slots":          ["t5xxl"],
        "notes":               "PixArt-Σ / PixArt-α; 4ch latent DiT with T5 conditioning.",
    },
    "aura_flow": {
        "latent_channels":     4,
        "scale_factor":        0.18215,
        "log_curve":           "ARRI LogC3",
        "compression_ratio":   0.40,
        "norm_center":         2.5,
        "vae_spatial_factor":  8,
        "vae_temporal_factor": 1,
        "noise_schedule":      "ddpm",
        "text_embed_seq_len":  77,
        "text_embed_hidden":   768,
        "clip_slots":          ["clip_l"],
        "notes":               "AuraFlow; 4ch latent flow model.",
    },
}

# Aliases so that architecture detectors (model/detect.py) and CLI args
# resolve to the same canonical config regardless of naming convention.
_MODEL_VAE_ALIASES: dict[str, str] = {
    # LTX variants
    "ltx":         "ltx-video",
    "ltxv":        "ltx-video",
    "ltx-video":   "ltx-video",
    "ltxav":       "ltx-video",
    # Flux variants
    "flux1":       "flux",
    "flux.1":      "flux",
    "flux-dev":    "flux",
    "flux-schnell":"flux",
    # Z-Image variants
    "z_image":     "zimage",
    "z-image":     "zimage",
    # CogVideoX variants
    "cogvideo":    "cogvideox",
    "cogvideox5b": "cogvideox",
    # Cosmos variants
    "cosmos1":     "cosmos",
    "cosmos-1":    "cosmos",
    # Mochi variants
    "mochi1":      "mochi",
    "mochi-1":     "mochi",
    "genmo-mochi": "mochi",
    # Wan variants
    "wanvideo":    "wan",
    "wan2":        "wan",
    "wan2.1":      "wan",
    "wan-2.1":     "wan",
    # 3.5 families
    "qwen_image":  "qwen",
    "qwen-image":  "qwen",
    "krea-2":      "krea2",
    "hidream-i1":  "hidream",
    "omnigen-2":   "omnigen2",
    "longcat":     "longcat_image",
    "kandinsky5-image": "kandinsky5_image",
    "hunyuanimage": "hunyuan_image",
    "hunyuan-image": "hunyuan_image",
    "hunyuanvideo15": "hunyuan_video_15",
    "hunyuan_video_1.5": "hunyuan_video_15",
    # Hunyuan variants
    "hunyuan":     "hunyuanvideo",
    "hunyuan_video":"hunyuanvideo",
    "hyvideo":     "hunyuanvideo",
    # SD3 variants
    "sd3.5":       "sd3",
    "sd3":         "sd3",
    "stable-diffusion-3": "sd3",
    # SDXL variants
    "sd_xl":       "sdxl",
    "sdxl-base":   "sdxl",
    # SD1.x variants
    "sd1":         "sd15",
    "sd2":         "sd15",
    "sd1.5":       "sd15",
    "sd-1.5":      "sd15",
    "v1-5":        "sd15",
    # Lumina2
    "lumina":      "lumina2",
    "lumina-next": "lumina2",
    # PixArt
    "pixart_sigma":"pixart",
    "pixart-alpha":"pixart",
    # AuraFlow
    "auraflow":    "aura_flow",
    "aura":        "aura_flow",
}


def resolve_model_vae_config(model_hint: str) -> dict | None:
    """
    Look up the unified VAE/training config for a model name or alias.

    Args:
        model_hint: Any model name string (e.g. 'flux', 'wan2.1',
                    'hunyuan_video', 'sd1.5'). Case-insensitive.

    Returns:
        A dict with all VAE/training parameters, or None if not found.
    """
    key = model_hint.strip().lower()
    if key in MODEL_VAE_CONFIG:
        return MODEL_VAE_CONFIG[key]
    if key in _MODEL_VAE_ALIASES:
        canonical = _MODEL_VAE_ALIASES[key]
        return MODEL_VAE_CONFIG.get(canonical)
    # Partial substring match against aliases
    for alias, canonical in _MODEL_VAE_ALIASES.items():
        if alias in key:
            return MODEL_VAE_CONFIG.get(canonical)
    # Partial match against canonical names
    for canonical in MODEL_VAE_CONFIG:
        if canonical in key:
            return MODEL_VAE_CONFIG[canonical]
    return None


def get_model_vae_param(model_hint: str, param: str, default=None):
    """
    Convenience: get a single parameter from the unified config.

    Example:
        scale = get_model_vae_param("flux", "scale_factor", default=0.18215)
    """
    cfg = resolve_model_vae_config(model_hint)
    if cfg is None:
        return default
    return cfg.get(param, default)


# ALBABIT-FIX: each preset only carries model_type/weight_dtype/clip_dtype,
# the only fields _apply_preset_override() (nodes_loader.py) reads; hints
# and CLIP slot layout live in js/radiance_loader.js instead. "default"
# dtypes let ComfyUI's own VRAM-aware auto-selection apply. Keys stay
# alphabetical ("Custom" pinned first) to match the dropdown order.
CHECKPOINT_PRESETS: dict = {
    "Custom": {},
    "AuraFlow": {
        "model_type": "aura_flow",
        "weight_dtype": "fp16",
        "clip_dtype": "fp16",
    },
    "Chroma": {
        "model_type": "chroma",
        "weight_dtype": "default",
        "clip_dtype": "default",
    },
    "CogVideoX": {
        "model_type": "cogvideox",
        "weight_dtype": "default",
        "clip_dtype": "default",
    },
    "Cosmos World": {
        "model_type": "cosmos",
        "weight_dtype": "default",
        "clip_dtype": "default",
    },
    # ALBABIT-FIX: Dev and Schnell merged -- both already shared this exact
    # model_type/weight_dtype/clip_dtype, and are architecturally identical
    # (unlike Flux.2 Klein); the Sampler's model_meta mechanism already tells
    # them apart by filename for guidance/steps, so no reason to make the
    # user pick manually here either.
    "Flux.1": {
        "model_type": "flux",
        "weight_dtype": "fp8_e4m3fn",
        "clip_dtype": "fp16",
    },
    "Flux.1 (Low VRAM)": {
        "model_type": "flux",
        "weight_dtype": "fp8_e4m3fn",
        "clip_dtype": "fp8_e4m3fn",
    },
    # ALBABIT-FIX: Dev and Klein merged into one preset -- model/detect.py's
    # Auto-Detect can now tell them apart on its own (single_blocks count),
    # so there's no need for the user to pick the right one manually.
    "Flux.2": {
        "model_type": "Auto-Detect",
        "weight_dtype": "default",
        "clip_dtype": "default",
    },
    # ALBABIT-FIX: clip_dtype forced (unlike "Flux.2"'s "default") -- Dev's
    # Mistral-3 24B encoder is heavy enough that VRAM-constrained users
    # benefit from an explicit push, on top of the offload_mode escape hatch.
    "Flux.2 (Low VRAM)": {
        "model_type": "Auto-Detect",
        "weight_dtype": "default",
        "clip_dtype": "fp8_e4m3fn",
    },
    "HunyuanVideo": {
        "model_type": "hunyuan_video",
        "weight_dtype": "fp8_e4m3fn",
        "clip_dtype": "default",
    },
    # ALBABIT-FIX: 2B and 13B merged into one preset, identical model_type/
    # clip_dtype/vae_hints/clip_hints already. "default" weight_dtype lets
    # comfy.sd auto-pick per the real loaded file's size; the "(Low VRAM)"
    # sibling forces fp8 instead, same pattern as Flux.1/Flux.2/LTX 2.3.
    "LTX Video": {
        "model_type": "ltxv",  # ALBABIT-FIX: "ltx" → "ltxv" — matches sampler_utils.py
        "weight_dtype": "default",
        "clip_dtype": "default",
    },
    "LTX Video (Low VRAM)": {
        "model_type": "ltxv",
        "weight_dtype": "fp8_e4m3fn",
        "clip_dtype": "default",
    },
    "LTX Video 2.3": {
        "model_type": "ltxav",
        # ALBABIT-FIX: ltx-2.3-22b-dev.safetensors is bf16-native; "fp16"
        # forced a bf16->fp16 cast with no VRAM benefit and overflow risk
        # (same issue as the Z-Image clip_dtype fix above).
        "weight_dtype": "default",
        "clip_dtype": "default",
    },
    "LTX Video 2.3 (Low VRAM)": {
        "model_type": "ltxav",
        "weight_dtype": "fp8_e4m3fn",
        "clip_dtype": "fp8_e4m3fn",
    },
    # ALBABIT-FIX: LTX 2.5 reuses "ltxav" -- same 128ch latent format, confirmed
    # from the transformer's patchify_proj.weight shape [4096, 128]. Detection
    # is capability-based (recombine_audio_and_video_latents), not version-string
    # based, so no new model_type is needed. See project_radiance_ltx25 memory.
    "LTX Video 2.5": {
        "model_type": "ltxav",
        "weight_dtype": "default",
        "clip_dtype": "default",
    },
    "LTX Video 2.5 (Low VRAM)": {
        "model_type": "ltxav",
        "weight_dtype": "fp8_e4m3fn",
        "clip_dtype": "fp8_e4m3fn",
    },
    # ALBABIT-FIX: both presets keep weight_dtype/clip_dtype at "default".
    # The size difference is a smaller checkpoint (unet_hints in
    # radiance_loader.js), not a dtype cast; the pruned/nvfp4_awq files use
    # MiniMax's own quantization, already detected natively by comfy.sd.
    "MiniMax H3": {
        "model_type": "minimax",
        "weight_dtype": "default",
        "clip_dtype": "default",
    },
    "MiniMax H3 (Low VRAM)": {
        "model_type": "minimax",
        "weight_dtype": "default",
        "clip_dtype": "default",
    },
    "Lumina2": {
        "model_type": "lumina2",
        "weight_dtype": "fp16",
        "clip_dtype": "default",
    },
    "Mochi": {
        "model_type": "mochi",
        "weight_dtype": "default",
        "clip_dtype": "default",
    },
    "PixArt Sigma": {
        "model_type": "pixart",
        "weight_dtype": "fp16",
        "clip_dtype": "fp16",
    },
    "SD 1.5": {
        "model_type": "sd1.5",
        "weight_dtype": "fp16",
        "clip_dtype": "fp16",
    },
    # ALBABIT-FIX: Large, Large Turbo, and Medium merged into one preset,
    # same model_type/weight_dtype/clip_dtype/vae_hints/clip_hints already.
    # Turbo is Large's distilled variant (Sampler tells it apart by
    # filename); Large vs Medium resolved via a combined unet_hints list.
    "SD3.5": {
        "model_type": "sd3.5",
        "weight_dtype": "fp16",
        "clip_dtype": "fp16",
    },
    # ALBABIT-FIX: Base and Turbo merged -- same reasoning as Flux.1/SD3.5
    # above.
    "SDXL": {
        "model_type": "sdxl",
        "weight_dtype": "fp16",
        "clip_dtype": "fp16",
    },
    # ALBABIT-FIX: weight_dtype now "default" -- 1.3B is small enough that
    # forcing fp8 for every user cost quality for no reason; comfy.sd
    # auto-picks per the real loaded file's param count (same mechanism as
    # LTX Video/Flux.2 above). See "Wan 2.1 (Low VRAM)" for the old forced-fp8
    # behavior, still needed by 14B users on tight VRAM.
    "Wan 2.1": {
        "model_type": "wan",
        "weight_dtype": "default",
        "clip_dtype": "default",
    },
    "Wan 2.1 (Low VRAM)": {
        "model_type": "wan",
        "weight_dtype": "fp8_e4m3fn",
        "clip_dtype": "default",
    },
    # ALBABIT-FIX: separate preset for Wan 2.2, same CLIP slot layout as Wan
    # 2.1 but distinct unet_hints to avoid matching the wrong version when
    # both are installed. weight_dtype "default" for the same reason as
    # "Wan 2.1" above; see "Wan 2.2 (Low VRAM)" for the forced-fp8 sibling.
    "Wan 2.2": {
        "model_type": "wan",
        "weight_dtype": "default",
        "clip_dtype": "default",
    },
    "Wan 2.2 (Low VRAM)": {
        "model_type": "wan",
        "weight_dtype": "fp8_e4m3fn",
        "clip_dtype": "default",
    },
    # ALBABIT-FIX: TI2V-5B is a single-UNET WAN 2.2 variant (no high/low_noise
    # pair), 48ch VAE not 16ch. model_type "wan_ti2v" (was "wan") is a real bug
    # fix: "wan" implies 16ch everywhere downstream, so Resolution built a
    # wrong-shaped, crash-risk latent for this preset before.
    "Wan 2.2 TI2V": {
        "model_type": "wan_ti2v",
        "weight_dtype": "default",
        "clip_dtype": "default",
    },
    "Z-Image": {
        "model_type": "z_image",
        "weight_dtype": "default",
        "clip_dtype": "default",
    },
}

# ALBABIT-FIX: presets for video-generation architectures — used to filter
# the Loader's "preset" dropdown (RadianceVideoLoader shows these,
# RadianceUnifiedLoader shows the rest).
VIDEO_PRESET_NAMES: set = {
    "CogVideoX",
    "Cosmos World",
    "HunyuanVideo",
    "LTX Video",
    "LTX Video (Low VRAM)",
    "LTX Video 2.3",
    "LTX Video 2.3 (Low VRAM)",
    "LTX Video 2.5",
    "LTX Video 2.5 (Low VRAM)",
    "MiniMax H3",
    "MiniMax H3 (Low VRAM)",
    "Mochi",
    "Wan 2.1",
    "Wan 2.1 (Low VRAM)",
    "Wan 2.2",
    "Wan 2.2 (Low VRAM)",
    "Wan 2.2 TI2V",
}

# ALBABIT-FIX: model_type analog of VIDEO_PRESET_NAMES — filters the
# "model_type" dropdown (Custom/Auto-Detect) the same way "preset" is
# filtered. "ltx" renamed to "ltxv" to match sampler_utils.py.
VIDEO_MODEL_TYPES: set = {
    "hunyuan_video", "wan", "wan_ti2v", "ltxv", "ltxav",
    "cosmos", "cogvideox", "mochi", "minimax",
    "hunyuan_video_15", "kandinsky5",
}
