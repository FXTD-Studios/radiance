"""Immutable package-level constants."""

PACKAGE_NAME = "radiance"
PACKAGE_DISPLAY_NAME = "Radiance"
VERSION = "4.1.0"
AUTHOR = "Radiance"
WEB_DIRECTORY = "./js"

#: Floor for the number of nodes the entry point is expected to publish.
#:
#: `load_node_mappings` treats a failed optional import as non-fatal so a broken
#: dependency can never stop ComfyUI from starting. The cost is that a group
#: import failure silently removes every node in that group, and the startup
#: banner reported the surviving count as a success -- "successfully loaded 12
#: nodes" reads exactly like "successfully loaded 100 nodes" in a scrolling log.
#: Comparing against this floor turns that into a visible ERROR.
#:
#: Raise it when the catalog grows; a drop below it means something failed to
#: import, not that the catalog shrank.
#:
#: This is the ONLY floor. It used to read 129 against 131 actually
#: registering, while tests/test_package_cleanup.py asserted >= 129 of its own
#: and tests/test_node_smoke.py asserted >= 85 -- three numbers, none of them
#: the truth, and a two-node slack in which a node could stop registering with
#: every count test still green. Both tests read this constant now, so the
#: floor moves in one place and nothing hides underneath it.
#:
#: 2026-09-18: 131 -> 156. Twenty-six nodes the implementation packages
#: declared and no group registered; twenty-five of them ship, and
#: RadianceHDRHistogram is withheld for a 5-D IMAGE output. See
#: nodes/catalog.py and tests/test_node_publication_completeness.py.
#: 2026-09-24: 156 -> 157. RadianceHDRVAEEncode, the encoder HDR VAE Decode
#: inverts, implemented as RadianceVAE4KEncode but never registered.
#: 2026-09-25: 158 -> 157. RadianceLiteViewer removed; the Viewer's Simple
#: mode replaces it and the frontend converts saved Lite Viewers on load.
#: 2026-09-25: 157 -> 156. RadianceVectorMaskDraw (Roto) removed.
#: 4.0: 156 -> 152. The hidden placeholders RadianceSAMModelLoader,
#: RadianceSAMGenerator, RadianceHDRLatentEncoder and RadianceHDRTurboEncoder
#: removed (they only raised since 3.5.0).
#: 4.0: 152 -> 151. RadianceLoadImageMask (Read Mask) removed; ComfyUI's Load
#: Image and its mask editor do the same job.
#: 4.0: 151 -> 152. RadianceSDRToHDRAIFill (RUDRA AI fill v3) added.
#: 4.0: 152 -> 153. RadianceSDRToHDRAIFillRerender (RUDRA AI fill v4, region
#: re-render, experimental) added.
#: 4.0: 153 -> 154. RadianceSDRToHDRAnalytic (RUDRA's analytic route: source
#: curves, clipped-highlight fill, highlight detail, anchor) added.
#: 4.0: 154 -> 157. RadianceRudraRegion, RadianceRudraCalibrate and
#: RadianceRudraRegionEV (RUDRA's luminance region, calibration / reference
#: fit and Region EV) added.
#: 4.0: 157 -> 158. RadianceRudraMasterWrite (the master with RUDRA's mattes,
#: regions picture, EV brackets and provenance sidecar) added.
#: 4.0: 158 -> 162. RadianceRudraPeakFit, RadianceRudraDynamicMetadata,
#: RadianceRudraFlickerReport and RadianceRudraShotHold (RUDRA's delivery
#: steps) added.
#: 4.0: 162 -> 164. RadianceRudraLoadStill (a still's colour tag and EXIF
#: orientation, as RUDRA reads them) and RadianceRudraFillPrompt (RUDRA's AI fill
#: prompt presets and suggestion) added.
EXPECTED_MIN_NODE_COUNT = 164
