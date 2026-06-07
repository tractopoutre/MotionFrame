# MotionFrame

**MotionFrame** is a Rust tool for generating motion-vector textures for
flipbook animation. It analyzes an image sequence, computes optical flow,
accumulates sub-frame motion, and writes color and motion atlases that a
runtime shader can sample for smoother playback with fewer texture frames.

The project includes a desktop application plus a command-line converter.

**Try it in your browser: [motionframe.aki-null.net](https://motionframe.aki-null.net)**

## Sample Renders

| Fade Blend                                                                                         | Motion Blend                                                                                         |
| -------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| ![Explosion Fade](https://github.com/user-attachments/assets/00159a36-f49e-4593-9221-0b4c80ca4113) | ![Explosion Motion](https://github.com/user-attachments/assets/239eac78-90b2-4018-bd08-68a8148c7642) |
| ![Smoke Fade](https://github.com/user-attachments/assets/e20742f8-35bc-403d-8da0-a7d2397d7e89)     | ![Smoke Motion](https://github.com/user-attachments/assets/c9e847f7-06a3-452e-b592-bc7e0b674782)     |

## Key Features

- **GUI Frontend**
- **Batch Conversion CLI**
- **Cross-Platform**
  - Supports macOS, Windows, and Linux.
- **Skipped Frame Analysis**
  - Enhances motion analysis precision with high frame rate input.
  - Analyzes every input frame and accumulates skipped frames into each motion vector frame.
  - Useful for fast movement that is difficult for image-based motion analysis.
- **Motion Vector [Stagger Packing](https://realtimevfx.com/t/flipbook-texture-packing-atlas-super-pack-and-stagger-pack/5609)**
  - Motion vector textures use 2 channels, but this may not be optimal for some platforms.
  - Packing them into 4 channels reduces texture size by half and compresses well with formats like ASTC.
- **Color Atlas Packing**
- **Motion Visualization**
- **Free and Open Source**

See the [reference shader implementation (MIT)](https://github.com/aki-null/UnityFlipbookMotionBlending) for shaders to render this.

## Installation

### Self Contained Binary

[Releases](https://github.com/aki-null/MotionFrame/releases)

### Build from Source

Install the pinned Rust toolchain, then build:

```bash
git clone https://github.com/aki-null/MotionFrame.git
cd MotionFrame
cargo build --release --bin motionframe
```

The desktop binary is written to:

```text
target/release/motionframe
```

## Usage

Launch the desktop application:

```bash
cargo run --release --bin motionframe
```

![Main Window](https://github.com/user-attachments/assets/d44658e8-3a2d-4908-afb1-a22fbbed1fde)

Convert one image sequence or atlas from the command line:

```bash
cargo run --release --bin motionframe -- convert \
  --input frames/explosion \
  --output out/explosion \
  --output-count 64 \
  --layout auto
```

The CLI writes:

- `<prefix>_color_atlas.tga`
- `<prefix>_motion_atlas.tga`
- `<prefix>_meta.json`

Use `motionframe convert --help` for all conversion options.

## Migration from Python MotionFrame

MotionFrame is now a Rust application. The previous Python implementation is
archived on the `legacy-python` branch and the `python-final` tag.

MotionFrame 2.0 is a Rust rewrite of the original Python application.

### What changed

- Desktop application and CLI are built from Rust.
- Optical flow and atlas generation are implemented in the Rust engine.
- Runtime dependencies are no longer installed through Python.
- The previous Python implementation is archived on the `legacy-python` branch and the `python-final` tag.

### Compatibility

Existing image-sequence workflows should map to the Rust CLI and desktop app.
Output quality and performance are expected to improve. If a script depended
on Python internals, migrate it to the `motionframe` CLI.

### Legacy source

Use these refs for the old Python implementation:

- `legacy-python`
- `python-final`
- `v1.0.0-python-final`

## Toolchain

Stable Rust 1.95 is pinned in `rust-toolchain.toml`.

## Verification

```bash
./scripts/verify.sh        # fmt + clippy + test + release build + web build + license check
./scripts/verify-quick.sh  # fmt + clippy + tests
./scripts/verify-parity.sh # flow + atlas pipeline parity tests
```

## License

[GPL v3.0](https://www.gnu.org/licenses/gpl-3.0.txt)

MotionFrame is licensed under GPL v3.0. Third-party notices and license texts
are listed in `THIRD-PARTY-LICENSES.md`.

## Notes

Downloading this tool and using generated textures for your game does not
contaminate your software with GPL v3.0.

## Appendix

Generated atlases:

| Color Atlas                                                                                       | Motion Atlas                                                                                       |
| ------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| ![Explosion Color Atlas](https://github.com/user-attachments/assets/f68db0fe-9348-4ae3-bffa-cb2839ddc5a8) | ![Explosion Motion Atlas](https://github.com/user-attachments/assets/5354ece8-0127-43a0-8b90-a9d358bab4e5) |
| ![Smoke Color Atlas](https://github.com/user-attachments/assets/8bcb4457-e245-4f93-a1aa-4ec88763777f)     | ![Smoke Motion Atlas](https://github.com/user-attachments/assets/66e98695-c152-40ad-9aad-8b098f963d09)     |

## References

- <https://www.klemenlozar.com/frame-blending-with-motion-vectors/>
