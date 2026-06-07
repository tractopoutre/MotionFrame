## Bundled Assets

### LINE Seed JP

MotionFrame Web bundles `LINESeedJP_A_TTF_Rg.ttf` as its primary UI font.

LINE Seed JP is licensed under the SIL Open Font License 1.1. The full license
text is included at `crates/motionframe-web/assets/fonts/LINESeedJP_OFL.txt`.

## Ported Source Code

The following sections cover external source code incorporated directly into
MotionFrame.

### OpenCV — Farneback Optical Flow

The optical flow implementation in `crates/motionframe-engine/src/flow/` is derived from OpenCV's
Farneback algorithm (`modules/video/src/optflowgf.cpp`).

OpenCV is licensed under the Apache License 2.0:

    Copyright (C) 2000-2024, Intel Corporation, all rights reserved.
    Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
    Copyright (C) 2009-2016, NVIDIA Corporation, all rights reserved.
    Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
    Copyright (C) 2015-2024, OpenCV Foundation, all rights reserved.
    Copyright (C) 2008-2016, Itseez Inc., all rights reserved.
    Copyright (C) 2019-2024, Xperience AI, all rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

Source: https://github.com/opencv/opencv
