//! UI translation catalog. Enum-keyed, compile-time-checked, zero runtime parsing.
//!
//! Adding a new key: add a variant to `Key`, then add an arm for every `Lang`
//! in `t()`. The catalog completeness test asserts every key returns a
//! non-empty string.

use serde::{Deserialize, Serialize};

/// UI language. Persisted under the eframe storage key `"motionframe.locale"`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum Lang {
    #[default]
    En,
    Ja,
}

impl Lang {
    /// Display name shown in the language picker.
    pub const fn display_name(self) -> &'static str {
        match self {
            Self::En => "English",
            Self::Ja => "日本語",
        }
    }

    /// All variants — for the picker combobox and tests.
    pub const fn all() -> &'static [Self] {
        &[Self::En, Self::Ja]
    }
}

/// Catalog key. One variant per user-visible static string in the UI.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Key {
    // Sidebar — Input group
    InputHeading,
    InputTilesPerRow,
    InputTilesPerColumn,
    InputTilesHover,

    // Sidebar — Atlas group
    AtlasHeading,
    AtlasResolution,
    AtlasResolutionHover,
    Columns,
    Rows,
    LockBadgeAuto,
    LockBadgeManual,
    TilePixelWidth,
    TilePixelWidthHover,
    MaxTextureDim,
    MaxTextureDimHover,
    ResizeAlgorithm,
    ResizeAlgorithmHover,
    ResizeCubic,
    ResizeLinear,
    ResizeLanczos,
    ResizeNearest,
    Extrude,
    ExtrudeHover,
    OutputFrames,
    OutputFramesHover,
    TrimTailForExactOutputCount,
    TrimTailForExactOutputCountHover,
    OutputFrameSummary,
    FrameSkipSummary,
    IgnoredTailFramesSummary,
    WastedTilesSummary,
    LoadMoreFramesHint,
    ExceedsTextureLimit,

    // Sidebar — Motion group
    MotionHeading,
    Encoding,
    EncodingR8G8Remap01,
    EncodingSidefxLabsR8G8,
    EncodingHover,
    HalveMotionVector,
    HalveMotionVectorHover,
    StaggerPack,
    StaggerPackHover,
    TemporalSmoothing,
    TemporalSmoothingHover,
    AnalyzeSkippedFrames,
    AnalyzeSkippedFramesHover,
    LoopOption,
    LoopOptionHover,

    // Sidebar — Color group
    ColorHeading,
    PremultipliedAlpha,
    PremultipliedAlphaHover,

    // Predicted dims readouts
    ColorAtlasDims,
    MotionAtlasDims,

    // Sidebar — Language picker
    LanguageLabel,
    LanguageDisabledTooltip,

    // Tabs
    TabColor,
    TabMotion,
    TabVisualization,
    TabPreview,

    // Empty / ready states
    DropFrameOrFolder,
    AcceptedFormats,
    ClickGenerate,

    // Image preview
    Zoom,

    // Playback
    Frame,
    Fps,
    Strength,
    Background,
    BgBlack,
    BgGray,
    BgWhite,
    BgChecker,
    Blend,
    BlendMotionVector,
    BlendMotionVectorHover,
    BlendCrossFade,
    BlendCrossFadeHover,
    Play,
    Pause,

    // App shell
    DropHereOrBrowse,
    BrowseEllipsis,
    BrowseDots,
    FramesCount,
    Licenses,
    LicensesWindowTitle,
    SaveOutputsDialogTitle,
    Generate,
    Cancel,
    Save,
    LoadingProgress,
    ErrCouldNotDecode,
    ErrCouldNotDecodeWith,
    ErrMetadataSerialize,
    ErrWorkerPanic,
    ErrSaveOutputs,
    ClearSelection,

    // Sidebar — Output group
    OutputHeading,
    OutputFormat,
    OutputFormatHover,
    OutputBaseName,
    OutputBaseNameHover,
    ColorSuffix,
    ColorSuffixHover,
    MotionSuffix,
    MotionSuffixHover,
    MetaSuffix,
    MetaSuffixHover,
    OutputPreview,
    OutputPreviewEmpty,
    // Sidebar — Frame range
    FrameRangeHeading,
    StartFrame,
    StartFrameHover,
    EndFrame,
    EndFrameHover,
}

/// Translate `key` to the user-visible string for `lang`.
///
/// Format strings carry positional placeholders `{0}`, `{1}`, …; substitute
/// values via [`fmt`] at the call site.
// match_same_arms: distinct (Lang, Key) pairs that happen to share text
//   ("Motion" / "Color" appear as both group heading and tab label). Merging
//   would couple unrelated keys and break independent translation later.
// too_many_lines: the catalog is intrinsically long; splitting into helpers
//   only obscures the parallel-column structure.
#[allow(clippy::match_same_arms, clippy::too_many_lines)]
pub const fn t(lang: Lang, key: Key) -> &'static str {
    match (lang, key) {
        // Sidebar — Input
        (Lang::En, Key::InputHeading) => "Input",
        (Lang::En, Key::InputTilesPerRow) => "Input tiles per row:",
        (Lang::En, Key::InputTilesPerColumn) => "Input tiles per column:",
        (Lang::En, Key::InputTilesHover) => {
            "How the dropped image is sliced before motion analysis. \
             Auto-detected at drop; override here."
        }

        // Sidebar — Atlas
        (Lang::En, Key::AtlasHeading) => "Atlas",
        (Lang::En, Key::AtlasResolution) => "Atlas resolution:",
        (Lang::En, Key::AtlasResolutionHover) => {
            "Target total atlas pixel dimension. Tile size is auto-computed from this and the input aspect ratio."
        }
        (Lang::En, Key::Columns) => "Columns:",
        (Lang::En, Key::Rows) => "Rows:",
        (Lang::En, Key::LockBadgeAuto) => "(auto)",
        (Lang::En, Key::LockBadgeManual) => "(manual)",
        (Lang::En, Key::TilePixelWidth) => "Tile pixel width:",
        (Lang::En, Key::TilePixelWidthHover) => "Pixel width of each output tile.",
        (Lang::En, Key::MaxTextureDim) => "Max texture dim:",
        (Lang::En, Key::MaxTextureDimHover) => {
            "Per-axis pixel cap for both color and motion atlases. \
             Lower this when targeting GPUs with smaller texture limits."
        }
        (Lang::En, Key::ResizeAlgorithm) => "Resize algorithm:",
        (Lang::En, Key::ResizeAlgorithmHover) => "Resampling algorithm for atlas blits.",
        (Lang::En, Key::ResizeCubic) => "Cubic",
        (Lang::En, Key::ResizeLinear) => "Linear",
        (Lang::En, Key::ResizeLanczos) => "Lanczos",
        (Lang::En, Key::ResizeNearest) => "Nearest",
        (Lang::En, Key::Extrude) => "Extrude:",
        (Lang::En, Key::ExtrudeHover) => {
            "Edge-replicate padding around each atlas tile. Use when sampling \
             near tile boundaries with linear filtering."
        }
        (Lang::En, Key::OutputFrames) => "Output frames:",
        (Lang::En, Key::OutputFramesHover) => {
            "Number of frames to output. Clamped to the available frame count and atlas slots."
        }
        (Lang::En, Key::TrimTailForExactOutputCount) => "Trim tail for exact output count",
        (Lang::En, Key::TrimTailForExactOutputCountHover) => {
            "Allows exact output frame counts by ignoring ending input frames. Source files are not changed."
        }
        (Lang::En, Key::OutputFrameSummary) => "{0} output frames",
        (Lang::En, Key::FrameSkipSummary) => "skip {0}",
        (Lang::En, Key::IgnoredTailFramesSummary) => "ignores last {0}",
        (Lang::En, Key::WastedTilesSummary) => "{0} wasted tiles",
        (Lang::En, Key::LoadMoreFramesHint) => "(load 3+ frames to choose output count)",
        (Lang::En, Key::ExceedsTextureLimit) => {
            "Exceeds {0}px GPU texture limit. \
             Reduce tile pixel width, raise max texture dim, or skip more frames."
        }

        // Sidebar — Motion
        (Lang::En, Key::MotionHeading) => "Motion",
        (Lang::En, Key::Encoding) => "Encoding:",
        (Lang::En, Key::EncodingR8G8Remap01) => "R8G8 Remap [0,1]",
        (Lang::En, Key::EncodingSidefxLabsR8G8) => "SideFX Labs R8G8",
        (Lang::En, Key::EncodingHover) => {
            "How to pack the motion vector into 2 bytes. R8G8 Remap is straightforward \
             symmetric. SideFX Labs uses polar encoding with a flip bit."
        }
        (Lang::En, Key::HalveMotionVector) => "Halve motion vector",
        (Lang::En, Key::HalveMotionVectorHover) => {
            "Output motion vector at half the color atlas width. \
             Halves texture memory; mild quality loss."
        }
        (Lang::En, Key::StaggerPack) => "Stagger pack",
        (Lang::En, Key::StaggerPackHover) => {
            "Pack two adjacent motion frames into one RGBA tile, halving texture height."
        }
        (Lang::En, Key::TemporalSmoothing) => "Temporal smoothing:",
        (Lang::En, Key::TemporalSmoothingHover) => {
            "Filter motion vectors across output frames to reduce visible \
             stepping at frame transitions. 0 = off (exports unchanged); \
             1 = full 3-tap binomial filter. Baked into the atlas."
        }
        (Lang::En, Key::AnalyzeSkippedFrames) => "Analyze skipped frames",
        (Lang::En, Key::AnalyzeSkippedFramesHover) => {
            "Use intermediate input frames to compute each motion vector. \
             Slower but much more accurate on fast motion."
        }
        (Lang::En, Key::LoopOption) => "Loop",
        (Lang::En, Key::LoopOptionHover) => {
            "Treat the sequence as looping. The final motion vector wraps from frame N−1 to frame 0."
        }

        // Sidebar — Color
        (Lang::En, Key::ColorHeading) => "Color",
        (Lang::En, Key::PremultipliedAlpha) => "Premultiplied alpha",
        (Lang::En, Key::PremultipliedAlphaHover) => {
            "Store the color atlas with premultiplied alpha (rgb already scaled by alpha). \
             Default off — most engines expect straight (non-premultiplied) RGBA."
        }

        // Predicted dims readouts
        (Lang::En, Key::ColorAtlasDims) => "Color atlas: {0} × {1} px",
        (Lang::En, Key::MotionAtlasDims) => "Motion atlas: {0} × {1} px",

        // Sidebar — Language picker
        (Lang::En, Key::LanguageLabel) => "Language:",
        (Lang::En, Key::LanguageDisabledTooltip) => {
            "No Japanese font found. Install Noto CJK or fonts-japanese-gothic."
        }

        // Tabs
        (Lang::En, Key::TabColor) => "Color",
        (Lang::En, Key::TabMotion) => "Motion",
        (Lang::En, Key::TabVisualization) => "Visualization",
        (Lang::En, Key::TabPreview) => "Preview",

        // Empty / ready states
        (Lang::En, Key::DropFrameOrFolder) => "Drop a frame or folder here",
        (Lang::En, Key::AcceptedFormats) => "Accepted: jpg, jpeg, png, bmp, tiff, tga",
        (Lang::En, Key::ClickGenerate) => "Click Generate to produce output",

        // Image preview
        (Lang::En, Key::Zoom) => "Zoom",

        // Playback
        (Lang::En, Key::Frame) => "Frame:",
        (Lang::En, Key::Fps) => "FPS:",
        (Lang::En, Key::Strength) => "Strength: {0}",
        (Lang::En, Key::Background) => "Background:",
        (Lang::En, Key::BgBlack) => "Black",
        (Lang::En, Key::BgGray) => "Gray",
        (Lang::En, Key::BgWhite) => "White",
        (Lang::En, Key::BgChecker) => "Checker",
        (Lang::En, Key::Blend) => "Blend:",
        (Lang::En, Key::BlendMotionVector) => "Motion vector",
        (Lang::En, Key::BlendMotionVectorHover) => {
            "Warp UVs by the encoded motion vectors, then mix."
        }
        (Lang::En, Key::BlendCrossFade) => "Cross-fade",
        (Lang::En, Key::BlendCrossFadeHover) => {
            "Plain mix(c0, c1, t) — no warp. A/B baseline for what motion vectors add."
        }
        (Lang::En, Key::Play) => "Play",
        (Lang::En, Key::Pause) => "Pause",

        // App shell
        (Lang::En, Key::DropHereOrBrowse) => "Drop here or browse",
        (Lang::En, Key::BrowseEllipsis) => "Browse…",
        (Lang::En, Key::BrowseDots) => "Browse...",
        (Lang::En, Key::FramesCount) => "{0} frames",
        (Lang::En, Key::Licenses) => "Licenses",
        (Lang::En, Key::LicensesWindowTitle) => "Third-Party Licenses",
        (Lang::En, Key::SaveOutputsDialogTitle) => "Save outputs as…",
        (Lang::En, Key::Generate) => "Generate",
        (Lang::En, Key::Cancel) => "Cancel",
        (Lang::En, Key::Save) => "Save",
        (Lang::En, Key::LoadingProgress) => "Loading {0}/{1}",
        (Lang::En, Key::ErrCouldNotDecode) => "Could not decode '{0}'",
        (Lang::En, Key::ErrCouldNotDecodeWith) => "Could not decode '{0}': {1}",
        (Lang::En, Key::ErrMetadataSerialize) => "Metadata serialize failed: {0}",
        (Lang::En, Key::ErrWorkerPanic) => "worker panic: {0}",
        (Lang::En, Key::ErrSaveOutputs) => "Save failed: {0}",
        (Lang::En, Key::ClearSelection) => "×",

        // Sidebar — Output
        (Lang::En, Key::OutputHeading) => "Output",
        (Lang::En, Key::OutputFormat) => "Format",
        (Lang::En, Key::OutputFormatHover) => {
            "Use [basename], [cols], [rows], [suffix], [ext] tokens"
        }
        (Lang::En, Key::OutputBaseName) => "Base name",
        (Lang::En, Key::OutputBaseNameHover) => {
            "Override the [basename] token (empty = auto-detect)"
        }
        (Lang::En, Key::ColorSuffix) => "Color suffix",
        (Lang::En, Key::ColorSuffixHover) => {
            "What [suffix] resolves to in the color atlas filename"
        }
        (Lang::En, Key::MotionSuffix) => "Motion suffix",
        (Lang::En, Key::MotionSuffixHover) => {
            "What [suffix] resolves to in the motion atlas filename"
        }
        (Lang::En, Key::MetaSuffix) => "Meta suffix",
        (Lang::En, Key::MetaSuffixHover) => {
            "What [suffix] resolves to in the metadata filename"
        }
        (Lang::En, Key::OutputPreview) => "Preview",
        (Lang::En, Key::OutputPreviewEmpty) => "Using default format",

        // Sidebar — Frame range
        (Lang::En, Key::FrameRangeHeading) => "Frame Range",
        (Lang::En, Key::StartFrame) => "Start frame",
        (Lang::En, Key::StartFrameHover) => "First frame to process (0-based)",
        (Lang::En, Key::EndFrame) => "End frame",
        (Lang::En, Key::EndFrameHover) => {
            "Last frame to process (0-based, exclusive; 0 = all)"
        }

        // Sidebar — Input
        (Lang::Ja, Key::InputHeading) => "入力",
        (Lang::Ja, Key::InputTilesPerRow) => "入力タイル数(横):",
        (Lang::Ja, Key::InputTilesPerColumn) => "入力タイル数(縦):",
        (Lang::Ja, Key::InputTilesHover) => {
            "ドロップした画像をモーション解析前にどう分割するか。\
             ドロップ時に自動検出されますが、ここで上書きできます。"
        }

        // Sidebar — Atlas
        (Lang::Ja, Key::AtlasHeading) => "アトラス",
        (Lang::Ja, Key::AtlasResolution) => "アトラス解像度:",
        (Lang::Ja, Key::AtlasResolutionHover) => {
            "目標アトラス全体のピクセル寸法。タイルサイズはこれと入力アスペクト比から自動計算されます。"
        }
        (Lang::Ja, Key::Columns) => "列:",
        (Lang::Ja, Key::Rows) => "行:",
        (Lang::Ja, Key::LockBadgeAuto) => "(自動)",
        (Lang::Ja, Key::LockBadgeManual) => "(手動)",
        (Lang::Ja, Key::TilePixelWidth) => "タイルのピクセル幅:",
        (Lang::Ja, Key::TilePixelWidthHover) => "出力タイル1枚あたりのピクセル幅。",
        (Lang::Ja, Key::MaxTextureDim) => "最大テクスチャサイズ:",
        (Lang::Ja, Key::MaxTextureDimHover) => {
            "カラー/モーション両アトラスの軸ごとのピクセル上限。\
             テクスチャ上限が小さいGPUを対象とする場合は値を下げてください。"
        }
        (Lang::Ja, Key::ResizeAlgorithm) => "リサイズアルゴリズム:",
        (Lang::Ja, Key::ResizeAlgorithmHover) => "アトラス転送時のリサンプリングアルゴリズム。",
        (Lang::Ja, Key::ResizeCubic) => "Cubic",
        (Lang::Ja, Key::ResizeLinear) => "Linear",
        (Lang::Ja, Key::ResizeLanczos) => "Lanczos",
        (Lang::Ja, Key::ResizeNearest) => "Nearest",
        (Lang::Ja, Key::Extrude) => "エッジパディング:",
        (Lang::Ja, Key::ExtrudeHover) => {
            "各アトラスタイル周囲の端ピクセルを複製するパディング。\
             リニアフィルタでタイル境界付近をサンプルする場合に使用します。"
        }
        (Lang::Ja, Key::OutputFrames) => "出力フレーム数:",
        (Lang::Ja, Key::OutputFramesHover) => {
            "出力するフレーム数。利用可能なフレーム数とアトラススロットに制限されます。"
        }
        (Lang::Ja, Key::TrimTailForExactOutputCount) => "末尾を無視して出力数を正確に合わせる",
        (Lang::Ja, Key::TrimTailForExactOutputCountHover) => {
            "末尾の入力フレームを無視して、出力フレーム数を正確に合わせます。元ファイルは変更されません。"
        }
        (Lang::Ja, Key::OutputFrameSummary) => "出力 {0} フレーム",
        (Lang::Ja, Key::FrameSkipSummary) => "スキップ {0}",
        (Lang::Ja, Key::IgnoredTailFramesSummary) => "末尾 {0} フレームを無視",
        (Lang::Ja, Key::WastedTilesSummary) => "未使用タイル {0}",
        (Lang::Ja, Key::LoadMoreFramesHint) => "(出力数を選ぶには3枚以上のフレームを読み込んでください)",
        (Lang::Ja, Key::ExceedsTextureLimit) => {
            "GPUのテクスチャ上限 {0}px を超えました。\
             タイルのピクセル幅を下げるか、最大テクスチャサイズを上げるか、\
             フレームスキップを増やしてください。"
        }

        // Sidebar — Motion
        (Lang::Ja, Key::MotionHeading) => "モーション",
        (Lang::Ja, Key::Encoding) => "エンコード方式:",
        (Lang::Ja, Key::EncodingR8G8Remap01) => "R8G8 Remap [0,1]",
        (Lang::Ja, Key::EncodingSidefxLabsR8G8) => "SideFX Labs R8G8",
        (Lang::Ja, Key::EncodingHover) => {
            "モーションベクトルを2バイトにどうパックするか。\
             R8G8 Remap はシンプルな対称エンコード。\
             SideFX Labs は反転ビット付きの極座標エンコードを使用します。"
        }
        (Lang::Ja, Key::HalveMotionVector) => "モーションベクトルを半分にする",
        (Lang::Ja, Key::HalveMotionVectorHover) => {
            "モーションベクトルをカラーアトラスの半分の幅で出力します。\
             テクスチャメモリは半減しますが、品質は若干低下します。"
        }
        (Lang::Ja, Key::StaggerPack) => "Stagger Pack",
        (Lang::Ja, Key::StaggerPackHover) => {
            "隣接する2フレームのモーションを1つのRGBAタイルにパックし、\
             テクスチャの高さを半減します。"
        }
        (Lang::Ja, Key::TemporalSmoothing) => "時間方向スムージング:",
        (Lang::Ja, Key::TemporalSmoothingHover) => {
            "出力フレーム間でモーションベクトルをフィルタし、\
             フレーム遷移での目に見えるカクつきを軽減します。\
             0=無効(変更なしで出力)、1=3タップ二項フィルタ。アトラスに焼き込まれます。"
        }
        (Lang::Ja, Key::AnalyzeSkippedFrames) => "スキップフレームも解析",
        (Lang::Ja, Key::AnalyzeSkippedFramesHover) => {
            "中間入力フレームも使ってモーションベクトルを計算します。\
             低速ですが、速い動きでより正確になります。"
        }
        (Lang::Ja, Key::LoopOption) => "ループ",
        (Lang::Ja, Key::LoopOptionHover) => {
            "シーケンスをループとして扱います。最終モーションベクトルはフレーム N−1 から 0 へ折り返します。"
        }

        // Sidebar — Color
        (Lang::Ja, Key::ColorHeading) => "カラー",
        (Lang::Ja, Key::PremultipliedAlpha) => "乗算済みアルファ",
        (Lang::Ja, Key::PremultipliedAlphaHover) => {
            "カラーアトラスを乗算済みアルファ(rgbにあらかじめalphaを掛けた状態)で保存します。\
             デフォルトは無効 — 多くのエンジンはストレートアルファのRGBAを想定しています。"
        }

        // Predicted dims readouts
        (Lang::Ja, Key::ColorAtlasDims) => "カラーアトラス: {0} × {1} px",
        (Lang::Ja, Key::MotionAtlasDims) => "モーションアトラス: {0} × {1} px",

        // Sidebar — Language picker
        (Lang::Ja, Key::LanguageLabel) => "言語:",
        (Lang::Ja, Key::LanguageDisabledTooltip) => {
            "日本語フォントが見つかりません。Noto CJK または fonts-japanese-gothic をインストールしてください。"
        }

        // Tabs
        (Lang::Ja, Key::TabColor) => "カラー",
        (Lang::Ja, Key::TabMotion) => "モーション",
        (Lang::Ja, Key::TabVisualization) => "可視化",
        (Lang::Ja, Key::TabPreview) => "プレビュー",

        // Empty / ready states
        (Lang::Ja, Key::DropFrameOrFolder) => "フレームまたはフォルダをここにドロップ",
        (Lang::Ja, Key::AcceptedFormats) => "対応形式: jpg, jpeg, png, bmp, tiff, tga",
        (Lang::Ja, Key::ClickGenerate) => "「生成」を押して出力を作成",

        // Image preview
        (Lang::Ja, Key::Zoom) => "ズーム",

        // Playback
        (Lang::Ja, Key::Frame) => "フレーム:",
        (Lang::Ja, Key::Fps) => "FPS:",
        (Lang::Ja, Key::Strength) => "強度: {0}",
        (Lang::Ja, Key::Background) => "背景:",
        (Lang::Ja, Key::BgBlack) => "黒",
        (Lang::Ja, Key::BgGray) => "グレー",
        (Lang::Ja, Key::BgWhite) => "白",
        (Lang::Ja, Key::BgChecker) => "チェッカー",
        (Lang::Ja, Key::Blend) => "ブレンド:",
        (Lang::Ja, Key::BlendMotionVector) => "モーションベクトル",
        (Lang::Ja, Key::BlendMotionVectorHover) => {
            "エンコードされたモーションベクトルでUVをワープしてからミックスします。"
        }
        (Lang::Ja, Key::BlendCrossFade) => "クロスフェード",
        (Lang::Ja, Key::BlendCrossFadeHover) => {
            "単純な mix(c0, c1, t) — ワープなし。モーションベクトルの効果を比較するためのA/Bベースライン。"
        }
        (Lang::Ja, Key::Play) => "再生",
        (Lang::Ja, Key::Pause) => "一時停止",

        // App shell
        (Lang::Ja, Key::DropHereOrBrowse) => "ここにドロップまたは参照",
        (Lang::Ja, Key::BrowseEllipsis) => "参照…",
        (Lang::Ja, Key::BrowseDots) => "参照...",
        (Lang::Ja, Key::FramesCount) => "{0} フレーム",
        (Lang::Ja, Key::Licenses) => "ライセンス",
        (Lang::Ja, Key::LicensesWindowTitle) => "サードパーティライセンス",
        (Lang::Ja, Key::SaveOutputsDialogTitle) => "出力の保存先…",
        (Lang::Ja, Key::Generate) => "生成",
        (Lang::Ja, Key::Cancel) => "キャンセル",
        (Lang::Ja, Key::Save) => "保存",
        (Lang::Ja, Key::LoadingProgress) => "読み込み中 {0}/{1}",
        (Lang::Ja, Key::ErrCouldNotDecode) => "'{0}' をデコードできませんでした",
        (Lang::Ja, Key::ErrCouldNotDecodeWith) => "'{0}' をデコードできませんでした: {1}",
        (Lang::Ja, Key::ErrMetadataSerialize) => "メタデータのシリアライズに失敗しました: {0}",
        (Lang::Ja, Key::ErrWorkerPanic) => "ワーカースレッドが異常終了しました: {0}",
        (Lang::Ja, Key::ErrSaveOutputs) => "保存に失敗しました: {0}",
        (Lang::Ja, Key::ClearSelection) => "×",

        // Sidebar — Output
        (Lang::Ja, Key::OutputHeading) => "出力",
        (Lang::Ja, Key::OutputFormat) => "フォーマット",
        (Lang::Ja, Key::OutputFormatHover) => {
            "[basename], [cols], [rows], [suffix], [ext] トークンを使用"
        }
        (Lang::Ja, Key::OutputBaseName) => "ベース名",
        (Lang::Ja, Key::OutputBaseNameHover) => {
            "[basename] トークンの上書き（空=自動検出）"
        }
        (Lang::Ja, Key::ColorSuffix) => "カラーサフィックス",
        (Lang::Ja, Key::ColorSuffixHover) => {
            "カラーアトラスファイル名の [suffix] 値"
        }
        (Lang::Ja, Key::MotionSuffix) => "モーションサフィックス",
        (Lang::Ja, Key::MotionSuffixHover) => {
            "モーションアトラスファイル名の [suffix] 値"
        }
        (Lang::Ja, Key::MetaSuffix) => "メタサフィックス",
        (Lang::Ja, Key::MetaSuffixHover) => {
            "メタデータファイル名の [suffix] 値"
        }
        (Lang::Ja, Key::OutputPreview) => "プレビュー",
        (Lang::Ja, Key::OutputPreviewEmpty) => "デフォルトフォーマットを使用中",

        // Sidebar — Frame range
        (Lang::Ja, Key::FrameRangeHeading) => "フレーム範囲",
        (Lang::Ja, Key::StartFrame) => "開始フレーム",
        (Lang::Ja, Key::StartFrameHover) => "処理する最初のフレーム（0始まり）",
        (Lang::Ja, Key::EndFrame) => "終了フレーム",
        (Lang::Ja, Key::EndFrameHover) => {
            "処理する最後のフレーム（0始まり、排他。0=すべて）"
        }
    }
}

/// Substitute positional placeholders `{0}`, `{1}`, … in a translated
/// template. Order matches the values slice. Missing placeholders are left
/// in place; extra values are ignored.
pub fn fmt(template: &str, values: &[&dyn std::fmt::Display]) -> String {
    let mut out = template.to_string();
    for (i, v) in values.iter().enumerate() {
        out = out.replace(&format!("{{{i}}}"), &v.to_string());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every (Lang, Key) pair must return a non-empty string.
    #[test]
    fn i18n_catalog_complete() {
        for &lang in Lang::all() {
            for &key in ALL_KEYS {
                let s = t(lang, key);
                assert!(!s.is_empty(), "empty translation for ({lang:?}, {key:?})");
            }
        }
    }

    #[test]
    fn fmt_substitutes_positional_placeholders() {
        assert_eq!(fmt("{0} frames", &[&7]), "7 frames");
        assert_eq!(fmt("atlas {0}×{1}", &[&1024, &768]), "atlas 1024×768");
        // Missing placeholder: leaves it alone.
        assert_eq!(fmt("hello {0} {1}", &[&"world"]), "hello world {1}");
        // No placeholders: identity.
        assert_eq!(fmt("static", &[]), "static");
    }

    /// Single source of truth for "every Key variant" used by the test
    /// above. The compile-time guard below catches a `Key` variant added
    /// without being listed here.
    const ALL_KEYS: &[Key] = &[
        Key::InputHeading,
        Key::InputTilesPerRow,
        Key::InputTilesPerColumn,
        Key::InputTilesHover,
        Key::AtlasHeading,
        Key::AtlasResolution,
        Key::AtlasResolutionHover,
        Key::Columns,
        Key::Rows,
        Key::LockBadgeAuto,
        Key::LockBadgeManual,
        Key::TilePixelWidth,
        Key::TilePixelWidthHover,
        Key::MaxTextureDim,
        Key::MaxTextureDimHover,
        Key::ResizeAlgorithm,
        Key::ResizeAlgorithmHover,
        Key::ResizeCubic,
        Key::ResizeLinear,
        Key::ResizeLanczos,
        Key::ResizeNearest,
        Key::Extrude,
        Key::ExtrudeHover,
        Key::OutputFrames,
        Key::OutputFramesHover,
        Key::TrimTailForExactOutputCount,
        Key::TrimTailForExactOutputCountHover,
        Key::OutputFrameSummary,
        Key::FrameSkipSummary,
        Key::IgnoredTailFramesSummary,
        Key::WastedTilesSummary,
        Key::LoadMoreFramesHint,
        Key::ExceedsTextureLimit,
        Key::MotionHeading,
        Key::Encoding,
        Key::EncodingR8G8Remap01,
        Key::EncodingSidefxLabsR8G8,
        Key::EncodingHover,
        Key::HalveMotionVector,
        Key::HalveMotionVectorHover,
        Key::StaggerPack,
        Key::StaggerPackHover,
        Key::TemporalSmoothing,
        Key::TemporalSmoothingHover,
        Key::AnalyzeSkippedFrames,
        Key::AnalyzeSkippedFramesHover,
        Key::LoopOption,
        Key::LoopOptionHover,
        Key::ColorHeading,
        Key::PremultipliedAlpha,
        Key::PremultipliedAlphaHover,
        Key::ColorAtlasDims,
        Key::MotionAtlasDims,
        Key::LanguageLabel,
        Key::LanguageDisabledTooltip,
        Key::TabColor,
        Key::TabMotion,
        Key::TabVisualization,
        Key::TabPreview,
        Key::DropFrameOrFolder,
        Key::AcceptedFormats,
        Key::ClickGenerate,
        Key::Zoom,
        Key::Frame,
        Key::Fps,
        Key::Strength,
        Key::Background,
        Key::BgBlack,
        Key::BgGray,
        Key::BgWhite,
        Key::BgChecker,
        Key::Blend,
        Key::BlendMotionVector,
        Key::BlendMotionVectorHover,
        Key::BlendCrossFade,
        Key::BlendCrossFadeHover,
        Key::Play,
        Key::Pause,
        Key::DropHereOrBrowse,
        Key::BrowseEllipsis,
        Key::BrowseDots,
        Key::FramesCount,
        Key::Licenses,
        Key::LicensesWindowTitle,
        Key::SaveOutputsDialogTitle,
        Key::Generate,
        Key::Cancel,
        Key::Save,
        Key::LoadingProgress,
        Key::ErrCouldNotDecode,
        Key::ErrCouldNotDecodeWith,
        Key::ErrMetadataSerialize,
        Key::ErrWorkerPanic,
        Key::ErrSaveOutputs,
        Key::ClearSelection,
        Key::OutputHeading,
        Key::OutputFormat,
        Key::OutputFormatHover,
        Key::OutputBaseName,
        Key::OutputBaseNameHover,
        Key::ColorSuffix,
        Key::ColorSuffixHover,
        Key::MotionSuffix,
        Key::MotionSuffixHover,
        Key::MetaSuffix,
        Key::MetaSuffixHover,
        Key::OutputPreview,
        Key::OutputPreviewEmpty,
        Key::FrameRangeHeading,
        Key::StartFrame,
        Key::StartFrameHover,
        Key::EndFrame,
        Key::EndFrameHover,
    ];

    /// Compile-time guard: adding a `Key` variant without updating this
    /// match (and `ALL_KEYS` above) fails the build.
    #[allow(dead_code)]
    fn match_all_keys_compiles(k: Key) {
        match k {
            Key::InputHeading
            | Key::InputTilesPerRow
            | Key::InputTilesPerColumn
            | Key::InputTilesHover
            |             Key::AtlasHeading
            | Key::AtlasResolution
            | Key::AtlasResolutionHover
            | Key::Columns
            | Key::Rows
            | Key::LockBadgeAuto
            | Key::LockBadgeManual
            | Key::TilePixelWidth
            | Key::TilePixelWidthHover
            | Key::MaxTextureDim
            | Key::MaxTextureDimHover
            | Key::ResizeAlgorithm
            | Key::ResizeAlgorithmHover
            | Key::ResizeCubic
            | Key::ResizeLinear
            | Key::ResizeLanczos
            | Key::ResizeNearest
            | Key::Extrude
            | Key::ExtrudeHover
            | Key::OutputFrames
            | Key::OutputFramesHover
            | Key::TrimTailForExactOutputCount
            | Key::TrimTailForExactOutputCountHover
            | Key::OutputFrameSummary
            | Key::FrameSkipSummary
            | Key::IgnoredTailFramesSummary
            | Key::WastedTilesSummary
            | Key::LoadMoreFramesHint
            | Key::ExceedsTextureLimit
            | Key::MotionHeading
            | Key::Encoding
            | Key::EncodingR8G8Remap01
            | Key::EncodingSidefxLabsR8G8
            | Key::EncodingHover
            | Key::HalveMotionVector
            | Key::HalveMotionVectorHover
            | Key::StaggerPack
            | Key::StaggerPackHover
            | Key::TemporalSmoothing
            | Key::TemporalSmoothingHover
            | Key::AnalyzeSkippedFrames
            | Key::AnalyzeSkippedFramesHover
            | Key::LoopOption
            | Key::LoopOptionHover
            | Key::ColorHeading
            | Key::PremultipliedAlpha
            | Key::PremultipliedAlphaHover
            | Key::ColorAtlasDims
            | Key::MotionAtlasDims
            | Key::LanguageLabel
            | Key::LanguageDisabledTooltip
            | Key::TabColor
            | Key::TabMotion
            | Key::TabVisualization
            | Key::TabPreview
            | Key::DropFrameOrFolder
            | Key::AcceptedFormats
            | Key::ClickGenerate
            | Key::Zoom
            | Key::Frame
            | Key::Fps
            | Key::Strength
            | Key::Background
            | Key::BgBlack
            | Key::BgGray
            | Key::BgWhite
            | Key::BgChecker
            | Key::Blend
            | Key::BlendMotionVector
            | Key::BlendMotionVectorHover
            | Key::BlendCrossFade
            | Key::BlendCrossFadeHover
            | Key::Play
            | Key::Pause
            | Key::DropHereOrBrowse
            | Key::BrowseEllipsis
            | Key::BrowseDots
            | Key::FramesCount
            | Key::Licenses
            | Key::LicensesWindowTitle
            | Key::SaveOutputsDialogTitle
            | Key::Generate
            | Key::Cancel
            | Key::Save
            | Key::LoadingProgress
            | Key::ErrCouldNotDecode
            | Key::ErrCouldNotDecodeWith
            | Key::ErrMetadataSerialize
            | Key::ErrWorkerPanic
            | Key::ErrSaveOutputs
            | Key::ClearSelection
            | Key::OutputHeading
            | Key::OutputFormat
            | Key::OutputFormatHover
            | Key::OutputBaseName
            | Key::OutputBaseNameHover
            | Key::ColorSuffix
            | Key::ColorSuffixHover
            | Key::MotionSuffix
            | Key::MotionSuffixHover
            | Key::MetaSuffix
            | Key::MetaSuffixHover
            | Key::OutputPreview
            | Key::OutputPreviewEmpty
            | Key::FrameRangeHeading
            | Key::StartFrame
            | Key::StartFrameHover
            | Key::EndFrame
            | Key::EndFrameHover => {}
        }
    }
}
