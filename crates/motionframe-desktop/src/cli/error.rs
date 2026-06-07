use std::fmt;

/// User-facing CLI error with exit-code classification.
///
/// Exit codes: `2` for argument/config errors, `1` for I/O or pipeline failures.
#[derive(Debug)]
pub enum CliError {
    /// Missing or invalid command-line argument.
    Argument(String),
    /// TOML config parsing or validation error.
    Config(String),
    /// Filesystem I/O error.
    Io(std::io::Error),
    /// Pipeline processing error.
    Pipeline(String),
    /// GUI initialization error.
    Gui(String),
}

impl CliError {
    pub const fn exit_code(&self) -> i32 {
        match self {
            Self::Argument(_) | Self::Config(_) => 2,
            Self::Io(_) | Self::Pipeline(_) | Self::Gui(_) => 1,
        }
    }
}

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Argument(msg) | Self::Config(msg) | Self::Pipeline(msg) | Self::Gui(msg) => {
                write!(f, "{msg}")
            }
            Self::Io(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for CliError {}

impl From<std::io::Error> for CliError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<motionframe_engine::pipeline::PipelineError> for CliError {
    fn from(e: motionframe_engine::pipeline::PipelineError) -> Self {
        Self::Pipeline(format!("{e}"))
    }
}
