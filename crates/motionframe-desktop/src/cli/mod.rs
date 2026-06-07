pub mod args;
pub mod config;
pub mod error;
pub mod job;
pub mod run;

pub use args::Cli;
pub use error::CliError;

/// Entry point: dispatch `convert` subcommand or launch GUI.
pub fn run(cli: Cli) -> Result<(), CliError> {
    match cli.command {
        Some(args::Command::Convert(args)) => {
            let cfg = if let Some(path) = &args.config {
                config::CliConfig::load(path)?
            } else {
                config::CliConfig::default()
            };
            let cfg = cfg.merge_args(&args)?;
            let convert_job = job::ConvertJob::from_config(cfg)?;
            run::run_convert(convert_job)
        }
        None => crate::app::run_gui().map_err(|e| CliError::Gui(format!("{e}"))),
    }
}
