use std::{fs, path::PathBuf, sync::Arc};

use polars::{error::PolarsResult, io::parquet::write::ParquetWriteOptions, lazy::{dsl::{FileWriteFormat, SinkTarget, UnifiedSinkArgs}, frame::LazyFrame}, prelude::PlRefPath};

pub fn sink_to_local(lf: LazyFrame, path: &PathBuf) -> PolarsResult<()> {
    let unified_args = UnifiedSinkArgs::default();
    let write_options = ParquetWriteOptions::default();

    if let Some(par) = path.parent() {
        fs::create_dir_all(par)?;
    }

    let _ = lf.sink(
        polars::lazy::dsl::SinkDestination::File { target: SinkTarget::Path(PlRefPath::new(path.to_string_lossy().to_owned()))}, 
        FileWriteFormat::Parquet(Arc::new(write_options)),
        unified_args,
    )?
    .collect();

    Ok(())
}