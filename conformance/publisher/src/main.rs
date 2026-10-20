// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "conformance-publish",
    about = "Native conformance site publisher"
)]
struct Args {
    #[arg(long, default_value = ".")]
    root: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    revision: String,
    #[arg(long)]
    candidate_tree: Option<String>,
    #[arg(long)]
    data_only: bool,
    #[arg(long, default_value = "auto")]
    workers: String,
}

fn main() -> anyhow::Result<()> {
    // This CLI owns its process and sets the publication timezone before threads start.
    unsafe {
        std::env::set_var("TZ", "America/Los_Angeles");
    }
    let args = Args::parse();
    conformance_publisher::publication::publish(conformance_publisher::publication::Options {
        root: args.root,
        output: args.output,
        revision: args.revision,
        candidate_tree: args.candidate_tree,
        data_only: args.data_only,
        workers: args.workers,
    })
}
