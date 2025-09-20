// Local orp replacement module for direct ort integration
// This module provides a simplified version of orp functionality using ort 2.0.0-rc.10

pub mod model;
pub mod pipeline;
pub mod params;

pub type Result<T> = core::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;