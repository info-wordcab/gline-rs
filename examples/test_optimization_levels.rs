use std::path::Path;
use std::time::Instant;
use ort::session::builder::GraphOptimizationLevel;
use gliner::orp::params::RuntimeParameters;
use gliner::model::{GLiNER, input::text::TextInput, params::Parameters};
use gliner::model::pipeline::span::SpanMode;
use gliner::orp::Result as OrpResult;

fn test_with_optimization_level(name: &str) -> OrpResult<()> {
    println!("\n=== Testing with {} ===", name);

    let model_path = "/home/aleks/.local/share/com.pais.handy/models/pii/model.onnx";
    let tokenizer_path = "/home/aleks/.local/share/com.pais.handy/models/pii/tokenizer.json";

    // Check if files exist
    if !Path::new(model_path).exists() {
        println!("Model not found at: {}", model_path);
        return Ok(());
    }
    if !Path::new(tokenizer_path).exists() {
        println!("Tokenizer not found at: {}", tokenizer_path);
        return Ok(());
    }

    println!("Loading model with {} optimization...", name);
    let start_load = Instant::now();

    // Create GLiNER model (uses current optimization level from model.rs)
    let model = match GLiNER::<SpanMode>::new(
        Parameters::default(),
        RuntimeParameters::default(),
        tokenizer_path,
        model_path,
    ) {
        Ok(model) => {
            println!("✓ Model loaded successfully in {:?}", start_load.elapsed());
            model
        },
        Err(e) => {
            println!("✗ Failed to load model: {:?}", e);
            return Err(e);
        }
    };

    // Test data with various PII types
    let test_texts = [
        "My name is John Smith and I live in New York.",
        "Contact me at john.smith@email.com or call 555-123-4567.",
        "My SSN is 123-45-6789 and DOB is 01/15/1985.",
        "I drive a Tesla Model 3 with license plate ABC-123.",
    ];

    let pii_labels = [
        "name", "first name", "last name", "email address", "phone number",
        "ssn", "dob", "location city", "location state", "vehicle id"
    ];

    println!("Running inference tests...");
    let start_inference = Instant::now();

    for (i, text) in test_texts.iter().enumerate() {
        println!("Test {}: '{}'", i + 1, text);

        let input = TextInput::from_str(&[text], &pii_labels)?;

        match model.inference(input) {
            Ok(output) => {
                println!("  ✓ Inference successful");
                println!("  Found {} spans", output.spans[0].len());
                for span in &output.spans[0] {
                    let (start, end) = span.offsets();
                    println!("    '{}' -> {} ({}..{})", span.text(), span.class(), start, end);
                }
            },
            Err(e) => {
                println!("  ✗ Inference failed: {:?}", e);
                return Err(e);
            }
        }
    }

    let inference_duration = start_inference.elapsed();
    println!("✓ All inference tests passed in {:?}", inference_duration);
    println!("Average per text: {:?}", inference_duration / test_texts.len() as u32);

    Ok(())
}

fn main() -> OrpResult<()> {
    println!("ONNX Optimization Level Test");
    println!("============================");

    // Test with current setting (should be Disable based on our analysis)
    println!("Testing with current optimization level (likely Disable)...");
    test_with_optimization_level("current setting")?;

    // Note: To properly test Level3, we need to temporarily modify the source code
    // For now, let's document what we would do:
    println!("\n=== Next Steps ===");
    println!("To test Level3 optimization:");
    println!("1. Temporarily modify src/orp/model.rs to use GraphOptimizationLevel::Level3");
    println!("2. Recompile and run this test again");
    println!("3. If successful, make the change permanent");

    // Test basic functionality to ensure our test setup works
    println!("\n=== Basic Functionality Test ===");
    let model_path = "/home/aleks/.local/share/com.pais.handy/models/pii/model.onnx";
    let tokenizer_path = "/home/aleks/.local/share/com.pais.handy/models/pii/tokenizer.json";

    if Path::new(model_path).exists() && Path::new(tokenizer_path).exists() {
        println!("✓ Model files found");
        println!("✓ Test framework ready");

        // Quick inference test
        let model = GLiNER::<SpanMode>::new(
            Parameters::default(),
            RuntimeParameters::default(),
            tokenizer_path,
            model_path,
        )?;

        let input = TextInput::from_str(&["My name is Alice"], &["name"])?;
        let output = model.inference(input)?;
        println!("✓ Basic inference test passed");

        if !output.spans.is_empty() && !output.spans[0].is_empty() {
            println!("✓ PII detection working: found '{}' as {}",
                output.spans[0][0].text(),
                output.spans[0][0].class()
            );
        }
    } else {
        println!("✗ Model files not found - please ensure PII model is downloaded");
    }

    println!("\n=== Test Complete ===");
    println!("Ready to proceed with optimization level changes if basic test passed.");

    Ok(())
}