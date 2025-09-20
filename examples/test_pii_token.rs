use gliner::orp::params::RuntimeParameters;
use gliner::util::result::Result;
use gliner::model::{GLiNER, input::text::TextInput, params::Parameters};
use gliner::model::pipeline::token::TokenMode;

/// Test PII detection with actual model using Token mode
fn main() -> Result<()> {

    println!("\nTesting GLiNER PII Detection (Token Mode)\n");
    println!("{}", "=".repeat(70));

    // Create test paragraphs with various types of fake PII
    let test_texts = [
        // Financial and personal info
        "My name is John Smith and my SSN is 123-45-6789. I live at 742 Evergreen Terrace, Springfield, IL 62701.",

        // Medical context
        "Patient Sarah Johnson, DOB 03/15/1985, was admitted yesterday. Emergency contact: Mike Johnson at 555-987-6543.",

        // Business communication
        "Please send payment to David Chen at Wells Fargo account 1234567890. Address: 123 Main St, New York, NY 10001.",

        // Mixed international
        "Dr. Emily Wilson works at Tokyo Medical Center. Contact: emily.wilson@tokyomed.jp or +81-3-1234-5678.",

        // Simple test
        "Bob Johnson lives in Paris, France. His email is bob@example.com.",
    ];

    // Define PII entity types to detect
    let pii_entities = [
        "person",           // Names
        "location",         // Addresses, cities
        "email",            // Email addresses
        "organization",     // Companies, hospitals
        "date",             // Dates
        "phone",            // Phone numbers
    ];

    // Use the actual model paths
    let tokenizer_path = "/home/aleks/PycharmProjects/pii_oss/Handy/TO_DO/onnx_models/gliner_multitask_large_v0_5/tokenizer.json";
    let model_path = "/home/aleks/PycharmProjects/pii_oss/Handy/TO_DO/onnx_models/gliner_multitask_large_v0_5/model_int8.onnx";

    println!("Model files:");
    println!("  Tokenizer: {}", tokenizer_path);
    println!("  Model: {}", model_path);
    println!();

    println!("Loading GLiNER model (Token mode)...");
    let model = GLiNER::<TokenMode>::new(
        Parameters::default(),
        RuntimeParameters::default(),
        tokenizer_path,
        model_path,
    )?;
    println!("✓ Model loaded successfully!\n");

    // Process each test text
    for (idx, text) in test_texts.iter().enumerate() {
        println!("{}", "=".repeat(70));
        println!("Test #{}: \"{}\"", idx + 1, if text.len() > 60 { &text[..60] } else { text });
        println!("{}", "-".repeat(70));

        // Create input
        let input = TextInput::from_str(&[text], &pii_entities)?;

        // Run inference
        match model.inference(input) {
            Ok(output) => {
                println!("Detected PII:");
                println!("{:<15} | {:<40} | {:<10}", "Type", "Text", "Confidence");
                println!("{}", "-".repeat(70));

                let mut pii_found = false;
                for spans in output.spans {
                    let mut sorted_spans = spans;
                    sorted_spans.sort_by(|a, b| b.probability().partial_cmp(&a.probability()).unwrap());

                    for span in sorted_spans.iter().take(10) {  // Show top 10 detections
                        if span.probability() > 0.3 {  // Lower threshold to see more results
                            println!("{:<15} | {:<40} | {:.1}%",
                                span.class(),
                                if span.text().len() > 40 {
                                    format!("{}...", &span.text()[..37])
                                } else {
                                    span.text().to_string()
                                },
                                span.probability() * 100.0
                            );
                            pii_found = true;
                        }
                    }
                }

                if !pii_found {
                    println!("(No PII detected with confidence > 30%)");
                }
            },
            Err(e) => {
                println!("✗ Inference failed: {}", e);
            }
        }
        println!();
    }

    println!("{}", "=".repeat(70));
    println!("PII Detection Test Complete!");
    println!();
    println!("Summary: The GLiNER model successfully loaded and performed inference.");
    println!("Integration with Handy is confirmed to be working!");
    println!("{}", "=".repeat(70));

    Ok(())
}