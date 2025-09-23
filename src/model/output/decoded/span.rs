//! First step of span decoding (in span mode)

use composable::Composable;
use crate::util::math::sigmoid;
use crate::util::result::Result;
use crate::text::span::Span;
use crate::model::pipeline::context::EntityContext;
use crate::model::output::tensors::TensorOutput;
use super::SpanOutput;


const TENSOR_LOGITS: &str = "logits";


/// Decoding method for span mode.
/// 
/// See sections 2.1 and 2.3 of the [original paper](https://arxiv.org/abs/2311.08526).
/// Note: greedy search is not included in this step and must be applied subsequently.
pub struct TensorsToDecoded {
    threshold: f32,
    max_width: usize,
}

impl TensorsToDecoded {
    pub fn new(threshold: f32, max_width: usize) -> Self {
        Self { 
            threshold,
            max_width,
        }
    }

    pub fn outputs() -> [&'static str; 1] {
        [TENSOR_LOGITS]
    }

    fn decode(&self, input: &TensorOutput) -> Result<Vec<Vec<Span>>> {        
        // prepare output vector
        let batch_size = input.context.texts.len();
        let mut result: Vec<Vec<Span>> = Vec::new();

        // look for logits and check its shape
        let logits = input.tensors.get(TENSOR_LOGITS).ok_or("logits not found in model output")?;
        self.check_shape(logits.shape().to_vec(), &input.context)?;
        
        // extract the actual array
        let (shape, array_data) = logits.try_extract_tensor::<f32>()?;

        // Get dimensions from the shape
        if shape.len() != 4 {
            return Err(format!("Expected 4D tensor, got {}D", shape.len()).into());
        }
        // The expected shape is [batch_size, num_words, max_width, num_classes]
        let num_words = shape[1] as usize;
        let max_width = shape[2] as usize;
        let num_classes = shape[3] as usize;

        // Verify max_width matches what we expect
        if max_width != self.max_width {
            return Err(format!("Unexpected max_width: got {}, expected {}", max_width, self.max_width).into());
        }

        // Convert to ndarray - we need to construct the array from the raw data
        // The shape should be [batch_size, num_words, max_width, num_classes]
        let array = ndarray::ArrayView4::from_shape([batch_size, num_words, max_width, num_classes], array_data)
            .map_err(|e| format!("Failed to create array view: {}", e))?;

        // iterate over the sequences
        for sequence_id in 0..batch_size {
            // get a slice for the current sequence (1st dimension)
            let sequence = array.slice(ndarray::s![sequence_id, .., .., ..]);
            let num_tokens = input.context.tokens.get(sequence_id).unwrap().len();
            //println!("{:?}", sequence.map(|x| crate::util::math::sigmoid(*x)));
            
            // prepare the list of spans for this sequence
            let mut spans = Vec::new();
            
            // iterate over all spans
            for ((start, end, class), score) in sequence.indexed_iter() {
                // check that the tokens actually exist in the current sequence (we could do better here, to avoid iterating over these ones)
                // NOTE: 'end' here is a width/offset, not an absolute position, so the end position is start+end
                let end_position = start + end;
                if start >= num_tokens || end_position > num_tokens {
                    continue;
                }
                // check that the score is above threshold (otherwise continue)
                let score = sigmoid(*score);
                if score >= self.threshold {
                    // if yes, create the span (end_position is already calculated as start+end)
                    spans.push(input.context.create_span(sequence_id, start, end_position, class, score)?);
                }
            }
            
            // add the list of spans for this sequence
            result.push(spans);
        }
        
        // return
        Ok(result)
    }


    /// Checks coherence of the output shape
    /// Expected shape is (batch_size, num_words, num_spans, num_classes)
    fn check_shape(&self, actual_shape: Vec<i64>, context: &EntityContext) -> Result<()> {
        let expected_shape = vec![context.texts.len() as i64, context.num_words as i64, self.max_width as i64, context.entities.len() as i64];
        if actual_shape != expected_shape {
            Err("unexpected logits shape".into())
        }
        else {
            Ok(())
        }
    }

}

impl Composable<TensorOutput<'_>, SpanOutput> for TensorsToDecoded {
    fn apply(&self, input: TensorOutput) -> Result<SpanOutput> {        
        let decoded = self.decode(&input)?;
        Ok(SpanOutput::new(input.context.texts, input.context.entities, decoded))
    }
}