use ort::session::input::SessionInputs;
use ort::{inputs, value::Tensor};
use composable::Composable;
use crate::util::result::Result;
use super::super::encoded::EncodedInput;
use super::super::super::pipeline::context::EntityContext;


const TENSOR_INPUT_IDS: &str = "input_ids";
const TENSOR_ATTENTION_MASK: &str = "attention_mask";
const TENSOR_WORD_MASK: &str = "words_mask";
const TENSOR_TEXT_LENGTHS: &str = "text_lengths";


/// Ready-for-inference tensors (token mode)
pub struct TokenTensors<'a> {
    pub tensors: SessionInputs<'a, 'a>,
    pub context: EntityContext,
}

impl TokenTensors<'_> {

    pub fn from(encoded: EncodedInput) -> Result<Self> {
        // Create owned tensors from the encoded data
        let inputs = inputs![
            TENSOR_INPUT_IDS => Tensor::from_array(encoded.input_ids.clone())?,
            TENSOR_ATTENTION_MASK => Tensor::from_array(encoded.attention_masks.clone())?,
            TENSOR_WORD_MASK => Tensor::from_array(encoded.word_masks.clone())?,
            TENSOR_TEXT_LENGTHS => Tensor::from_array(encoded.text_lengths.clone())?,
        ];
        Ok(Self {
            tensors: SessionInputs::from(inputs),
            context: EntityContext { 
                texts: encoded.texts, 
                tokens: encoded.tokens, 
                entities: encoded.entities, 
                num_words: encoded.num_words 
            },            
        })
    }

    pub fn inputs() -> [&'static str; 4] {
        [TENSOR_INPUT_IDS, TENSOR_ATTENTION_MASK, TENSOR_WORD_MASK, TENSOR_TEXT_LENGTHS]
    }

}


/// Composable: Encoded => TokenTensors
#[derive(Default)]
pub struct EncodedToTensors { }


impl<'a> Composable<EncodedInput, TokenTensors<'a>> for EncodedToTensors {
    fn apply(&self, input: EncodedInput) -> Result<TokenTensors<'a>> {
        TokenTensors::from(input)
    }
}


/// Composable: TokenTensors => (SessionInput, TensorsMeta) 
#[derive(Default)]
pub struct TensorsToSessionInput { }


impl<'a> Composable<TokenTensors<'a>, (SessionInputs<'a, 'a>, EntityContext)> for TensorsToSessionInput {
    fn apply(&self, input: TokenTensors<'a>) -> Result<(SessionInputs<'a, 'a>, EntityContext)> {
        Ok((input.tensors, input.context))
    }
}
