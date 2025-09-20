use ort::execution_providers::ExecutionProviderDispatch;

#[derive(Debug, Clone)]
pub struct RuntimeParameters {
    threads: usize,
    execution_providers: Vec<ExecutionProviderDispatch>,
}

impl RuntimeParameters {
    pub fn threads(&self) -> usize {
        self.threads
    }

    pub fn into_execution_providers(self) -> Vec<ExecutionProviderDispatch> {
        self.execution_providers
    }
}

impl Default for RuntimeParameters {
    fn default() -> Self {
        Self {
            threads: 1,
            execution_providers: vec![ExecutionProviderDispatch::CPU],
        }
    }
}