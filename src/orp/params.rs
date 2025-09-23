use ort::execution_providers::{ExecutionProviderDispatch, CPUExecutionProvider};

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

    /// Builder method to set execution providers
    pub fn with_execution_providers<I>(mut self, providers: I) -> Self
    where
        I: IntoIterator<Item = ExecutionProviderDispatch>
    {
        self.execution_providers = providers.into_iter().collect();
        self
    }

    /// Builder method to set number of threads
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads;
        self
    }
}

impl Default for RuntimeParameters {
    fn default() -> Self {
        Self {
            threads: 1,
            execution_providers: vec![CPUExecutionProvider::default().build()],
        }
    }
}