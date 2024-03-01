"""Compare the outputs of HF and vLLM when using greedy sampling.

The data is from HF implementation is generated ahead of time on A100 or H100 by using
`python generate-snapshot.py` or manually update the `snapshot.json` file.

Similar to test_models.py, this test should be ran with `--forked` flag locally to avoid
hanging and nccl process group errors.
"""
import json
import pytest
import ray

SNAPSHOT_PATH = "./snapshot.json"

MODELS = [
    "facebook/opt-125m",
    "meta-llama/Llama-2-7b-hf",
    "mistralai/Mistral-7B-v0.1",
    #"Deci/DeciLM-7b",  # NOTE(simon): DeciLMForCausalLM.__init__() takes from 1 to 3 positional arguments but 4 were given, NOTE(sahil): vLLM vs HF Outputs not matching for 6/8 test prompts
    #"tiiuae/falcon-7b", # NOTE(simon): ValueError: Total number of attention heads (71) must be divisible by tensor parallel size (2)
    "gpt2",
    "bigcode/tiny_starcoder_py",
    "EleutherAI/gpt-j-6b",
    "EleutherAI/pythia-70m",
    "bigscience/bloom-560m",
    "mosaicml/mpt-7b",
    "microsoft/phi-2",
    "stabilityai/stablelm-3b-4e1t",
]
DTYPE = "float"
MAX_TOKENS = 128

@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", [DTYPE])
@pytest.mark.parametrize("max_tokens", [MAX_TOKENS])
def test_model_snapshot(
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
):
    key = f"{model}-{dtype}-{max_tokens}"
    with open(SNAPSHOT_PATH) as f:
        snapshot = json.load(f)

    # Note we are using 2 L4 GPUs for tensor parallelism to run 7b model at float32.
    vllm_model = vllm_runner(
        model,
        dtype=dtype,
        tensor_parallel_size=2,
        enforce_eager=True,
        max_model_len=1024,
    )
    example_prompts = [i.strip() for i in example_prompts]
    vllm_outputs = vllm_model.generate_greedy(example_prompts, max_tokens)
    del vllm_model

    for i in range(len(example_prompts)):
        hf_output_str = snapshot[key][example_prompts[i]]
        _, vllm_output_str = vllm_outputs[i]
        assert hf_output_str == vllm_output_str, (
            f"Test{i}:\nHF: {hf_output_str!r}\nvLLM: {vllm_output_str!r}")


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])
