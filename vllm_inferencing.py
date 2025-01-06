from vllm import LLM, SamplingParams
import torch.distributed as dist


dist.init_process_group(backend="nccl", init_method="env://")

prompts = [
        "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(model="HuggingFaceTB/SmolLM-135M-Instruct", dtype="float16")


outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")