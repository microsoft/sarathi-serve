from vllm import LLM, SamplingParams

NUM_SAMPLES = 100
INPUT_LENGTH = 10
OUTPUT_LENGTH = 20

# Create a sampling params object.
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    ignore_eos=True,
    max_tokens=OUTPUT_LENGTH,
)

# Create an LLM.
llm = LLM(model="codellama/CodeLlama-7b-Instruct-hf")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompt_token_ids=[[1] * INPUT_LENGTH
                                         for _ in range(NUM_SAMPLES)],
                       sampling_params=sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt_token_ids
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
