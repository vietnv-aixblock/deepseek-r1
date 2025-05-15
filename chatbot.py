from dataclasses import asdict, dataclass
from textwrap import dedent
from types import SimpleNamespace

import gradio as gr
import torch
from loguru import logger

# Log in to Hugging Face Hub
from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('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')

# Determine the device to use (GPU if available, otherwise CPU)
device = 0 if torch.cuda.is_available() else -1

# Dictionary mapping model names to their Hugging Face Hub identifiers
llama_models = {
    "DeepSeek-R1-Distill-Qwen-1.5B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "DeepSeek-R1-Distill-Qwen-7B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "DeepSeek-R1-Distill-Llama-8B": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "DeepSeek-R1-Distill-Qwen-14B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    # "Llama3.2": "meta-llama/Llama-3.2-1B",
    # "Qwen2.5-7B-Instruct-1M": "Qwen/Qwen2.5-7B-Instruct-1M",
    # "Qwen2.5-14B-Instruct-1M": "Qwen/Qwen2.5-14B-Instruct-1M",
    # "llama-nemotron": "itsnebulalol/Llama-3.2-Nemotron-3B-Instruct",
}
SYSTEM_PROMPT = "You are a helpful assistant."
MAX_MAX_NEW_TOKENS = 2048  # sequence length 2048
MAX_NEW_TOKENS = 512

@dataclass
class Config:
    max_new_tokens: int = MAX_NEW_TOKENS
    repetition_penalty: float = 1.1
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 0.9


# stats_default = SimpleNamespace(llm=model, system_prompt=SYSTEM_PROMPT, config=Config())
stats_default = SimpleNamespace(llm=None, system_prompt=SYSTEM_PROMPT, config=Config())
# Function to load the model and tokenizer
def load_model(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="cpu",
        quantization_config=None
    )
    model.to('cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id =  tokenizer.eos_token_id
    # tokenizer.padding_side = 'left'
    return model,tokenizer

# Cache to store loaded models
model_cache = {}
tokenizer_cache= {}
# Function to generate chat responses
def generate_chat(user_input, history, model_choice):
    # print(model_choice)
    # print(model_cache)
    # print(tokenizer_cache)

    # Load the model if not already cached
    try:
        model = model_cache[model_choice]
        tokenizer = tokenizer_cache[model_choice]
    except KeyError:
        model,tokenizer = load_model(llama_models[model_choice])
        model_cache[model_choice] = model
        tokenizer_cache[model_choice] = tokenizer

    # Initial system prompt
    system_prompt = {"role": "system", "content": stats_default.system_prompt}

    # Initialize history if it's None
    if history is None:
        history = [system_prompt]
     # Append user input to history
    history.append({"role": "user", "content": user_input})
   
    text = tokenizer.apply_chat_template(
       [{"role": "system", "content": stats_default.system_prompt},{"role": "user", "content": user_input}],
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True)

    outputs = model.generate(
        input_ids=inputs["input_ids"].to(model.device),
        attention_mask=inputs["attention_mask"].to(model.device),
        max_length=stats_default.config.max_new_tokens, ##change this to align with the official usage
        num_return_sequences=2,
        do_sample=True,  ##change this to align with the official usage,
        top_p=stats_default.config.top_p, top_k=stats_default.config.top_k, temperature=stats_default.config.temperature,
        pad_token_id=tokenizer.eos_token_id
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs["input_ids"], outputs)
    ]

    outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    response=outputs[0]
    # print(response)
    # Append model response to history
    history.append({"role": "assistant", "content": response})
    
    return history

# Create Gradio interface
css = """
    .importantButton {
        background: linear-gradient(45deg, #7e0570,#5d1c99, #6e00ff) !important;
        border: none !important;
    }
    .importantButton:hover {
        background: linear-gradient(45deg, #ff00e0,#8500ff, #6e00ff) !important;
        border: none !important;
    }
    .disclaimer {font-variant-caps: all-small-caps; font-size: xx-small;}
    .xsmall {font-size: x-small;}
"""
with gr.Blocks(
            theme=gr.themes.Soft(text_size="sm"),
            title="Deepseek R1 Chatbot",
            css=css,
        ) as demo:
    stats = gr.State(stats_default)
    """Do exmaple_list css."""
    # pylint: disable=invalid-name, line-too-long,


    etext = """In America, where cars are an important part of the national psyche, a decade ago people had suddenly started to drive less, which had not happened since the oil shocks of the 1970s. """
    example_list = [
        ["What are the top A.I. companies in the world?"],
        [
            " What is deep learning?"
        ],
        ["What is natural language processing (NLP)?"],
        [
            "Will robotics kill jobs or create them?"
        ],
        [
            "A.I. in gaming and sports betting."
        ],
        ["What is artificial intelligence?"],
        ["What role will A.I. play in health care?"],
        [
            "A.I. in self-driving cars?"
        ],
        [" A.I. in investment and finance."],
        [" A.I. in marketing and advertising."],
        [" A.. in inductry and buildings."],
        [" A.I. and the future of work."],
        ["Let talkk me about DeepSeek R1"],
        ["Let talkk me about Nemotron"],
        ["Let talkk me about Kimi k1.5"],
        ["Let talkk me about Qwen2.5 1M tokens"],
        ["What is the AI Control Problem?"],
        ["Where are we with the AI Control Problem currently?"],
        ["What Is AI Safety?"],

    ]
    gr.Markdown("<h1><center>Chat with DeepSeek R1 Models</center></h1>")
    with gr.Row():
        with gr.Column(scale=5):
            # Dropdown to select model
            model_choice = gr.Dropdown(list(llama_models.keys()), label="Select LLM Model", type="value")
            # Chatbot interface
            chatbot = gr.Chatbot(label="Chatbot Interface", type="messages")
            # Textbox for user input
            txt_input = gr.Textbox(show_label=False, placeholder="Type your message here...")

            # Function to handle user input and generate response
            def respond(user_input, chat_history,model_choice):
                # model_choice= "DeepSeek-R1-1B"
                if model_choice is None:
                    model_choice = list(llama_models.keys())[0]
                updated_history = generate_chat(user_input, chat_history, model_choice)
                return "", updated_history

            # Submit user input on pressing Enter
            txt_input.submit(respond, [txt_input, chatbot,model_choice], [txt_input, chatbot])
            # Button to submit user input
            submit_btn = gr.Button("Submit")
            submit_btn.click(respond, [txt_input, chatbot,model_choice], [txt_input, chatbot])
    with gr.Row():
        with gr.Accordion(label="Advanced Options", open=False):
            system_prompt = gr.Textbox(
                label="System prompt",
                value=stats_default.system_prompt,
                lines=3,
                visible=True,
            )
            max_new_tokens = gr.Slider(
                label="Max new tokens",
                minimum=1,
                maximum=MAX_MAX_NEW_TOKENS,
                step=1,
                value=stats_default.config.max_new_tokens,
            )
            repetition_penalty = gr.Slider(
                label="Repetition penalty",
                minimum=0.1,
                maximum=40.0,
                step=0.1,
                value=stats_default.config.repetition_penalty,
            )
            temperature = gr.Slider(
                label="Temperature",
                minimum=0.51,
                maximum=40.0,
                step=0.1,
                value=stats_default.config.temperature,
            )
            top_p = gr.Slider(
                label="Top-p (nucleus sampling)",
                minimum=0.05,
                maximum=1.0,
                step=0.05,
                value=stats_default.config.top_p,
            )
            top_k = gr.Slider(
                label="Top-k",
                minimum=0,
                maximum=1000,
                step=1,
                value=stats_default.config.top_k,
            )

            def system_prompt_fn(system_prompt):
                stats.value.system_prompt = system_prompt
                logger.debug(f"{stats.value.system_prompt=}")

            def max_new_tokens_fn(max_new_tokens):
                stats.value.config.max_new_tokens = max_new_tokens
                logger.debug(f"{stats.value.config.max_new_tokens=}")

            def repetition_penalty_fn(repetition_penalty):
                stats.value.config.repetition_penalty = repetition_penalty
                logger.debug(f"{stats.value=}")

            def temperature_fn(temperature):
                stats.value.config.temperature = temperature
                logger.debug(f"{stats.value=}")

            def top_p_fn(top_p):
                stats.value.config.top_p = top_p
                logger.debug(f"{stats.value=}")

            def top_k_fn(top_k):
                stats.value.config.top_k = top_k
                logger.debug(f"{stats.value=}")

            system_prompt.change(system_prompt_fn, system_prompt)
            max_new_tokens.change(max_new_tokens_fn, max_new_tokens)
            repetition_penalty.change(repetition_penalty_fn, repetition_penalty)
            temperature.change(temperature_fn, temperature)
            top_p.change(top_p_fn, top_p)
            top_k.change(top_k_fn, top_k)

            def reset_fn(stats_):
                logger.debug("reset_fn")
                stats_ = gr.State(stats_default)
                logger.debug(f"{stats_.value=}")
                return (
                    stats_,
                    stats_default.system_prompt,
                    stats_default.config.max_new_tokens,
                    stats_default.config.repetition_penalty,
                    stats_default.config.temperature,
                    stats_default.config.top_p,
                    stats_default.config.top_k,
                )

            reset_btn = gr.Button("Reset")
            reset_btn.click(
                reset_fn,
                stats,
                [
                    stats,
                    system_prompt,
                    max_new_tokens,
                    repetition_penalty,
                    temperature,
                    top_p,
                    top_k,
                ],
            )
    with gr.Row():
        with gr.Accordion("Example inputs", open=True):
            etext = """In America, where cars are an important part of the national psyche, a decade ago people had suddenly started to drive less, which had not happened since the oil shocks of the 1970s. """
            examples = gr.Examples(
                examples=example_list,
                inputs=[txt_input],
                examples_per_page=60,
            )
    

# Launch the Gradio demo
demo.launch()
