
import torch

from transformers import AutoTokenizer, T5ForConditionalGeneration
from peft import LoraConfig, get_peft_model


ARCH = "t5-base"

CONFIG = LoraConfig(
    task_type="SEQ_2_SEQ_LM",
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)


def main():

    tokenizer = AutoTokenizer.from_pretrained(ARCH, model_max_length=512)
    model = T5ForConditionalGeneration.from_pretrained(ARCH)

    encoder = model.encoder
    decoder = model.decoder

    pefted = get_peft_model(decoder, CONFIG)

    torch.no_grad()

    input_ids = tokenizer.encode("Hello, my dog is cute", add_special_tokens=True, return_tensors="pt")
    enc = encoder(input_ids)

    decoder.forward(input_ids, encoder_hidden_states=enc)

if __name__ == "__main__":
    main()