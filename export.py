# %%
import dac
import torch
import transformers
from peft import LoraConfig, get_peft_model

# 1. Load Model
encoder = transformers.Wav2Vec2BertModel.from_pretrained(
    'facebook/w2v-bert-2.0', num_hidden_layers=8, layerdrop=0.0
).eval()

# 2. Configure LoRA
adapter_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="lora_only",
    target_modules=["output_dense"],
)

# --- FIX 1: Apply the attribute patch BEFORE get_peft_model ---
if not hasattr(encoder, "base_model_name_or_path"):
    encoder.base_model_name_or_path = "facebook/w2v-bert-2.0"

# --- FIX 2: Correct argument order (model, config) ---
encoder = get_peft_model(encoder, adapter_config)

# 3. Load Weights
original_weight = torch.load("/work/gj18/e43001/epoch=2-step=566956.ckpt", map_location="cpu", weights_only=False)["state_dict"]

# Filter keys and load
peft_state_dict = {
    # CHANGE: Replace 'student_ssl_model.' with 'base_model.model.' to match PEFT structure
    k.replace("student_ssl_model.", "base_model.model."): v 
    for k, v in original_weight.items() 
    if k.startswith("student_ssl_model.")
}
encoder.load_state_dict(peft_state_dict, strict=True)
# 4. Save
encoder.save_pretrained("sarulab-speech/sidon_raw_weight")
decoder = dac.model.dac.Decoder(
    input_channel=encoder.config.hidden_size,  # type: ignore
    channels=1536,
    rates=[8, 5, 4, 3, 2],
)
decoder.load_state_dict(
    {k.replace("decoder.", ""): v for k, v in original_weight.items() if k.startswith("decoder.")}, 
    strict=True
)

torch.save(decoder.state_dict(), "decoder_state_dict.pt")