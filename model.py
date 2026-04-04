import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM

class ClinicalMicroVLM(nn.Module):
    def __init__(self, bnb_config=None):
        super().__init__()
        
        # 1. Vision Encoder (SigLIP 2)
        # Using so400m-patch14-384 which has a hidden size of 1152
        print("Loading Vision Encoder (SigLIP 2)...")
        # Extract the standalone vision_model to discard the text tower of SigLIP
        self.vision_encoder = AutoModel.from_pretrained(
            "google/siglip2-so400m-patch14-384", 
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).vision_model
        
        # 2. LLM Decoder (Qwen 2.5 Coder 1.5B)
        # Hidden size is 1536
        print("Loading LLM Decoder (Qwen 2.5 Coder 1.5B)...")
        if bnb_config is not None:
            self.llm = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-Coder-1.5B-Instruct",
                torch_dtype=torch.bfloat16,
                quantization_config=bnb_config,
                attn_implementation="sdpa",
                trust_remote_code=True
            )
        else:
            self.llm = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-Coder-1.5B-Instruct",
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                trust_remote_code=True
            )
        
        # 3. The Projector (2-layer MLP)
        print("Initializing Projector...")
        self.projector = nn.Sequential(
            nn.Linear(1152, 1536, dtype=torch.bfloat16),
            nn.GELU(),
            nn.Linear(1536, 1536, dtype=torch.bfloat16)
        )

    def forward(self, pixel_values, input_ids, attention_mask, labels=None):
        """
        Forward pass for the Micro-VLM.
        Args:
            pixel_values: (Batch_Size, Channels, Height, Width)
            input_ids: (Batch_Size, Sequence_Length)
            attention_mask: (Batch_Size, Sequence_Length)
            labels: (Batch_Size, Sequence_Length) optional, for causal LM loss computation
        Returns:
            CausalLMOutput containing loss and logits
        """
        # 1. Pass the pixel values through the Vision Encoder
        # The output last_hidden_state represents the image patches embedding
        # We wrap in torch.no_grad() because the vision encoder is fully frozen
        with torch.no_grad():
            vision_outputs = self.vision_encoder(pixel_values=pixel_values)
            image_embeds = vision_outputs.last_hidden_state # Shape: (B, num_patches, 1152)
        
        # 2. Pass the image embeddings through the Projector
        projected_image_embeds = self.projector(image_embeds) # Shape: (B, num_patches, 1536)
        
        # 3. Get text embeddings from the LLM
        with torch.no_grad():
            text_embeds = self.llm.get_input_embeddings()(input_ids) # Shape: (B, text_seq_len, 1536)
        
        # 4. Concatenate projected image embeddings with text embeddings
        # We prepend the image embeddings, which act as the "<image>" token representation.
        # So the sequence becomes: [Image Patches] + [Text Embeddings (\nExtract...)]
        inputs_embeds = torch.cat([projected_image_embeds, text_embeds], dim=1) # Shape: (B, num_patches + text_seq_len, 1536)
        
        # 5. Expand attention mask and labels for the image tokens
        B, num_patches, _ = projected_image_embeds.shape
        device = projected_image_embeds.device
        
        # Create attention mask for the image tokens (all 1s because we want the LLM to attend to the image)
        image_attention_mask = torch.ones((B, num_patches), dtype=attention_mask.dtype, device=device)
        extended_attention_mask = torch.cat([image_attention_mask, attention_mask], dim=1)
        
        if labels is not None:
            # We assign -100 to image tokens so the Causal LM loss function ignores them
            image_labels = torch.full((B, num_patches), -100, dtype=labels.dtype, device=device)
            extended_labels = torch.cat([image_labels, labels], dim=1)
        else:
            extended_labels = None
            
        # 6. Feed the concatenated embeddings into the LLM decoder
        # Using inputs_embeds instead of input_ids
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_attention_mask,
            labels=extended_labels
        )
        
        return outputs
