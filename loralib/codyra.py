import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class codyra_Linear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        module_name=None,
        lora_threshold=1,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        
        self.module_name = module_name
        self.rank = r
        self.init_rank = r
        self.in_features = in_features
        self.out_features = out_features
        
        self.lora_threshold = lora_threshold
        
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.gate = nn.Parameter(torch.randn(1, r))
            self.lora_scaling = 1
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)
            
        # Initialize variables for easy loading previous task's LoRA parameters
        self.old_lora_A = None
        self.old_lora_B = None
        self.old_rank = 0
        self.register_buffer("old_lora_merged", torch.zeros(1, dtype=torch.bool))
        
    def reset_parameters(self):
        """Reset LoRA parameters"""
        super().reset_parameters()
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)


    def _merge_lora_into_weight(self, lora_A, lora_B, gate=None):
        """Add the provided LoRA factors into the frozen base weight."""
        if lora_A is None or lora_B is None:
            return
        if lora_A.numel() == 0 or lora_B.numel() == 0:
            return

        with torch.no_grad():
            if gate is not None:
                scaled_B = lora_B.detach() * gate.detach()
            else:
                scaled_B = lora_B.detach()
            delta_weight = scaled_B @ lora_A.detach()
            if self.fan_in_fan_out:
                delta_weight = delta_weight.transpose(0, 1)
            self.weight.data.add_(delta_weight)


    def _ensure_old_lora_merged(self):
        """Merge any stored old LoRA weights into the base weight once."""
        if self.old_lora_A is None or self.old_lora_B is None:
            return
        if self.old_rank <= 0:
            return
        if bool(self.old_lora_merged.item()):
            return

        self._merge_lora_into_weight(self.old_lora_A, self.old_lora_B, gate=None)
        self.old_lora_merged.fill_(True)


    def increment_task(self):
        # Ensure any previously stored LoRA factors are merged before proceeding.
        self._ensure_old_lora_merged()

        if getattr(self, "lora_A", None) is not None:
            # Remove LoRA components where gate rank is zero.
            active_indices = (self.gate.abs() > 0).squeeze()
            if active_indices.dim() == 0:
                active_indices = active_indices.unsqueeze(0)

            if active_indices.any():
                self.lora_A = nn.Parameter(self.lora_A[active_indices, :])
                self.lora_B = nn.Parameter(self.lora_B[:, active_indices])
                self.gate = nn.Parameter(self.gate[:, active_indices])

                current_lora_A = self.lora_A
                current_lora_B = self.lora_B
                current_gate = self.gate

                # Merge current LoRA contribution into the frozen base weight.
                self._merge_lora_into_weight(current_lora_A, current_lora_B, gate=current_gate)

                device = self.weight.device
                scaled_current_A = current_lora_A.data.detach().to(device).clone() * current_gate.data.detach().T.to(device).clone()
                current_B_clone = current_lora_B.data.detach().to(device).clone()

                if self.old_lora_A is not None:
                    old_A = self.old_lora_A.data.detach().to(device).clone()
                    old_B = self.old_lora_B.data.detach().to(device).clone()
                    self.old_lora_A = nn.Parameter(torch.cat([old_A, scaled_current_A], dim=0))
                    self.old_lora_B = nn.Parameter(torch.cat([old_B, current_B_clone], dim=1))
                else:
                    # Register the current LoRA factors as historical parameters.
                    self.old_lora_A = nn.Parameter(scaled_current_A)
                    self.old_lora_B = nn.Parameter(current_B_clone)

                # Make them non-trainable and record the aggregated rank.
                self.old_lora_A.requires_grad = False
                self.old_lora_B.requires_grad = False
                self.old_rank += current_gate.shape[1]
                self.old_lora_merged.fill_(True)
            else:
                # No active components survived thresholding; keep merge flag intact.
                if self.old_lora_A is not None and self.old_rank > 0:
                    self.old_lora_merged.fill_(True)

        # Reset the current LoRA parameters for the new task
        if self.init_rank > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((self.init_rank, self.in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((self.out_features, self.init_rank)))
            self.gate = nn.Parameter(torch.randn(1, self.init_rank))
            # self.gate = nn.Parameter(torch.ones(1, self.init_rank))
            self.lora_scaling = 1
            self.rank = self.init_rank
            self.r = self.init_rank
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)    
            

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs
    ):
        merge_key = prefix + "old_lora_merged"
        if merge_key not in state_dict:
            state_dict[merge_key] = torch.zeros_like(self.old_lora_merged, dtype=torch.bool)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


    def forward(self, x: torch.Tensor, **kwargs):
        self._ensure_old_lora_merged()

        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        # Compute the base linear output
        result = F.linear(x, T(self.weight), bias=self.bias)

        # Add the contribution of the current LoRA parameters if available and not merged
        # if not get_cur_feat:
        if self.rank > 0 and not self.merged:
            lora_contribution = ((self.lora_dropout(x) @ self.lora_A.T).mul(self.gate) @ self.lora_B.T) 
            result += lora_contribution
        
        return result, 0 
