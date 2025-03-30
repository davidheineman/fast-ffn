import torch
import torch.nn as nn

class FFF(nn.Module):
    def __init__(self, input_width: int, output_width: int, depth: int, activation=nn.GELU, device=None):
        super().__init__()

        self.input_width = input_width
        self.output_width = output_width

        self.depth = depth
        self.n_nodes = 2 ** (depth + 1) - 1

        self.activation = activation

        self.linear_in = nn.Linear(input_width, self.n_nodes, bias=False, device=device)
        self.linear_out = nn.Linear(self.n_nodes, output_width, bias=False, device=device)

    def forward(self, x):
        batch_size, seq_length, input_width = x.shape # (batch_size, seq_length, input_width)

        current_nodes = torch.zeros((batch_size, seq_length), dtype=torch.long, device=x.device)  # (batch_size, seq_length)
        all_nodes = torch.zeros(batch_size, seq_length, self.depth + 1, dtype=torch.long, device=x.device)  # (batch_size, seq_length, depth+1)
        all_logits = torch.empty((batch_size, seq_length, self.depth + 1), dtype=torch.float, device=x.device)  # (batch_size, seq_length, depth+1)

        for i in range(self.depth + 1):
            all_nodes[:, :, i] = current_nodes
            plane_coeffs = self.linear_in.weight.index_select(dim=0, index=current_nodes.flatten()) \
                .view(batch_size * seq_length, input_width)
            
            plane_coeff_score = torch.bmm(
                x.view(-1, 1, input_width),  # Flattened x
                plane_coeffs.unsqueeze(-1)  # Add feature dim
            )  # (batch_size * seq_length, 1, 1)
            
            plane_score = plane_coeff_score.view(batch_size, seq_length)
            all_logits[:, :, i] = plane_score
            
            plane_choices = (plane_score >= 0).long()
            current_nodes = current_nodes * 2 + plane_choices + 1

        selected_linear_out = self.linear_out.weight.T.index_select(0, index=all_nodes.flatten()) \
            .view(batch_size, seq_length, self.depth + 1, self.output_width)  # (batch_size, seq_length, depth+1, output_width)

        mlp1 = self.activation(all_logits)  # (batch_size, seq_length, depth+1)
        mlp2 = torch.einsum("bsl,bslw->bsw", mlp1, selected_linear_out)  # (batch_size, seq_length, output_width)

        return mlp2