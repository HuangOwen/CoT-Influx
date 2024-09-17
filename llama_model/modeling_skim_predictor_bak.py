from torch import nn
import torch

class C2FPromptPruner_Embedding(nn.Module):

    # Input [M, L(512), E(4096)]:
    # M: original few-shot num, defeault=8
    # L: token limit for each shot (of CoT examplar), default=512
    # E: embedding size of each token, default=768 (BERT embedding size)
    # Output binary mask [M] and [N, L], where
    # sum(M) = N: final few-shot num, default=4
    # N * sum(L) < 2048-256 

    def __init__(self):
        super(C2FPromptPruner_Embedding, self).__init__()

        self.targer_shot = 4
        self.targer_token = 256

        self.shot_pruner = SkimPredictor(input_size = 4096, output_size = 2, hidden_size = 4096)
        self.token_pruner = SkimPredictor(input_size = 4096, output_size = 2, hidden_size = 4096)

    def forward(self, input_embeds):

        # first stage: coarse shot selection
        input_sentence_embeds = torch.mean(input_embeds, dim=1) # size should be [M, 4096]
        sentence_skim = self.shot_pruner(input_sentence_embeds) 
        sentence_skim_mask = nn.functional.gumbel_softmax(sentence_skim, hard=False, tau=1) # size should be [M, 2]
        sentence_skim_mask = sentence_skim_mask[:,1] # size should be [M] 

        _, top_shot_indices = torch.topk(sentence_skim_mask, self.targer_shot)
        top_shot_positions, _ = torch.sort(top_shot_indices) # size should be [N] 
        top_shot_positions_exapand = top_shot_positions.unsqueeze(1).unsqueeze(2)
        pruned_input_embeds = torch.gather(input_embeds, 0, top_shot_positions_exapand.expand(-1, input_embeds.size(1), input_embeds.size(2)))
        pruned_input_embeds.requires_grad_(True)

        pruned_input_embeds_flatten = pruned_input_embeds.view(-1, 4096) # size should be [N*512, 4096]
        #print(pruned_input_embeds_flatten.size())

        token_skim = self.token_pruner(pruned_input_embeds_flatten)
        token_skim_mask = nn.functional.gumbel_softmax(token_skim, hard=False, tau=1) # size should be [N*512, 2]
        token_skim_mask = token_skim_mask[:,1] # size should be [N*512] 

        _, top_token_indices = torch.topk(token_skim_mask, self.targer_shot*self.targer_token) # size should be [N*256]
        top_token_positions, _ = torch.sort(top_token_indices) # size should be [N*256] 
        top_token_positions_exapand = top_token_positions.unsqueeze(1)
        pruned_input_embeds_final = torch.gather(pruned_input_embeds_flatten, 0, top_token_positions_exapand.expand(-1, pruned_input_embeds_flatten.size(1)))
        #print(pruned_input_embeds_final.size())

        return pruned_input_embeds, pruned_input_embeds_final, top_shot_positions, top_token_positions
    

class C2FPromptPruner(nn.Module):

    # Input [M, L(512), E(768)]:
    # M: original few-shot num, defeault=8
    # L: token limit for each shot (of CoT examplar), default=512
    # E: embedding size of each token, default=768 (BERT embedding size)
    # Output binary mask [M] and [N, L], where
    # sum(M) = N: final few-shot num, default=4
    # N * sum(L) < 2048-256 

    def __init__(self):
        super(C2FPromptPruner, self).__init__()

        self.targer_shot = 4
        self.targer_token = 256

        self.shot_pruner = SkimPredictor(input_size = 768, output_size = 2, hidden_size = 768)
        self.token_pruner = SkimPredictor(input_size = 768, output_size = 2, hidden_size = 768)

    def forward(self, input_embeds):

        # first stage: coarse shot selection
        input_sentence_embeds = torch.mean(input_embeds, dim=1) # size should be [M, 768]
        sentence_skim = self.shot_pruner(input_sentence_embeds) 
        sentence_skim_mask = nn.functional.gumbel_softmax(sentence_skim, hard=False, tau=1) # size should be [M, 2]
        sentence_skim_mask = sentence_skim_mask[:,1] # size should be [M] 

        _, top_shot_indices = torch.topk(sentence_skim_mask, self.targer_shot)
        top_shot_positions, _ = torch.sort(top_shot_indices) # size should be [N] 
        top_shot_positions_exapand = top_shot_positions.unsqueeze(1).unsqueeze(2)
        pruned_input_embeds = torch.gather(input_embeds, 0, top_shot_positions_exapand.expand(-1, input_embeds.size(1), input_embeds.size(2)))
        pruned_input_embeds.requires_grad_(True)

        pruned_input_embeds_flatten = pruned_input_embeds.view(-1, 768) # size should be [N*512, 768]
        #print(pruned_input_embeds_flatten.size())

        token_skim = self.token_pruner(pruned_input_embeds_flatten)
        token_skim_mask = nn.functional.gumbel_softmax(token_skim, hard=False, tau=1) # size should be [N*512, 2]
        token_skim_mask = token_skim_mask[:,1] # size should be [N*512] 

        _, top_token_indices = torch.topk(token_skim_mask, self.targer_shot*self.targer_token) # size should be [N*256]
        top_token_positions, _ = torch.sort(top_token_indices) # size should be [N*256] 
        top_token_positions_exapand = top_token_positions.unsqueeze(1)
        pruned_input_embeds_final = torch.gather(pruned_input_embeds_flatten, 0, top_token_positions_exapand.expand(-1, pruned_input_embeds_flatten.size(1)))
        #print(pruned_input_embeds_final.size())

        return pruned_input_embeds, pruned_input_embeds_final, sentence_skim_mask, token_skim_mask
    
def init_skim_predictor(module, mean_bias=5.0):

    if not isinstance(module, torch.nn.Linear):
        raise ValueError("only support initialization of linear skim predictor")

    # module.bias.data[1].fill_(5.0)
    # module.bias.data[0].fill_(-5.0)
    # module.weight.data.zero_()
    module.bias.data[1].normal_(mean=mean_bias, std=0.02)
    module.bias.data[0].normal_(mean=-mean_bias, std=0.02)
    module.weight.data.normal_(mean=0.0, std=0.02)
    module._skim_initialized = True

class SkimPredictor(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=None):
        super().__init__()
        
        self.hidden_size = hidden_size if hidden_size else input_size

        self.predictor = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, self.hidden_size),
            # nn.GELU(),
            # nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, output_size),
        )
        init_skim_predictor(self.predictor[-1],1.0)

    def forward(self, hidden_states):
        return self.predictor(hidden_states)

import torch
from torch import nn
from torch.distributions import Categorical

class SkimPredictor_Encoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=None, nhead=8):
        super().__init__()
        
        self.hidden_size = hidden_size if hidden_size else input_size
        self.nhead = nhead
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=self.nhead)

        self.predictor = nn.Sequential(
            nn.TransformerEncoder(encoder_layer, num_layers=1),
            nn.Linear(input_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, output_size),
        )
        init_skim_predictor(self.predictor[-1],1.0)

    def forward(self, hidden_states):
        return self.predictor(hidden_states)

def test_init_skim_predictor():

    #skim_predictor = torch.nn.Linear(768,2) 
    skim_predictor = SkimPredictor(768,2) 
    #init_skim_predictor(skim_predictor)

    #print(skim_predictor.weight, skim_predictor.bias)

    rand_input = torch.rand((16, 512, 768))
    print(skim_predictor(rand_input))

if __name__ == "__main__":
    test_init_skim_predictor()