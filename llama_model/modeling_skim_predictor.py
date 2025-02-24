import torch
from torch import nn
from torch.distributions import Categorical

class C2FPromptPruner_PolicyNetwork(nn.Module):

    # Input [M, L(512), E(768)]:
    # M: original few-shot num
    # L: token limit for each shot (of CoT examplar), default=512
    # E: embedding size of each token, default=768 (BERT embedding size)
    # Output binary mask [M] and [N, L], where
    # sum(M) = N: final few-shot num, default=4
    # N * sum(L) < 2048-256 

    def __init__(self, alpha1, alpha2, target_token, feat_shape):
        super(C2FPromptPruner_PolicyNetwork, self).__init__()

        self.length_limit = torch.tensor(target_token)
        self.alpha1 = torch.tensor(alpha1)
        self.alpha2 = torch.tensor(alpha2)

        self.shot_pruner = SkimPredictor(input_size = feat_shape, output_size = 2, hidden_size = 512)
        self.shot_classfier = nn.Softmax(dim = 1)
        self.token_pruner = SkimPredictor(input_size = feat_shape, output_size = 2, hidden_size = 512)
        self.token_classfier = nn.Softmax(dim = 2) 

    def forward(self, input_embeds):

        # first stage: coarse shot selection
        input_sentence_embeds = torch.mean(input_embeds, dim=1) # size should be [M, 768]
        sentence_skim = self.shot_pruner(input_sentence_embeds) # size should be [M, 2]
        shot_prob = self.shot_classfier(sentence_skim) # size should be [M]
        log_shot_action_prob, shot_action = self.select_action(shot_prob)

        while torch.sum(shot_action) == 0: # make sure that there is at least one shot being selected
            log_shot_action_prob, shot_action = self.select_action(shot_prob)
            
        shot_selection = shot_action > 0 # boolean index tensor of size [M], where N index are True

        pruned_input_embeds = input_embeds[shot_selection,:,:] # size should be [N, 512, 768]
        token_skim = self.token_pruner(pruned_input_embeds) # size should be [N, 512, 2]
        token_prob = self.token_classfier(token_skim) # size should be [N, 512]

        token_prob_flatten = torch.flatten(token_prob, end_dim=1)
        log_token_action_prob, token_action = self.select_action(token_prob_flatten) # size should be [N, 512]

        token_action = token_action.view(-1,512)

        # avoid using inplace operation
        token_level_mask = torch.zeros_like(token_action)  
        token_level_mask[:, :3] = 1

        token_action = torch.where(token_level_mask == 1, torch.ones_like(token_action), token_action) 
        token_action = token_action.view(-1)

        token_selection = token_action > 0 
        total_remained_token = torch.sum(token_action)
        reward_selector = self.reward(total_remained_token)

        return reward_selector, shot_selection, token_selection, log_shot_action_prob, log_token_action_prob

    def select_action(self, probs):
        m = Categorical(probs)
        action = m.sample()
        return m.log_prob(action), action
    
    def reward(self, total_remained_token):
        if total_remained_token>self.length_limit:
            beta = self.alpha1
        else:
            beta = self.alpha2
        reward = (total_remained_token/self.length_limit).pow(beta)
        return reward

    def fix_forward(self, input_embeds, target_shot, target_token):

        # first stage: coarse shot selection
        input_sentence_embeds = torch.mean(input_embeds, dim=1) # size should be [M, 768]
        sentence_skim = self.shot_pruner(input_sentence_embeds) # size should be [M, 2]
        shot_prob = self.shot_classfier(sentence_skim) # size should be [M, 2]
        shot_prob = shot_prob[:,1]

        _, top_shot_indices = torch.topk(shot_prob, target_shot)
        top_shot_positions_exapand = top_shot_indices.unsqueeze(1).unsqueeze(2)
        pruned_input_embeds = torch.gather(input_embeds, 0, top_shot_positions_exapand.expand(-1, input_embeds.size(1), input_embeds.size(2))) # size should be [N, 512, 768]

        shot_selection = torch.zeros_like(shot_prob)
        shot_selection[top_shot_indices] = 1

        token_skim = self.token_pruner(pruned_input_embeds) # size should be [N, 512, 2]
        token_prob = self.token_classfier(token_skim) # size should be [N, 512]
        token_prob = token_prob[:,:,1]

        token_prob_flatten = torch.flatten(token_prob, end_dim=1)
        _, top_token_indices = torch.topk(token_prob_flatten, target_token) # size should be [N*256]

        token_action = torch.zeros_like(token_prob_flatten)
        token_action[top_token_indices] = 1
        token_action = token_action.view(-1,512)

        # avoid using inplace operation
        token_level_mask = torch.zeros_like(token_action)  
        token_level_mask[:, :3] = 1

        token_action = torch.where(token_level_mask == 1, torch.ones_like(token_action), token_action) 
        token_action = token_action.view(-1)

        return shot_selection, token_action
    
class C2FPromptPruner_PolicyNetwork_Padding(nn.Module):

    def __init__(self, alpha1, alpha2, target_token, feat_shape, padding_size):
        super(C2FPromptPruner_PolicyNetwork_Padding, self).__init__()

        self.length_limit = torch.tensor(target_token)
        self.alpha1 = torch.tensor(alpha1)
        self.alpha2 = torch.tensor(alpha2)
        self.padding_size = padding_size

        self.shot_pruner = SkimPredictor(input_size = feat_shape, output_size = 2, hidden_size = 512)
        self.shot_classfier = nn.Softmax(dim = 1)
        self.token_pruner = SkimPredictor(input_size = feat_shape, output_size = 2, hidden_size = 512)
        self.token_classfier = nn.Softmax(dim = 2) 

    def forward(self, input_embeds):

        # first stage: coarse shot selection
        input_sentence_embeds = torch.mean(input_embeds, dim=1) # size should be [M, 768]
        sentence_skim = self.shot_pruner(input_sentence_embeds) # size should be [M, 2]
        shot_prob = self.shot_classfier(sentence_skim) # size should be [M]
        log_shot_action_prob, shot_action = self.select_action(shot_prob)

        while torch.sum(shot_action) == 0: # make sure that there is at least one shot being selected
            log_shot_action_prob, shot_action = self.select_action(shot_prob)
            
        shot_selection = shot_action > 0 # boolean index tensor of size [M], where N index are True

        pruned_input_embeds = input_embeds[shot_selection,:,:] # size should be [N, 512, 768]
        token_skim = self.token_pruner(pruned_input_embeds) # size should be [N, 512, 2]
        token_prob = self.token_classfier(token_skim) # size should be [N, 512]

        token_prob_flatten = torch.flatten(token_prob, end_dim=1)
        log_token_action_prob, token_action = self.select_action(token_prob_flatten) # size should be [N, 512]

        token_action = token_action.view(-1, self.padding_size)

        # avoid using inplace operation
        token_level_mask = torch.zeros_like(token_action)  
        token_level_mask[:, :3] = 1

        token_action = torch.where(token_level_mask == 1, torch.ones_like(token_action), token_action) 
        token_action = token_action.view(-1)

        token_selection = token_action > 0 
        total_remained_token = torch.sum(token_action)
        reward_selector = self.reward(total_remained_token)

        return reward_selector, shot_selection, token_selection, log_shot_action_prob, log_token_action_prob

    def select_action(self, probs):
        m = Categorical(probs)
        action = m.sample()
        return m.log_prob(action), action
    
    def reward(self, total_remained_token):
        if total_remained_token>self.length_limit:
            beta = self.alpha1
        else:
            beta = self.alpha2
        reward = (total_remained_token/self.length_limit).pow(beta)
        return reward

    def fix_forward(self, input_embeds, target_shot, target_token):

        # first stage: coarse shot selection
        input_sentence_embeds = torch.mean(input_embeds, dim=1) # size should be [M, 768]
        sentence_skim = self.shot_pruner(input_sentence_embeds) # size should be [M, 2]
        shot_prob = self.shot_classfier(sentence_skim) # size should be [M, 2]
        shot_prob = shot_prob[:,1]

        _, top_shot_indices = torch.topk(shot_prob, target_shot)
        top_shot_positions_exapand = top_shot_indices.unsqueeze(1).unsqueeze(2)
        pruned_input_embeds = torch.gather(input_embeds, 0, top_shot_positions_exapand.expand(-1, input_embeds.size(1), input_embeds.size(2))) # size should be [N, 512, 768]

        shot_selection = torch.zeros_like(shot_prob)
        shot_selection[top_shot_indices] = 1

        token_skim = self.token_pruner(pruned_input_embeds) # size should be [N, 512, 2]
        token_prob = self.token_classfier(token_skim) # size should be [N, 512]
        token_prob = token_prob[:,:,1]

        token_prob_flatten = torch.flatten(token_prob, end_dim=1)
        _, top_token_indices = torch.topk(token_prob_flatten, target_token) # size should be [N*256]

        token_action = torch.zeros_like(token_prob_flatten)
        token_action[top_token_indices] = 1
        token_action = token_action.view(-1,self.padding_size)

        # avoid using inplace operation
        token_level_mask = torch.zeros_like(token_action)  
        token_level_mask[:, :3] = 1

        token_action = torch.where(token_level_mask == 1, torch.ones_like(token_action), token_action) 
        token_action = token_action.view(-1)
        return shot_selection, token_action
    
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

def test_init_skim_predictor():

    #skim_predictor = torch.nn.Linear(768,2) 
    skim_predictor = SkimPredictor(768,2) 
    #init_skim_predictor(skim_predictor)

    #print(skim_predictor.weight, skim_predictor.bias)

    rand_input = torch.rand((16, 512, 768))
    print(skim_predictor(rand_input))

if __name__ == "__main__":
    test_init_skim_predictor()