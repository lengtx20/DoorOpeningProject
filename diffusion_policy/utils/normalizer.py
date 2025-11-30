class Normalizer:
    def __init__(self, stats):
        self.stats = stats

    def normalize(self, batch):
        device = batch['action'].device

        act_mean = self.stats['action']['mean'].to(device)
        act_std  = self.stats['action']['std'].to(device)
        batch['action'] = (batch['action'] - act_mean) / act_std


        if 'agent_pos' in self.stats and 'agent_pos' in batch:
            if batch['agent_pos'].nelement() > 0:
                pos_mean = self.stats['agent_pos']['mean'].to(device)
                pos_std  = self.stats['agent_pos']['std'].to(device)

                batch['agent_pos'] = (batch['agent_pos'] - pos_mean) / pos_std

        return batch

    def unnormalize_action(self, action):
        device = action.device
        act_mean = self.stats['action']['mean'].to(device)
        act_std  = self.stats['action']['std'].to(device)
        return action * act_std + act_mean
