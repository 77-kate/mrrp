import torch
import torch.nn as nn

class DeepSurvLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def _compute_loss(self, P, T, E, M, mode):
        P_exp = torch.exp(P) # (B,)
        P_exp_B = torch.stack([P_exp for _ in range(P.shape[0])], dim=0) # (B, B)
        if mode == 'risk':
            E = E.float() * (M.sum(dim=1) > 0).float()
        elif mode == 'surv':
            E = (M.sum(dim=1) > 0).float()
        else:
            raise NotImplementedError
        P_exp_sum = (P_exp_B * M.float()).sum(dim=1)
        P_tmp = P_exp / (P_exp_sum+1e-6)
        loss = -torch.sum(torch.log(P_tmp.clip(1e-6, P_tmp.max().item())) * E) / torch.sum(E)
        return loss

    def forward(self, P_risk, T, E):
        # P: (B,)
        # T: (B,)
        # E: (B,) \in {0, 1}
        M_risk = T.unsqueeze(dim=1) < T.unsqueeze(dim=0) # (B, B)
        loss_risk = self._compute_loss(P_risk, T, E, M_risk, mode='risk')
        return loss_risk
    
def partial_ll_loss(lrisks, survival_times, event_indicators):
    """
    lrisks: log risks, B x 1
    survival_times: time bin, B
    event_indicators: event indicator, B x 1
    """    
    num_uncensored = torch.sum(event_indicators, 0)
    if num_uncensored.item() == 0:
        # return {'loss': torch.sum(lrisks) * 0}
        return torch.sum(lrisks) * 0
    
    if len(survival_times.shape)>1:
        survival_times = survival_times.squeeze(1)
    if len(event_indicators.shape)>1:
        event_indicators = event_indicators.squeeze(1)
    if len(lrisks.shape)>1:
        lrisks = lrisks.squeeze(1)

    sindex = torch.argsort(-survival_times)
    survival_times = survival_times[sindex]
    event_indicators = event_indicators[sindex]
    lrisks = lrisks[sindex]

    log_risk_stable = torch.logcumsumexp(lrisks, 0)

    likelihood = lrisks - log_risk_stable
    uncensored_likelihood = likelihood * event_indicators
    logL = -torch.sum(uncensored_likelihood)
    # negative average log-likelihood
    return logL / num_uncensored


class CoxLoss(nn.Module):
    """
    """
    def __init__(self):
        super().__init__()

    def __call__(self, logits, times, event_indicators):
        return partial_ll_loss(lrisks = logits, survival_times=times, event_indicators=event_indicators.float())
