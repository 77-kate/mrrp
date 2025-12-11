import torch
import torch.nn as nn

def partial_ll_loss(lrisks, survival_times, event_indicators):
    """
    lrisks: log risks, B x 1
    survival_times: time bin, B
    event_indicators: event indicator, B x 1
    """    
    num_uncensored = torch.sum(event_indicators, 0)
    if num_uncensored.item() == 0:
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
