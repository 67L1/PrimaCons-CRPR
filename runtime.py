import numpy as np
def forward_with_hidden(model, state, question):
    from jevk5.prompt import decision_options
    options = decision_options(question)
    option_text = [x[1] for x in options]
    ids = model.encode(state, question["instructions"], option_text)
    import torch
    input_ids = torch.tensor([ids], dtype=torch.long, device=model.device)
    with torch.inference_mode():
        output = model.model.model(input_ids=input_ids, use_cache=False, output_hidden_states=True, return_dict=True)
        hidden_states = output.hidden_states
        last = len(ids) - 1
        hs = {"penultimate": hidden_states[-2][0, last].detach().float().cpu().numpy(), "third_last": hidden_states[-3][0, last].detach().float().cpu().numpy()}
        # JevK5's official readout uses the final backbone hidden state. The
        # earlier layers are calibration features only.
        # `last_hidden_state` includes the model's final normalization and is
        # exactly what JevK5.runtime._slot_logits reads. hidden_states[-1] is
        # the pre-final-normalization layer and is unsuitable for parity.
        final_hidden_t = output.last_hidden_state[0, last]
        raw_logits_t = final_hidden_t @ model.slot_weight[:len(options)].T
        raw_logits = raw_logits_t.float().cpu().numpy()
        probs = np.exp(raw_logits / model.temperature - np.max(raw_logits / model.temperature))
        probs /= probs.sum()
    return ids, hs, probs, options