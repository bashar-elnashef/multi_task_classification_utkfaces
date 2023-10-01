import torch.nn as nn

loss_fn_age = nn.CrossEntropyLoss()
loss_fn_gen = nn.BCELoss()
loss_fn_rac = nn.CrossEntropyLoss()
sig = nn.Sigmoid()

def multi_head_classifications_loss(age_out, gen_out, rac_out, age_tar, gen_tar, rac_tar):
    loss_age = loss_fn_age(age_out, age_tar)
    loss_gen = loss_fn_gen(sig(gen_out), gen_tar.unsqueeze(1).float())
    loss_rac = loss_fn_rac(rac_out, rac_tar)
    return loss_age, loss_gen, loss_rac
