import torch
import torch.optim as optim
import numpy as np

# Hinge Loss
def hinge_loss_discriminator(real_scores, fake_scores):
    return torch.mean(torch.relu(1.0 - real_scores)) + torch.mean(torch.relu(1.0 + fake_scores))

def hinge_loss_generator(fake_scores):
    return -torch.mean(fake_scores)

# Gradient Penalty
def compute_gradient_penalty(discriminator, real_data, fake_data, conditions):
    alpha = torch.rand(real_data.size(0), 1).expand_as(real_data)
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates = interpolates.requires_grad_(True)
    scores = discriminator(interpolates, conditions)
    gradients = torch.autograd.grad(
        outputs=scores,
        inputs=interpolates,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradient_norm = gradients.norm(2, dim=1)
    return ((gradient_norm - 1) ** 2).mean()
