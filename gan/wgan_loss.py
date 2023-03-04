import torch

def w_discriminator_loss(real, fake):
    
    loss = torch.mean(fake) - torch.mean(real)

    return loss

def w_generator_loss(fake):
    
    loss = -torch.mean(fake)
    
    return loss
