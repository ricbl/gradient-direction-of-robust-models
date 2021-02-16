#taken and modified from https://raw.githubusercontent.com/sky4689524/DefenseGAN-Pytorch/master/util_defense_GAN.py

import torch

def get_z_sets(model, y, data, n_classes, lr,norm_penalty_lambda, diff_loss, z_init, rec_iter = 200):
    model.eval()
    
    z_hat = z_init.clone()
    z_hat = z_hat.detach().requires_grad_()
    optimizer = torch.optim.Adam([z_hat], lr = lr)
    scaler =torch.cuda.amp.GradScaler()
    
    # iterations optimizing z_hat
    for iteration in range(rec_iter):            
        z_hat.grad = None
        with torch.cuda.amp.autocast():
            fake_image = model(z_hat, y)
            fake_image = fake_image.view(-1, data.size(1), data.size(2), data.size(3))
            reconstruct_loss = diff_loss(fake_image, data.detach()).mean()
            
            # L_proj, as proposed in Equation 15
            # measuring the distance between generated image fake_image 
            # and original image data
            # and the likelihood of the current z_hat
            reconstruct_loss = reconstruct_loss + norm_penalty_lambda * diff_loss(z_hat,torch.zeros_like(z_hat)).mean()
        
        #optimize using autocast syntax
        scaler.scale(reconstruct_loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    return z_hat.detach().clone()
