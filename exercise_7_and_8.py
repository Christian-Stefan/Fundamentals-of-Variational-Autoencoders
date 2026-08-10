from utils.Code_for_students.imports_submission import *
from utils.Code_for_students.autoencoder_submission import *
from utils.Code_for_students.utils_submission import *

def show_results_3_8_a():
            # 1. Network Inference
            # Put the network in evaluation mode
            weights = torch.load(r'utils\results\best_vae_weights_no_beta.pth')
            default_VAE = VAE(args.enc_arch_l_k_p, args.dec_arch_l_k_p, latent_dim=args.latent_dimension)
            default_VAE.load_state_dict(weights)
            default_VAE.eval() 
            with torch.no_grad():
                # 1.1 Import and Pass the NOISY examples through the autoencoder.
                # from Assignment3.Code_for_students.main_template import x_noisy_example, x_clean_example
                # ... and feed the network with it
                denoised_output, _, _ = default_VAE(x_noisy_example) 

            # 2. Conversion ('cpu' remains optional if cuda mode has not been engaged)
            noisy_imgs = x_noisy_example.cpu().squeeze().numpy()
            denoised_imgs = denoised_output.cpu().squeeze().numpy()
            clean_imgs = x_clean_example.cpu().squeeze().numpy()


            # 3. Create a 3x10 grid of subplots. 
            fig, axes = PLT.subplots(nrows=3, ncols=10, figsize=(20, 6))
            # 3.1 Populate the Grid
            for i in range(10):
                
                # ROW 1: Noisy Input
                axes[0, i].imshow(noisy_imgs[i], cmap='gray')
                axes[0, i].axis('off') # Hide the axis ticks/numbers
                if i == 0: 
                    axes[0, i].set_title("Noisy Input", fontsize=14, fontweight='bold', pad=10)
                
                # ROW 2: Autoencoder Output
                axes[1, i].imshow(denoised_imgs[i], cmap='gray')
                axes[1, i].axis('off')
                if i == 0: 
                    axes[1, i].set_title("AE Output", fontsize=14, fontweight='bold', pad=10)
                
                # ROW 3: Clean Ground Truth
                axes[2, i].imshow(clean_imgs[i], cmap='gray')
                axes[2, i].axis('off')
                if i == 0: 
                    axes[2, i].set_title("Clean Image", fontsize=14, fontweight='bold', pad=10)
            PLT.savefig(r'utils\results\Reconstruction_against_GT_Digit_Comparison_no_beta.png')
            PLT.show()

def show_results_3_8_b(reinforce_latent_space:bool=False):
            # 1.1 Loading the model's weights
            default_VAE = VAE(args.enc_arch_l_k_p, args.dec_arch_l_k_p, latent_dim=args.latent_dimension)
            default_VAE.load_state_dict(torch.load(r'utils\results\best_vae_weights_no_beta.pth'))
            # 1.2 Enabling the evaluation mode and 
            # ... ensuring that the weights `model.parameters` remain frozen `param.requires_grad=False`
            default_VAE.eval()
            for param in default_VAE.parameters():
                param.requires_grad = False
            print("\nStarting MAP Optimization ...")
            # 1.3 Initialize the lattent space and some other experiment variables by random estimation
            # ... define history holders
            if reinforce_latent_space:
                with torch.no_grad():
                    initial_guess = default_VAE.encoder(x_noisy_example)[0]
                estimated_latent = initial_guess.clone().detach().requires_grad_(True)     
                map_optim = torch.optim.Adam([estimated_latent], lr=0.01)
                num_iterations:int = 3000
                beta_map:float = 0.01
                map_loss_history:list = []
                label:str = 'enforced'
            
            else:
                estimated_latent = nn.Parameter(torch.randn(10,16))
                map_optim = torch.optim.Adam([estimated_latent], lr=1e-2)
                num_iterations:int = 3000
                beta_map = 0.01
                map_loss_history:list = []
                label:str = 'not_enforced'

            # 2. ========================================================
            # Shape the latent space based on the decoder loaded weights|
            #============================================================
            for i in range(num_iterations):
                # 2.1 Clean the gradient cache memory
                map_optim.zero_grad()
                # 2.2 Get an output based on the estimated_latent
                g_z = default_VAE.decoder(estimated_latent)
                
                # 2.3 Assemblying equation (4)
                # ... where fidelity terms represents the divergence between x_estimate and g(z_estimate)
                fidelity_term = torch.sum((x_noisy_example - g_z) ** 2)
                # ... and prior term stands for how much importance we attach to the estimated latent space;
                prior_term = beta_map * torch.sum(estimated_latent ** 2)
                map_loss = fidelity_term + prior_term
                # 2.4 Backprop, optimizer step and saving history;
                map_loss.backward()
                map_optim.step()
                map_loss_history.append(map_loss.item())
                if (i + 1) % 100 == 0:
                    print(f"Iteration {i+1}/{num_iterations} | MAP Loss: {map_loss.item():.2f}")

            print("MAP Optimization Complete!")


            # 3. Generate results
            with torch.no_grad():
                clean_estimates = default_VAE.decoder(estimated_latent).squeeze().numpy()
                
            x_noisy_np = x_noisy_example.squeeze().numpy()
            x_clean_np = x_clean_example.squeeze().numpy()

            PLT.figure(figsize=(8, 4))
            PLT.plot(map_loss_history, color='blue', linewidth=2)
            PLT.title("MAP Loss over Iterations (Equation 4)")
            PLT.xlabel("Iteration")
            PLT.ylabel("MAP Loss")
            PLT.grid(True)
            PLT.savefig(rf'utils\results\MAP_Loss_Over_iterations_{label}.png')
            PLT.show()

            # Plot 2: Image Grid
            fig, axes = PLT.subplots(nrows=3, ncols=10, figsize=(20, 6))
            for i in range(10):
                axes[0, i].imshow(x_noisy_np[i], cmap='gray')
                axes[0, i].axis('off')
                if i == 0: axes[0, i].set_title("Noisy Input", fontsize=14, fontweight='bold', pad=10)
                
                axes[1, i].imshow(clean_estimates[i], cmap='gray')
                axes[1, i].axis('off')
                if i == 0: axes[1, i].set_title("MAP Output", fontsize=14, fontweight='bold', pad=10)
                
                axes[2, i].imshow(x_clean_np[i], cmap='gray')
                axes[2, i].axis('off')
                if i == 0: axes[2, i].set_title("Clean Truth", fontsize=14, fontweight='bold', pad=10)

            PLT.tight_layout()
            PLT.savefig(rf'utils\results\ComparisonBetween_Noisy_MAPoutput_Clean_{label}.png')
            PLT.show()


if __name__ == "__main__":
    arg = argparse.ArgumentParser()     
    arg.add_argument("--load", action="store_true", help="load saved .tar files instead of training")
    arg.add_argument("--data_loc", type=str, default=r'Fundamentals-of-Variational-Autoencoders\Assignment3\Code_for_students\DataLoc', help="Data location path")
    arg.add_argument("--batch_size", type=int, default=120)
    arg.add_argument("--no_epochs", type=int, default=30)
    arg.add_argument("--learning_rate", type=float, default=0.001)
    arg.add_argument("--latent_dimension", type=int, default=16, help="Dimension of the bottleneck")
    arg.add_argument("--enc_arch_l_k_p", type=tuple, default=([1, 32, 64, 128, 256], [5, 3, 3, 3], [2, 1, 1, 1]) , help="Encoder architecture")
    arg.add_argument("--dec_arch_l_k_p", type=tuple, default=([256, 128, 64, 32, 1], 3, 1), help="Encoder architecture")
    arg.add_argument("--beta", type=float, default= 0.5, help="Weightinf factor used to balance KL and RECLoss in VAE custom loss")

    args = arg.parse_args()
    
    if args.load:
        print("==============================Loading Exercise 3.8.a) in 'load' mode==============================")
        print("Loading Weights for AE to generate results 3.8.a)")
        show_results_3_8_a()

        print("==============================Loading Exercise 3.8.b) in 'load' mode==============================")
        print("Loading Weights for AE to generate results 3.8.b)")
        show_results_3_8_b()
        print("==============================Loading Weights for AE to generate results 3.8.c)==============================")
        show_results_3_8_b(True)

    else:
        print("==============================Loading Exercise 3.8.a) in without 'load' mode==============================" )
        print("Without --load argument 'manual mode' will be engaged, meaning no weights or pre-trained model will be used but trained from scratch. After training (that might take up a bit of time) all the results will be generated based on the previously trained instance")
        # Informational statement describing the list of equipped hyperparameters and default experimental settings;
        print("List of hyperparameters: Dec Arch {}| Enc Arch {}| Beta factor {}| Latent dimension {}\nList of experimental settings: Batch Size {}| Learning rate {}| No. of epochs {}".format(
    args.dec_arch_l_k_p, args.enc_arch_l_k_p, args.beta, args.latent_dimension, args.batch_size, args.learning_rate, args.no_epochs
))

        # 1. Loading the data and initializing the model, the optimizer, its scheduler and history holder containers;
        train_loader, val_loader, test_loader = mini_Pipeline(batch_size=args.batch_size, data_loc=args.data_loc)
        default_VAE = VAE(args.enc_arch_l_k_p, args.dec_arch_l_k_p, latent_dim=args.latent_dimension)
        optim = torch.optim.AdamW(default_VAE.parameters(), lr=args.learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 
                                                            mode='min', 
                                                            factor=0.5, 
                                                            patience=2)

        train_history:list = []
        val_history:list = []
        best_val_loss = float('inf') # Tracker for the best validation loss
        print("Starting VAE Training...")
        for epoch in range(args.no_epochs):
        # 2. Training Starts ========
        #                           |
        # ===========================
            default_VAE.train() # 2.1 Initializing the training mode;
            batch_cum_tr_loss:float = 0.0 # 2.2 During-epoch loss holder variable
            for batch_idx, (x_clean, x_noisy, label) in enumerate(tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.no_epochs} [Train]")):
                optim.zero_grad() # 2.3 Reset cache gradient memotry
                _tr_output, mean, log_std = default_VAE(x_clean) # 2.4 Returns reconstruction, mean and variance
                _tr_loss, r_loss, kl_loss = vae_loss(_tr_output, x_clean, mean, log_std, None) # 2.5 The loss quantification is based on VAELoss (KL+RecError); No beta provided so we assume equal weigthing;
                _tr_loss.backward() # 2.6 Backprop
                optim.step() # 2.7 Optimizer step

            # 2.8 Iteratively saving the loss, averaging it over the number of batches and appending the estimate to the container;    
                batch_cum_tr_loss += _tr_loss.item()
            avg_tr_loss = batch_cum_tr_loss / len(train_loader)
            train_history.append(avg_tr_loss)
        # 3. Validation Starts =====================
        # Same steps as before are executed        |
        # apart excluding back-prop and optim-step |
        # ==========================================
            default_VAE.eval() 
            batch_cum_val_loss:float = 0.0
            with torch.no_grad(): 
                for batch_idx, (x_clean, x_noisy, label) in enumerate(tqdm.tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.no_epochs} [Valid]")):
                    _val_output, mean, log_std = default_VAE(x_clean)
                    _val_loss, r_loss, kl_loss = vae_loss(_val_output, x_clean, mean, log_std, None)
                    batch_cum_val_loss += _val_loss.item()

            avg_val_loss = batch_cum_val_loss / len(val_loader)
            val_history.append(avg_val_loss)
            
            scheduler.step(avg_val_loss)
            current_lr = optim.param_groups[0]['lr']

            print(f"Epoch {epoch+1} Summary --> Train Loss: {avg_tr_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Learning Rate: {current_lr}")
        # 3. Model saving ==============
        # Save both, the best and final|
        # =============================
            # 3.1 Keeping the track of the lowest validation loss ever recorded
            if avg_val_loss < best_val_loss:
                # New best was found and therefore this particular variant of the model must be saved
                best_val_loss = avg_val_loss
                # ... state_dict() extracts just the learned numbers (weights/biases), not the whole class
                torch.save(default_VAE.state_dict(), r'utils\results\best_vae_weights_no_beta.pth')
                print(f"*** New best model saved! (Val Loss: {best_val_loss:.4f}) ***")

        # Save the absolute final epoch's weights just in case you want to compare
        torch.save(default_VAE.state_dict(), 'final_vae_weights_no_beta.pth')
        print("Training Complete! Models safely saved to disk.")
        show_results_3_8_a()


        print("==============================Loading Weights for AE to generate results 3.8.b)==============================")
        show_results_3_8_b()
        print("==============================Loading Weights for AE to generate results 3.8.c)==============================")
        show_results_3_8_b(True)