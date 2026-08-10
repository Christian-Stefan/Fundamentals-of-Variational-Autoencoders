from utils.Code_for_students.imports_submission import *

# A Encoder
class Encoder(nn.Module):
    """
    Architecture Requirements for VAE:
  • The restriction on the Encoder and Decoder model no longer apply.
  • Only the last layers of the Encoder and the first layer of the Decoder can be a Linear layer.
  • Latent space changed to output two vectors: mean and (log) standard deviation (Size = 16).
  • Must contain a reparameterization function using only PyTorch operations.
  • Output 3 vectors: mean, log_std, and sampled.
    """
    def __init__(self, enc_arch:tuple, latent_dim:int=16):
        super(Encoder, self).__init__()
        # A1. Encoder sequentially-wrapped Convolutional architecture definition
        self.ENC = nn.Sequential()
        
        # A1. Build the downsampling layers, followed by BN, Tanh, and max pooling
        # A1.1. Extract the lists from the tuple
        channels = enc_arch[0]
        kernels = enc_arch[1]
        paddings = enc_arch[2]
        
        for layer in range(len(channels)-1):
            self.ENC.append(
                nn.Conv2d(in_channels=channels[layer], 
                          out_channels=channels[layer+1],
                          stride=2, 
                          kernel_size=kernels[layer], 
                          padding=paddings[layer])    
            )
            self.ENC.append(nn.BatchNorm2d(channels[layer+1]))
            self.ENC.append(nn.Tanh())
            
        # A2. Dynamic Shape Calculation
        dummy_img = torch.rand(1, 1, 32, 32)
        with torch.no_grad():
            dummy_out = self.ENC(dummy_img)
            self.spatial_shape = dummy_out.shape[1:] 
            self.flattened_size = dummy_out.view(1, -1).size(1) 
            
        # A3. The VAE Bottleneck (Linear mapping to Mean and Log Standard Deviation)
        # ... these by a Fully-Connected, or
        # ... its equivalent: Dense, layer. If necessary, you can use a Flatten and Activation layer
        # ... as well. 
        self.fc_mean = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logstd = nn.Linear(self.flattened_size, latent_dim)

    # A4. The Reparameterization Trick
    def reparameterize(self, mean, log_std):
        """ x_out = x_mean + n * exp(x_std) """
        n = torch.randn_like(input = log_std)
        return mean + n * torch.exp(log_std)

    def forward(self, x):
        x = self.ENC(x)                      # A5.1 Extract spatial features
        x = torch.flatten(x, start_dim=1)    # A5.2 Flatten to 1D
        mean = self.fc_mean(x)               # A5.3 Predict the mean
        log_std = self.fc_logstd(x)          # A5.4 Predict the log variance/std        
        sampled = self.reparameterize(mean, log_std) # A5.5 Sample a point

        return mean, log_std, sampled
    
# B  Decoder
class Decoder(nn.Module):
    """
    Architecture Requirements for VAE:
  • Must accept the sampled latent vector (Size = 16).
  • First layer must be a Linear layer to project it back to spatial dimensions.
  • Uses Upsampling/ConvTranspose to rebuild the image.
    """
    def __init__(self, dec_arch:tuple, spatial_shape:tuple, latent_dim:int=16):
        super(Decoder, self).__init__()
        
        # B1.1 Rebuild the Spatial Dimensions
        self.spatial_shape = spatial_shape
        flattened_size = spatial_shape[0] * spatial_shape[1] * spatial_shape[2]
        
        # B1.2 Projects the size-16 vector back into the size-256 vector
        self.fc_dec = nn.Linear(latent_dim, flattened_size)
        
        # B2.3 Decoder sequentially-wrapped architecture definition
        self.DEC = nn.Sequential()
        for layer in range(len(dec_arch[0])-1):
            self.DEC.append(
                nn.ConvTranspose2d(in_channels=dec_arch[0][layer], out_channels=dec_arch[0][layer+1],
                                   stride=1, kernel_size=dec_arch[1], padding=dec_arch[2])
            )
            
            # B2. 4
            if layer < len(dec_arch[0]) - 2:
                self.DEC.append(nn.BatchNorm2d(dec_arch[0][layer+1]))
                
            self.DEC.append(nn.Tanh())
            self.DEC.append(nn.UpsamplingBilinear2d(scale_factor=2))
        
    def forward(self, z):
        x = self.fc_dec(z)                              # B2.3. Project latent vector to flat spatial vector
        x = x.view(-1, *self.spatial_shape)             # B2.4 Reshape [Batch, Channels, H, W]
        r = self.DEC(x)                                 # B2.5. Upsample back to image
        return r
    
# %%  Variational Autoencoder Wrapper
class VAE(nn.Module):
    """
    Wraps the Encoder and Decoder together.
    """
    def __init__(self, enc_arch_l_k_p:tuple, dec_arch_l_k_p:tuple, latent_dim:int=16):
        super(VAE, self).__init__()
        self.enc_layers, self.enc_kernel_size, self.enc_padding = enc_arch_l_k_p
        self.dec_layers, self.dec_kernel_size, self.dec_padding = dec_arch_l_k_p

        # Initialize Encoder
        self.encoder = Encoder((self.enc_layers, self.enc_kernel_size, self.enc_padding), latent_dim)
        
        # Pass the dynamically calculated spatial shape from Encoder directly into Decoder
        self.decoder = Decoder((self.dec_layers, self.dec_kernel_size, self.dec_padding), 
                               spatial_shape=self.encoder.spatial_shape, 
                               latent_dim=latent_dim)
        
    def forward(self, x):
        mean, log_std, sampled = self.encoder(x)
        r = self.decoder(sampled) 
        
        return r, mean, log_std