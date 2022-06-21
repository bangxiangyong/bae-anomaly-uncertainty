import torch
import numpy as np
from bnn.bnn_utils import convert_chol_tril
from bnn.base_class import ANN_BaseModule, BNN_BaseModel

class AutoEncoderModule(ANN_BaseModule):
    """
    Vanilla autoencoder
    """
    def __init__(self, **kwargs):
        super(AutoEncoderModule, self).__init__(**kwargs)
        self.output_size = self.input_size

class AutoEncoderModule_DiagCov_Combined(ANN_BaseModule):
    """
    Has double output heads, for mu and diagonal covariance
    """
    def __init__(self, bottleneck_layer=1, return_latent=True, **kwargs):
        super(AutoEncoderModule_DiagCov_Combined, self).__init__(**kwargs)
        self.output_size = self.input_size
        self.log_noise_size = self.output_size
        self.log_noise_layer = torch.nn.Linear(self.architecture[-1], self.log_noise_size)
        self.bottleneck_layer= bottleneck_layer
        self.return_latent = return_latent

    def forward(self,x):
        #apply relu
        latent_z = 0
        for layer_index,layer in enumerate(self.layers):
            if layer_index ==0:
                #first layer
                x = layer(x)
            elif layer_index == (len(self.layers)-1):
                log_noise = self.log_noise_layer(self.activation(x))
                x_ = layer(self.activation(x))
            else:
                #other than first layer
                x = layer(self.activation(x))
            if self.return_latent and self.bottleneck_layer == layer_index:
                latent_z = x.detach().cpu().numpy()

        #return latent
        if self.return_latent:
            return x_,log_noise,latent_z
        else:
            return x_,log_noise

class CholLinear(torch.nn.Module):
    """
    Assume the raw output of the dense layer is lower Chol triangular of a precision matrix.
    Note the diagonal is a logged version which will be exponentiated and upper will be zerorised to recover full L.
    Ultimately, it returns L, and log diagonal to calculate neg loglik later.
    """
    def __init__(self, input_size, output_size):
        super(CholLinear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.lower_non_diag_size = ((output_size**2)-output_size)/2
        self.chol_tril_layer = torch.nn.Linear(input_size, output_size+int(self.lower_non_diag_size))

    def batch_matrix_diag(self,mat):
        if mat.dim() == 2:
            mat_diag= mat.as_strided((mat.shape[0],mat.shape[1]), [mat.stride(0), mat.size(2) + 1])
        else:
            mat_diag= torch.diag(mat)
        return mat_diag

    def get_chol_tril_logdiag(self,dense_layer_output, diagonal_size):
        if dense_layer_output.dim() == 2:
            log_diag = dense_layer_output[:,0:diagonal_size]
            batch_size = dense_layer_output.shape[0]
            #now to recover L
            if next(self.parameters()).is_cuda:
                chol_tril = torch.zeros((batch_size,diagonal_size,diagonal_size)).cuda()
            else:
                chol_tril = torch.zeros((diagonal_size,diagonal_size))
            lii,ljj = np.tril_indices(chol_tril.size(-2), k=-1)
            dii,djj = np.diag_indices(chol_tril.size(-2))
            chol_tril[...,lii,ljj] = torch.exp(dense_layer_output[:,diagonal_size:])
            chol_tril[...,dii,djj] = torch.exp(log_diag)
        else:
            log_diag = dense_layer_output[0:diagonal_size]
            #now to recover L
            chol_tril = torch.autograd.Variable(torch.zeros((diagonal_size,diagonal_size)))
            tril_index = np.tril_indices(diagonal_size)
            diag_index = np.diag_indices(diagonal_size)
            chol_tril[tril_index] = torch.exp(dense_layer_output)
            chol_tril[diag_index] = torch.exp(log_diag)
        return chol_tril, log_diag

    def forward(self, x):
        chol_lower_tri_torch = self.chol_tril_layer(x)
        diagonal_size = self.output_size
        chol_trils, log_diags = self.get_chol_tril_logdiag(chol_lower_tri_torch,diagonal_size)
        return chol_trils, log_diags

class AutoEncoderModule_FullCov_Combined(ANN_BaseModule):
    """
    This combined network, aims to predict both the mu and Cholesky lower triangle matrix at one go.

    """
    def __init__(self, **kwargs):
        super(AutoEncoderModule_FullCov_Combined, self).__init__(**kwargs)
        self.output_size = self.input_size
        self.log_noise_size = self.output_size
        self.log_noise_layer = CholLinear(self.architecture[-1], self.log_noise_size)

    def forward(self,x):
        #apply relu
        for layer_index,layer in enumerate(self.layers):
            if layer_index ==0:
                #first layer
                x = layer(x)
            elif layer_index == (len(self.layers)-1):
                chol_tril,log_noise = self.log_noise_layer(self.activation(x))
                x_ = layer(self.activation(x))
            else:
                #other than first layer
                x = layer(self.activation(x))
        return x_,chol_tril,log_noise

class AutoEncoderModule_FullCov_Dedicated(ANN_BaseModule):
    """
    This separate network is dedicated only to predict the Cholesky lower triangle matrix of full precision (covariance) matrix.
    """
    def __init__(self, **kwargs):
        super(AutoEncoderModule_FullCov_Dedicated, self).__init__(**kwargs)
        self.output_size = self.input_size
        self.log_noise_size = self.output_size
        self.log_noise_layer = CholLinear(self.architecture[-1], self.log_noise_size)

    def forward(self,x):
        #apply relu
        for layer_index,layer in enumerate(self.layers):
            if layer_index ==0:
                #first layer
                x = layer(x)
            elif layer_index == (len(self.layers)-1):
                chol_tril,log_noise = self.log_noise_layer(self.activation(x))
            else:
                #other than first layer
                x = layer(self.activation(x))
        return chol_tril,log_noise


#auto-encoder with double decoder for mu (reconstructed signal) and log sigma (aleatoric)
class AutoEncoderModel(BNN_BaseModel):
    def __init__(self, mode="hetero",num_epoch=5000, bottleneck_layer=1, return_latent=True, num_epoch_mu=5000, num_epoch_sig=2000, architecture=[], architecture_mu=[], architecture_sig=[], use_cuda=False, anchored=False, **kwargs):
        super(AutoEncoderModel, self).__init__(**kwargs)
        self.use_cuda = use_cuda
        self.mode= mode
        self.base_mode = mode #used for switching evaluation of reconstruction loss
        self.return_unc = False
        self.return_raw = False
        self.architecture_mu = architecture_mu
        self.architecture_sig = architecture_sig
        self.architecture = architecture
        self.num_epoch= num_epoch
        self.num_epoch_mu=num_epoch_mu
        self.num_epoch_sig=num_epoch_sig
        self.model_class_mu = AutoEncoderModule
        self.anchored = anchored
        self.return_latent = return_latent
        self.bottleneck_layer = bottleneck_layer

        #set network architectures to dedicated or combined
        if len(architecture) ==0: #no default architecture supplied, set to dedicated mode with two networks
            if mode == "diag_cov":
                self.model_class_sig = AutoEncoderModule
            elif mode == "full_cov":
                self.model_class_sig = AutoEncoderModule_FullCov_Dedicated
        else: #set to combined mode, i.e single network with final layer outputs for mu and cov.
            if mode == "diag_cov":
                self.model_class = AutoEncoderModule_DiagCov_Combined
            elif mode == "full_cov":
                self.model_class = AutoEncoderModule_FullCov_Combined
            else: #vanilla
                self.model_class = AutoEncoderModule

    def log_prior_loss(self, model, mu=torch.Tensor([0.]), scale=torch.Tensor([1.])):
        #prior 0 ,1
        if self.anchored:
            mu = model.anchored_prior
            scale = torch.ones_like(mu)

        if self.use_cuda:
            mu=mu.cuda()
            scale=scale.cuda()

        weights = torch.cat([parameter.flatten() for parameter in model.parameters()])
        log_prior = (-((weights - mu)**2/scale*0.5)-torch.log(scale))
        return log_prior.sum()

    def init_anchored_weight(self, model):
        model_weights = torch.cat([parameter.flatten_torch() for parameter in model.parameters()])
        anchored_weights = torch.ones_like(model_weights)*model_weights.detach()
        return anchored_weights

    def fit_single_network(self,x, model, optimiser, mode, epoch):
        #train for n epochs
        #train mu autoencoder
        for epoch in range(epoch):
            neg_loglik_loss = self.reconstruction_loss(x,mode=mode)
            prior_loss = self.log_prior_loss(model)
            loss = neg_loglik_loss - prior_loss
            optimiser.zero_grad()
            loss.backward()
            print(str(epoch)+ "LOG LOSS:"+str(neg_loglik_loss.item())+
                  "PW LOSS:"+str(prior_loss.item())+
                  "AE LOSS:"+str(loss.item()))
            self.total_losses.append(loss.item())
            optimiser.step()
        return model

    def instantiate_model(self, input_size,output_size,model_class, architecture):
        model = super(AutoEncoderModel, self).instantiate_model(input_size,output_size,model_class, architecture)

        #for anchored priors
        if self.anchored:
            model.anchored_prior = self.init_anchored_weight(model)

        #for returning latent at bottleneck layer
        if self.bottleneck_layer>0:
            model.bottleneck_layer = self.bottleneck_layer
            model.return_latent = True
        else:
            model.return_latent = False

        return model

    def fit(self, x, y=None):
        #extract input and output size from data
        #and convert into tensor, if not already
        input_size, output_size = x.shape[-1], x.shape[-1]
        x= self.convert_tensor(x)

        #use cuda
        if self.use_cuda:
            x = x.cuda()

        #handle dedicated architectures
        if len(self.architecture) == 0:
            #instantiate model and optimiser
            self.model_mu = self.instantiate_model(input_size,output_size,self.model_class_mu, self.architecture_mu)
            optimiser_mu = self.get_optimiser(self.model_mu)

            #fit model_mu
            self.model_mu = self.fit_single_network(x, self.model_mu, optimiser_mu, "homo", self.num_epoch_mu)

            #fit model_sig, requires model_mu needs to be fitted first
            if self.mode == "diag_cov" or self.mode == "full_cov":
                self.model_sig = self.instantiate_model(input_size,output_size,self.model_class_sig, self.architecture_sig)
                optimiser_sig = self.get_optimiser(self.model_sig)
                self.model_sig = self.fit_single_network(x, self.model_sig, optimiser_sig, self.mode, self.num_epoch_sig)

        #combined architecture i.e 2 in 1
        else:
            self.model = self.instantiate_model(input_size,output_size,self.model_class, self.architecture)
            optimiser = self.get_optimiser(self.model)
            self.model = self.fit_single_network(x, self.model, optimiser, self.mode, self.num_epoch)
        return self

    def predict(self, x, mode=None):
        """
        Returns the predicted mu, covariance (full/diagonal), and the reconstruction loss.
        Mode can be set to 'full_cov' , 'diag_cov' or 'homo' depending on the required recon. loss evaluation.
        If the model is originally trained at a more umbrella level e.g 'full_cov', it can evaluate recon. loss at a lower level but
        not the other way round.
        """
        if mode is None:
            mode = self.mode

        results = self.reconstruction_loss(x,return_raw=True, mode=mode)
        loss = results['loss']
        y_pred = results['y_pred']
        y_log_noise = results['y_log_noise']
        chol_tril = results['chol_tril'] #need to convert this to covariance matrix
        latent_z = results['latent_z']

        #if chol_tril is not available, return diagonal of cov. matrix
        #otherwise, return chol_tril computed for batch
        if len(chol_tril) == 0:
            return y_pred, np.exp(y_log_noise), loss,latent_z
        else:
            y_cov = convert_chol_tril(chol_tril)
            y_cov = np.nan_to_num(y_cov)
            return y_pred, y_cov, loss,latent_z

    def reconstruction_loss(self,x,mode=None, return_raw=False):
        #handle default mode argument
        if mode is None:
            mode = self.mode

        #set default optional returns
        chol_tril =[]
        y_log_noise =[]
        latent_z = []
        #handle specific modes, followed by dedicated/combined architectures
        if mode == "diag_cov":
            if len(self.architecture) == 0:
                #dedicated
                y_pred = self.model_mu(x).detach()
                if self.base_mode == "full_cov":
                    _, y_log_noise = self.model_sig(x)
                else:
                    y_log_noise = self.model_sig(x)
                loss = -self.log_likelihood_regression(x,y_pred,y_log_noise).sum(1)
            else:
                #combined
                y_pred, y_log_noise, latent_z = self.model(x)
                loss = -self.log_likelihood_regression(x,y_pred,y_log_noise).sum(1)
        elif mode == "full_cov":
            #dedicated
            if len(self.architecture) == 0:
                y_pred = self.model_mu(x).detach()
                chol_tril,y_log_noise = self.model_sig(x)
                y = (x-y_pred)
            #combined
            else:
                y_pred, chol_tril,y_log_noise = self.model(x)
                y = (x-y_pred)
            loss = self.calc_chol_recon_loss_torch(y,chol_tril,y_log_noise)
        else: #mu or homoskedestic
            if len(self.architecture) == 0:
                y_pred = self.model_mu(x)
                y_log_noise = self.model_mu.log_noise
                loss = -self.log_likelihood_regression(x,y_pred,y_log_noise).sum(1)
            #combined mode
            else:
                if self.base_mode == "diag_cov":
                    y_pred,_,__ = self.model(x)
                    y_log_noise = self.model.log_noise
                else:
                    y_pred = self.model(x)
                    y_log_noise = self.model.log_noise
                loss = -self.log_likelihood_regression(x,y_pred,y_log_noise).sum(1)


        #handle return raw (detached) or loss only
        if return_raw:
            #detach and convert to numpy array, if exists
            if len(chol_tril) != 0:
                chol_tril = chol_tril.detach().cpu().numpy()
            if len(y_log_noise) != 0:
                y_log_noise = y_log_noise.detach().cpu().numpy()
            return {'loss':loss.detach().cpu().numpy(), 'y_pred':y_pred.detach().cpu().numpy(),
                    'chol_tril':chol_tril,
                    'y_log_noise':y_log_noise,
                    'latent_z':latent_z}
        else:
            return loss.sum()/(x.shape[0])

    def batch_matrix_diag(self,mat):
        mat_diag= mat.as_strided((mat.shape[0],mat.shape[1]), [mat.stride(0), mat.size(2) + 1])
        return mat_diag

    def calc_chol_recon_loss_torch(self,y, chol_lower_tri,log_noise):
        #handle batch size of 1 by unsqueezing them
        if len(y.shape) == 1:
            y = y.unsqueeze(0)
            chol_lower_tri = chol_lower_tri.unsqueeze(0)
            log_noise = log_noise.unsqueeze(0)

        #calculate reconstruction loss
        chol_y = torch.matmul(torch.transpose((chol_lower_tri),2,1),y.unsqueeze(-1))
        chol_recon_loss = torch.matmul(torch.transpose(chol_y,2,1),chol_y)
        chol_recon_loss = chol_recon_loss.view(-1,1)

        #calculate log determinant
        log_det = (-2*(log_noise.sum(-1))).view(-1,1)
        return chol_recon_loss + log_det
