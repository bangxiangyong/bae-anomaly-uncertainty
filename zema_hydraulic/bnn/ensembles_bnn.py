import numpy as np
from bnn.autoencoder import AutoEncoderModel
from bnn.base_class import BNN_BaseModel

class BAE_Ensemble_Manager(BNN_BaseModel):
    def __init__(self,num_samples=5, use_cuda=False, model_class: BNN_BaseModel=AutoEncoderModel, **kwargs):
        super(BAE_Ensemble_Manager, self).__init__(**kwargs)
        self.num_samples = num_samples #number of ensembles to train
        self.model_class = model_class
        self.use_cuda = use_cuda

    def instantiate_model(self):
        return self.model_class(use_cuda=self.use_cuda, **self.model_kwargs)

    def fit_single_network(self, x, y=None):
        model = self.instantiate_model()
        model.fit(x,y)
        return model

    def fit(self, x, y=None):
        #extract input and output size from data
        #and convert into tensor, if not already
        self.input_size, self.output_size = self.get_input_output_size(x,y)
        self.x = self.convert_tensor(x)

        #instantiate model and optimiser
        self.model = [self.fit_single_network(x, y) for i in range(self.num_samples)]

        return self

    def predict_samples(self, x, num_samples=100, mode=None, return_latent=False):
        y_raw = [self.model[i].predict(x,mode=mode) for i in range(num_samples)]

        #extract the 3 columns
        y_preds_raw = np.array([y_raw[i][0] for i in range(num_samples)])
        y_cov_raw = np.array([y_raw[i][1] for i in range(num_samples)])
        recon_loss_raw = np.array([y_raw[i][2] for i in range(num_samples)])
        latent_z_raw = np.array([y_raw[i][3] for i in range(num_samples)])

        #compute best estimate and uncertainty
        y_preds_mu = y_preds_raw.mean(0)
        y_preds_std = y_preds_raw.std(0)

        y_cov_mu = y_cov_raw.mean(0)
        y_cov_std = y_cov_raw.std(0)

        recon_loss_mu = recon_loss_raw.mean(0)
        recon_loss_std = recon_loss_raw.std(0)

        latent_z_mu = latent_z_raw.mean(0)
        latent_z_std = latent_z_raw.std(0)

        #return a dictionary for easier interpretability
        return {'y_pred':(y_preds_mu,y_preds_std),
                'y_cov':(y_cov_mu,y_cov_std),
                'recon_loss':(recon_loss_mu,recon_loss_std),
                'latent_z':(latent_z_mu,latent_z_std),
                'raw':(y_preds_raw, y_cov_raw, recon_loss_raw)}

    def predict(self, x, mode=None, return_latent=False):
        x = self.convert_tensor(x)

        y_pred_dict = self.predict_samples(x, num_samples=self.num_samples, return_latent=return_latent, mode=mode)

        #dictionary which contains aggregated and raw results
        return y_pred_dict
