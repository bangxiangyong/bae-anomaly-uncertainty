import torch
import torch.nn.functional as F
from torch.nn import Parameter
from bnn.bnn_utils import parse_architecture_string, compute_entropy
from torch.autograd import Variable
from torch.distributions import Categorical
import numpy as np

class ANN_BaseModule(torch.nn.Module):
    def __init__(self, input_size, output_size, architecture=["d1","d1"], activation="relu", use_cuda=False, init_log_noise=1e-3,
                 layer_type=torch.nn.Linear, log_noise_size=1, **kwargs):
        super(ANN_BaseModule, self).__init__()
        self.architecture = architecture
        self.use_cuda = use_cuda
        self.input_size = input_size
        self.output_size = output_size
        self.init_log_noise = init_log_noise

        #parse architecture string and add
        self.layers = self.init_layers(layer_type)
        self.log_noise_size=log_noise_size
        self.set_log_noise(self.init_log_noise,log_noise_size=log_noise_size)
        self.model_kwargs = kwargs

        #handle activation layers
        if isinstance(activation,str) and activation == "relu":
            self.activation = F.relu
        elif isinstance(activation,str) and activation == "tanh":
            self.activation = torch.tanh
        else:
            self.activation = activation

    def set_log_noise(self, log_noise,log_noise_size=1):
        if log_noise_size == 0: #log noise is turned off
            self.log_noise = Parameter(torch.FloatTensor([[0.]]),requires_grad=False)
        else:
            self.log_noise = Parameter(torch.FloatTensor([[np.log(log_noise)]*log_noise_size]))

    def init_layers(self, layer_type=torch.nn.Linear, architecture=None, input_size=None,output_size=None):
        #resort to default input_size
        if input_size is None:
            input_size = self.input_size
        if output_size is None:
            output_size = self.output_size

        #resort to default architecture
        if architecture is None:
            layers = parse_architecture_string(input_size,output_size, self.architecture, layer_type=layer_type)
        else:
            layers = parse_architecture_string(input_size,output_size, architecture, layer_type=layer_type)

        if self.use_cuda:
            layers = torch.nn.ModuleList(layers).cuda()
        else:
            layers = torch.nn.ModuleList(layers)
        return layers

    def forward(self,x):
        #apply relu
        for layer_index,layer in enumerate(self.layers):
            # if layer_index ==0 or (layer_index == (len(self.layers)-1)):
            if layer_index ==0:
                #first layer
                x = layer(x)
            else:
                #other than first layer
                x = layer(self.activation(x))
        return x

class BNN_BaseModel():
    """
    This is an absract base class which acts as a manager for the pytorch module `BNN_BaseModule` which exposes the fit and predict function.

    See `BAE_Ensemble_Manager` class
    """
    def __init__(self, task={"regression","classification"}, num_samples=50,num_epochs=100,
                 learning_rate=0.001, return_raw=False, return_unc=True, optimiser="Adam", unc_class="softmax", ret_kl_loss=False,
                 init_log_noise=1e-3, use_cuda=False, verbose=True, model_class: ANN_BaseModule = ANN_BaseModule, **kwargs):

        self.model_class = model_class
        self.model = 0
        self.num_epochs = num_epochs
        self.num_samples = num_samples
        self.learning_rate = learning_rate
        self.model_kwargs = kwargs
        self.return_raw = return_raw
        self.return_unc = return_unc
        self.unc_class = unc_class
        self.ret_kl_loss = ret_kl_loss
        self.init_log_noise = init_log_noise
        self.verbose = verbose
        self.loglik_losses, self.regulariser_losses, self.total_losses = [],[],[]

        self.use_cuda=use_cuda

        #set task
        if type(task) == set:
            task = "classification"
        if task == "classification":
            criterion = torch.nn.CrossEntropyLoss()
        elif task == "regression":
            criterion = torch.nn.MSELoss()
        self.criterion = criterion
        self.task = task
        self.optimiser_ = optimiser

    def instantiate_model(self,input_size,output_size, model_class=None, architecture=None, **kwargs):
        #handle model class
        if model_class is not None:
            temp_model_class = model_class
        else:
            temp_model_class = self.model_class

        #handle specific architecture
        if architecture is None:
            model = temp_model_class(input_size=input_size,output_size=output_size,  init_log_noise=self.init_log_noise, **self.model_kwargs, **kwargs)
        else:
            # model = temp_model_class(input_size=input_size,output_size=output_size, init_log_noise=self.init_log_noise, architecture=architecture, **self.model_kwargs, **kwargs)
            model = temp_model_class(input_size=input_size,output_size=output_size, init_log_noise=self.init_log_noise, architecture=architecture, **kwargs)

        #use cuda
        if self.use_cuda:
            return model.cuda()
        else:
            return model

    def fit(self, x, y=None):
        """
        Template for vanilla fitting, developers are very likely to override this to provide custom fitting functions.
        """
        #extract input and output size from data
        #and convert into tensor, if not already
        x = self.preprocess(x)
        input_size, output_size = self.get_input_output_size(x,y)
        x,y = self.convert_tensor(x,y)

        #instantiate model and optimiser
        self.model = self.instantiate_model(input_size,output_size)
        optimiser = self.get_optimiser()

        #train for n epochs
        for epoch in range(self.num_epochs):
            y_pred = self.model(x)
            loss = self.criterion(y_pred, y)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            if self.verbose:
                print(str(epoch)+" LOSS:"+str(loss.item()))
        return self

    def convert_tensor(self,x,y=None):
        if "Tensor" not in type(x).__name__:
            x = Variable(torch.from_numpy(x).float())
        if y is not None:
            if "Tensor" not in type(y).__name__:
                if self.task == "classification":
                    y = Variable(torch.from_numpy(y).long())
                else:
                    y = Variable(torch.from_numpy(y).float())

        #use cuda
        if self.use_cuda:
            if y is None:
                return x.cuda()
            else:
                return x.cuda(),y.cuda()
        else:
            if y is None:
                return x
            else:
                return x,y


    def get_input_output_size(self, x,y=None):
        input_size = x.shape[-1]
        if self.task == "classification" and y is not None:
            output_size = np.unique(y).shape[0]

        elif self.task =="regression" and y is not None:
            if len(y.shape) >= 2:
                output_size = y.shape[-1]
            else:
                output_size = 1
        else:
            output_size = input_size

        return input_size,output_size

    def get_optimiser(self,model=None):
        if model is None:
            model = self.model
        if self.optimiser_.lower() == "adam":
            optimiser = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        else:
            optimiser = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        return optimiser

    def preprocess(self,x):
        """
        Custom preprocessing of x before fitting/predicting
        """
        return x

    def predict(self, x):
        """
        By default, returns a tuple of (y_pred,y_unc) corresponding to the best estimate and uncertainty
        Has option to return raw predictions samples from the model has the following dimensions: (num_ensemble_samples, num_examples, output_size)

        Developer should override`predict_samples` to customise where the ensembles are obtained from.
        To customise the return of individual model, override `predict_sample_one` instead.
        Parameters
        ----------

        return_raw: bool
            If true, return raw samples to be processed by user.
        return_unc: bool
            If true, return uncertainty along with best estimates

        """
        x = self.preprocess(x)
        x = self.convert_tensor(x)

        y_pred_raw = self.predict_samples(x, num_samples=self.num_samples)

        #calculate best estimate and uncertainty
        if self.return_unc:
            y_pred, y_unc = self.compute_estimates(y_pred_raw,self.task,self.return_unc,self.unc_class)
        else:
            y_pred = self.compute_estimates(y_pred_raw,self.task,self.return_unc,self.unc_class)

        #return results
        if self.return_unc and self.return_raw:
            return y_pred, y_unc, y_pred_raw
        elif self.return_unc:
            return y_pred, y_unc
        else:
            return y_pred

    def predict_sample_one(self, model, x, proba=False,task="regression"):
        """
        Gets a single forward output of a model (x can be batched) which is assumed to be part of ensembles.
        Override this function to customise the return of the individual model.
        Used in conjunction with `predict_samples` to gather an ensemble of predictions.
        """
        output = model(x)

        #handle task type of classification and regression
        if task == "classification":
            if not proba:
                y_pred = F.softmax(output, dim=1)
                y_pred = torch.argmax(y_pred,dim=1)
            else:
                y_pred = F.softmax(output, dim=1)
        else:
            y_pred = output

        return y_pred

    def predict_samples(self, x, num_samples=100):
        """
        Gathers raw predictions from ensembles in an array via list comprehension
        Override this function to customise where the ensembles are obtained from.
        """
        y_preds_raw = [self.predict_sample_one(self.model, x, proba=True, task=self.task).detach().cpu().numpy() for sample in range(num_samples)]
        y_preds_raw = np.array(y_preds_raw)
        return y_preds_raw

    def compute_estimates(self,y_pred_raw,task="regression",return_unc=True,unc_class="softmax"):
        #calculate best estimate and uncertainty for classification and regression
        if task =="classification":
            #get the mean of the raw predictions from the BNN samples
            #Y predictions are the argmax of the last dimension,
            #and the uncertainty is the 1-probability for that prediction
            #hence uncertainty is ranging from 0-1 (0 being maximum uncertainty)
            y_pred_top =torch.topk(torch.from_numpy(y_pred_raw.mean(0)).float(),k=1,dim=-1)
            y_pred = y_pred_top[1].view(-1) #MLE
            y_pred = y_pred.detach().numpy()

            #uncertainty as softmax output
            if unc_class == "softmax":
                y_unc = y_pred_top[0].squeeze(-1)
                y_unc = (y_unc.detach().numpy()*-1)+1
            elif unc_class =="entropy":
                #uncertainty as entropy
                y_unc = np.apply_along_axis(compute_entropy,axis=1,arr=y_pred_raw.mean(0))

        else: #regression case
            y_pred = np.mean(y_pred_raw,axis=0)
            y_unc = np.std(y_pred_raw, axis=0)

        #return results in numpy
        if return_unc:
            return y_pred, y_unc
        else:
            return y_pred

    def regulariser_cost(self,model,p=1,penalty=0.1):
        lp_reg = Variable(torch.tensor(0.),requires_grad=True)
        for param in model.parameters():
            lp_reg=lp_reg+(param.abs().pow(p).sum()).pow(1/p)
        return (lp_reg*penalty)

    def log_gaussian_loss(self, y_pred, y_true, log_sigma):
        log_likelihood = ((-((y_true - y_pred)**2)*torch.exp(-log_sigma)*0.5)-0.5*log_sigma)
        return log_likelihood

    def log_likelihood_regression(self,y_train,y_pred,sigma):
        log_likelihood = self.log_gaussian_loss(y_pred,y_train,sigma)
        return log_likelihood

    def log_likelihood_classification(self,y_train,y_pred):
        eval_likelihood = Categorical(F.softmax(y_pred,dim=1)).log_prob(y_train)
        return eval_likelihood

    def eval_log_likelihood(self,y_train,y_pred,task="regression"):
        if task =="classification":
            eval_likelihood = self.log_likelihood_classification(y_train,y_pred)
        else:
            eval_likelihood = self.log_likelihood_regression(y_train,y_pred, self.model.log_noise)
        log_likelihood = eval_likelihood.sum()
        return log_likelihood

    def get_losses(self):
        return self.loglik_losses, self.regulariser_losses, self.total_losses
