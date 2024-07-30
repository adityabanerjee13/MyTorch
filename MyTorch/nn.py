
import numpy as np
import matplotlib.pyplot as plt




##########################################################################################
                                # Peripheral functions #
##########################################################################################




def pearson_plot(Y_hat, Y_test, dim = 3):
    """
    input:- Y_hat, Y_test
    output:- pearson correlation coefficient plot
    """
    if dim == 1:
        Y1 = Y_test
        Y1_hat = Y_hat
        correlation_coefficient = np.corrcoef(Y1_hat, Y1)
        print(f"Pearson correlation coefficient for y1: ")
        print(correlation_coefficient[0,1])
        fig, axs = plt.subplots(1, 1, figsize=(10, 5))
        axs.scatter(Y1, Y1_hat, c='blue', marker='o', label='Data Points', edgecolors=[0,0,0])
        plt.xlabel("y1")
        plt.ylabel("y1_hat")
        plt.show()
        return

    Y1_hat = Y_hat[:,0]
    Y1 = Y_test[:,0]
    Y2_hat = Y_hat[:,1]
    Y2 = Y_test[:,1]
    Y3_hat = Y_hat[:,2]
    Y3 = Y_test[:,2]
    x = np.linspace(-20, 20, 100)
    correlation_coefficient = np.corrcoef(Y1_hat, Y1)
    print(f"Pearson correlation coefficient for y1: ")
    print(correlation_coefficient[0,1])
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    m = correlation_coefficient[0,1]
    axs.grid(True)
    axs.plot(x, m*x, label='Data Points')
    axs.plot(x, x, label="x = y")
    axs.scatter(Y1, Y1_hat, c='blue', marker='o', label='Data Points', edgecolors=[0,0,0])
    plt.xlabel("y1")
    plt.ylabel("y1_hat")
    plt.show()

    x = np.linspace(-30, 40, 100)
    correlation_coefficient = np.corrcoef(Y2_hat, Y2)
    print(f"Pearson correlation coefficient for y2: ")
    print(correlation_coefficient[0,1])
    m = correlation_coefficient[0,1]
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    axs.grid(True)
    axs.plot(x, m*x, label='Data Points')
    axs.plot(x, x, label="x = y")
    axs.scatter(Y2, Y2_hat, c='blue', marker='o', label='Data Points', edgecolors=[0,0,0])
    plt.xlabel("y2")
    plt.ylabel("y2_hat")
    plt.show()

    x = np.linspace(-30, 60, 100)
    correlation_coefficient = np.corrcoef(Y3_hat, Y3)
    print(f"Pearson correlation coefficient for y3: ")
    print(correlation_coefficient[0,1])
    m = correlation_coefficient[0,1]
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    axs.grid(True)
    axs.plot(x, m*x, label='Data Points')
    axs.plot(x, x, label="x = y")
    axs.scatter(Y3, Y3_hat, c='blue', marker='o', label='Data Points', edgecolors=[0,0,0])
    plt.xlabel("y3")
    plt.ylabel("y3_hat")

    plt.show()

def pad_image(image, padding_size):
  # print("pad", image.shape)
  if len(image.shape) == 4:
    padded_image = np.pad(image, ((0,0), (0,0), (padding_size, padding_size), (padding_size, padding_size)), mode='constant')
  # if len(image.shape) == 3:
  #   padded_image = np.pad(image[0], ((0,0), (padding_size, padding_size), (padding_size, padding_size)), mode='constant')
  if len(image.shape) == 2:
    padded_image = np.pad(image, ((padding_size, padding_size), (padding_size, padding_size)), mode='constant')
  # print("pad", padded_image.shape)
  return padded_image

def unpad_image(image, pad_width):
  # print("pad", image.shape)
  _, _, padded_height, padded_width = image.shape
  image_height = padded_height - 2 * pad_width
  image_width = padded_width - 2 * pad_width
  return image[:, :, pad_width:image_height + pad_width, pad_width:image_width + pad_width]

class Stride:
    def __init__(self, stride=1):
        self.f = stride
        self.input_shape = None

    def downsample(self, x):
        self.input_shape = x.shape
        # print("Stride:",self.input_shape)
        return x[:, :, ::self.f, ::self.f]

    def Upsample(self, y):
        upsampled_array = np.zeros(self.input_shape)
        upsampled_array[:, :, ::self.f, ::self.f] = y
        return upsampled_array

def onehot(y, lent):
  y_oh = np.zeros((len(y), lent), float)
  y_oh[np.arange(len(y)), y] = 1
  return y_oh

def rotate_matrix(matrix):
  # Reverse the matrix along both axes.
  return np.flipud(np.fliplr(matrix))

def conv3D(img_array, kernel, stride, pad = 0):

  # # padding
  # if pad!=0:
  #   img_array = pad_image(img_array, pad)

  c, k, _ = kernel.shape
  c, m, n = img_array.shape

  # convolution operation
  count=0
  for i in range(c):
    ker = kernel[i]
    img = img_array[i]
    fr_image = np.fft.fft2(img)
    fr_kernel = np.fft.fft2(ker, s=img.shape)
    cc = np.real(np.fft.ifft2(fr_image * fr_kernel))
    if count==0:
      out_img = cc.copy()
      count+=1
    else:
      out_img += cc
  out_img = out_img[k-1:, k-1:]

  return out_img

def W_conv3D(Y, X, stride , pad = 0):

  m_x, n_x = X.shape
  m, n = Y.shape

  # convolution operation in eq 5
  fr_Y = np.fft.fft2(Y, s=X.shape) # rotate_matrix
  fr_x_i = np.fft.fft2(X)
  cc = np.real(np.fft.ifft2((fr_Y) * fr_x_i))

  # k = m_x-m+1
  # W = cc[:k, :k] # rotated formula
  k = (m-1)
  W = cc[k:, k:] # conv formula
  # print("W",W.shape, cc.shape)

  return W

def full_conv3D(Y, W, stride , pad = 0):

  input_height, input_width = Y.shape
  kernel_height, kernel_width = W.shape

  # Pad the input data
  padded_Y = np.pad(Y, kernel_height - 1, 'constant', constant_values=0)

  # convolution operation as in eq 6
  fr_Y = np.fft.fft2(padded_Y)
  fr_W = np.fft.fft2(rotate_matrix(W), s=padded_Y.shape)
  out_X = (np.real(np.fft.ifft2(fr_Y * fr_W)))[kernel_height - 1:, kernel_width - 1:]

  return out_X







##########################################################################################
                                # Loss functions #
##########################################################################################

class MSE:
    def __init__(self):
        self.yh = None
        self.yt = None

    def evaluate(self, y_hat, y_true):
        self.batch = y_hat.shape[0]
        self.yh = y_hat
        self.yt = y_true
        return np.mean(np.linalg.norm(y_hat-y_true, axis=0)**2)/2

    def grad(self):
        return (self.yh-self.yt)/self.batch

class CrossEntropyLoss:
    def __init__(self):
        self.predictions = None
        self.targets = None

    def evaluate(self, predictions, targets):
        self.predictions = predictions
        if len(targets.shape)>1:
          self.targets = targets
        else:
          self.targets = onehot(targets, predictions.shape[1])
        # Clip values to avoid numerical instability
        epsilon = 1e-10
        clipped_predictions = np.minimum(np.maximum(predictions, epsilon), 1-epsilon)
        loss = -np.sum(self.targets * np.log(clipped_predictions)) / len(targets)
        return loss

    def grad(self):
        epsilon = 1e-10
        clipped_predictions = np.minimum(np.maximum(self.predictions, epsilon), 1-epsilon)
        return -(self.targets/clipped_predictions) / len(self.targets)






##########################################################################################
                                # Activation functions #
##########################################################################################


class Softmax:
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True))
        probabilities = exp_values / np.sum(exp_values, axis = 1, keepdims = True)
        # tmp = np.exp(input)
        self.output = probabilities
        return self.output

    def backward(self, output_gradient, learning_rate):
        b = self.output.shape[0]
        n = self.output.shape[1]
        repeated_vector = np.repeat(self.output[:,  np.newaxis, :], n, axis=1)
        return np.array([np.dot((np.identity(n) - repeated_vector[i]) * repeated_vector[i].T, output_gradient[i]) for i in range(b)])

class Sigmoid:
    def __init__(self):
        self.output = None

    def forward(self, input_data):
        self.output = 1 / (1 + np.exp(-input_data))
        return self.output

    def backward(self, grad_output, learning_rate):
        return grad_output * self.output * (1 - self.output)

class ReLU:
    def __init__(self):
        self.input = None

    def forward(self, input_data):
        self.input = input_data
        return np.maximum(0, input_data)

    def backward(self, grad_output, learning_rate):
        grad_input = grad_output.copy()
        grad_input[self.input <= 0] = 0
        return grad_input




##########################################################################################
                                # Linear Layers #
##########################################################################################



class linearLayer():
    def __init__(self, in_layer, out_layer, output_layer=False, acti = None):
        self.in_layer = in_layer
        self.out_layer = out_layer
        self.output_layer = output_layer
        self.acti = acti
        self.weights = np.random.randn(out_layer, in_layer)/in_layer
        self.bias = np.random.randn(out_layer,1)
        self.grd_f_k = None

    def forward(self, x):
        self.b = x.shape[0]
        self.x = x
        self.y = np.dot(self.weights, x.T) + self.bias
        return self.y.T

    def backward(self, grad_y, learning_rate, lamda_l2=0, lamda_l1=0):
        grad_b = np.mean(grad_y, axis=0)
        if self.b > 1:
            grad_W = np.dot(grad_y.T, self.x)
        else:
            grad_W = np.outer(grad_y, self.x)
        # Regularization terms
        l1_reg_term = 0
        if lamda_l1 != 0:
            l1_reg_term = lamda_l1 * np.sign(self.weights)
        l2_reg_term = 0
        if lamda_l1 != 0:
            l2_reg_term = lamda_l2 * self.weights

        # Update weights with regularization terms
        self.weights -= (grad_W + l1_reg_term + l2_reg_term) * learning_rate
        self.bias -= grad_b.reshape(grad_b.shape[0], 1) * learning_rate

        return np.dot(self.weights.T, grad_y.T).T



##########################################################################################
                                # Convolutional Layers #
##########################################################################################



class Conv2D():
    def __init__(self, in_chan, out_chan, size, stride=1, padding=0, acti=None, l1_reg=0.0, l2_reg=0.0):
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.size = size
        self.stride_ = Stride(stride=stride)
        self.acti = acti
        self.padding = padding
        self.W = np.random.randn(out_chan, in_chan, size, size) / size ** 2
        self.bias = np.random.randn(out_chan, size, size)
        self.f_k_1 = 0
        self.f_k = 0
        self.grd_f_k = 0
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg

    def forward(self, x):
        self.input_shape = x.shape
        self.batch = x.shape[0]
        self.chan = x.shape[1]
        out_shape = int((self.input_shape[2] - self.size + 2*self.padding) + 1)
        Y_shape = (self.batch, self.out_chan, out_shape, out_shape)
        Y = np.zeros(Y_shape)

        # pad
        if self.padding != 0:
            x = pad_image(x, self.padding)

        self.X_padded = x
        for b in range(self.batch):
            a = np.zeros((self.out_chan, out_shape, out_shape))
            for i in range(self.out_chan):
                a[i] += conv3D(x[b], self.W[i], self.stride_, pad = self.padding) #+ self.bias
            Y[b] += a

        # stride
        if self.stride_.f != 1:
            Y = self.stride_.downsample(Y)

        return Y

    def backward(self, grad_Y, learning_rate):
        grad_W = np.zeros_like(self.W)
        grad_X = np.zeros_like(self.X_padded)

        # stride Upsample
        if self.stride_.f != 1:
            grad_Y = self.stride_.Upsample(grad_Y)

        for b in range(self.batch):
            for i in range(self.out_chan):
                for j in range(self.in_chan):
                    grad_W[i, j] += W_conv3D(grad_Y[b, i], self.X_padded[b, j], stride=self.stride_, pad = self.padding)
                    grad_X[j] += full_conv3D(grad_Y[b, i], self.W[i, j], stride=self.stride_, pad = self.padding)

        # unpad grad x
        if self.padding != 0:
            grad_X = unpad_image(grad_X, self.padding)

        # Regularization terms
        l1_reg_term=0
        if self.l1_reg != 0:
            l1_reg_term = self.l1_reg * np.sign(self.weights_kernel)

        l2_reg_term=0
        if self.l2_reg != 0:
            l2_reg_term = self.l2_reg * self.weights_kernel

        self.W -= (grad_W + l1_reg_term + l2_reg_term) * learning_rate
        # self.bias = -np.mean(grad_Y, axis = 0) * learning_rate + self.bias

        return grad_X




##########################################################################################
                                # Multihead Attention #
##########################################################################################




# class MultiHeadAttention:
#     def __init__(self, d_model, num_heads):
#         super(MultiHeadAttention, self).__init__()
#         assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.d_k = d_model // num_heads
#         self.W_q = linearLayer(d_model, d_model)
#         self.W_k = linearLayer(d_model, d_model)
#         self.W_v = linearLayer(d_model, d_model)
#         self.W_o = linearLayer(d_model, d_model)

#     def scaled_dot_product_attention(self, Q, K, V, mask=None):
#         attn_scores = np.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
#         if mask is not None:
#             attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
#         attn_probs = np.softmax(attn_scores, dim=-1)
#         output = np.matmul(attn_probs, V)
#         return output
# shape = (20, 3, 9, 9)
# b = BatchNorm1d(shape)
# x = np.random.randint(1, 1000, shape)
# print(b.forward(x))




##########################################################################################
                                # Others #
##########################################################################################





class Reshape:
    def __init__(self):
        self.ip_shape = None

    def forward(self, x):
        a = x.shape
        self.ip_shape = a
        return x.reshape(a[0], -1)

    def backward(self, x1, learning_rate):
        return x1.reshape(self.ip_shape[0], self.ip_shape[1], self.ip_shape[2], self.ip_shape[3])

class MaxPool2D:
    def __init__(self, pool_size=2, stride=2):
        assert pool_size==stride, "pooling size and stride must be equal."
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, input_data):
        self.input_data = input_data
        b, c, self.h, self.w = input_data.shape
        # Calculate output dimensions
        out_height = (self.h - self.pool_size) // self.stride + 1
        out_width = (self.w - self.pool_size) // self.stride + 1
        # Reshape input data to allow vectorized operations
        reshaped_input = input_data[:, :, :out_height*self.pool_size, :out_width*self.pool_size].reshape(b, c, out_height, self.pool_size,
                                            out_width, self.pool_size)
        # Max pooling operation using NumPy's max function
        pooled_output = np.max(reshaped_input, axis=(3, 5))
        # Create a mask for the max values
        self.pooling_mask = np.zeros_like(self.input_data)
        self.pooling_mask[np.where(input_data[:, :, :out_height*self.pool_size, :out_width*self.pool_size] \
                                  == np.kron(pooled_output, np.ones((self.pool_size, self.pool_size))))] = 1
        return pooled_output

    def backward(self, grad_output, learning_rate):
        grad_input = np.zeros_like(self.input_data)
        # Reshape input data and grad_output to allow vectorized operations
        _, _, out_h, out_w = grad_output.shape
        expand_hight = out_h*self.pool_size
        expand_width = out_w*self.pool_size
        # Reshape input data and grad_output to allow vectorized operations
        grad_input[:, :, :expand_hight, :expand_width] = np.kron(grad_output, np.ones((self.stride, self.stride))) \
                                                          *self.pooling_mask[:, :, :expand_hight, :expand_width]
        return grad_input

class BatchNorm1d:
    def __init__(self, num_features, train=False, eps=1e-5, momentum=0.1):
        self.num_features = num_features
        self.train = train
        self.eps = eps
        self.momentum = momentum
        
        # Initialize parameters
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
  
    def forward(self, x):
        if self.train:
            # Calculate batch statistics
            mean = x.mean(dim=0)
            var = x.var(dim=0)
            # Update running statistics with momentum
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            # Normalize the input
            x_norm = (x - mean) / np.sqrt(var + self.eps)
            
            # Scale and shift
            out = self.gamma * x_norm + self.beta
        else:
            # Calculate batch statistics
            mean = x.mean(axis=0)
            var = x.var(axis=0)
            self.const_var = 1 / np.sqrt(var + self.eps)
            # During inference, use running statistics for normalization
            self.x_norm = (x - mean)
            out = self.gamma * x_norm + self.beta
        
        return out
  
    def backprop(self, grad_y, learning_rate):
        self.gamma -= (self.x_norm * grad_y) * learning_rate
        self.beta -= (grad_y) * learning_rate
        grad_x = self.const_var*self.gamma*grad_y
        return grad_x
