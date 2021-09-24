# Notes

## Convolutional NNs

- convolution layer: extract features using kernel/filter
- `stride`: number of pixels for kernel shift
- `padding`: added to border of image
  - idea: o/w, corner pixels will only be covered once, middle pixels will be covered multiple times
- sparse connections bc. of sliding window (size of window many input nodes are connected to the same output node)
- constant filter parameters: same weights for each set of nodes
- channels: multiple filters are used within the same layer, resulting in stacked outputs
  - each channel ends up being trained to detect a key feature in the image
  - output of convolutional layer for 2D images thus has 3 dimensions
- pooling layer: pre-processing (`nn.MaxPool2d`)
  - downscaling image obtained from previous layers
  - makes detection of certain features more invariant to scale and orientation changes
  - max, average, sum, ... 
- drop-out layer (`nn.Dropout()`): after convolutional, avoid overfitting
- fully connected layer: input from other layers is flattened into a vector (transforms into desired number of classes)

### how to choose hyperparameters

- kernel size
  - odd-sized filters symmetrically duvide the previous layer pixels around the output pixel, o/w distorsions
  - 3x3 is quite expensive, but most used
    - alternative: 1x3 followed by 3x1 (apparently quite cost-effective)
- output channels
  - larger number allows the layer to potentially learning more useful features

### encoder

- first layer
  - input channels: 3 (RGB)
  - output channels: 32 (seems to be a thing)
  - `kernel_size`: can do tuple for non-square kernels, o/w one number (e.g. 5)
  - padding: calculate from desired output width

Output size of convolutional filter/pooling in one dimension (width or height): $W_{out} = \frac{W_{in} - F + 2P}{S}+1$

- $W_{in}$: width of input
- $F$: filter size
- $P$: padding
- $S$: stride

Output size according to PyTorch docs: $W_{out} = \lfloor \frac{W_{in} + 2P - D*(F-1) -1 }{S} +1 \rfloor$

#### PlaNet original code

- filters (output channels): 32, 64, 128, 256
- kernel size: 4
- strides: 2
- no pooling layers...

#### sizes

- input 64x64
- after first layer: 32 channels of 31x31
  - $W_{out} = \lfloor \frac{64 + 2*0 - 1*(4-1) -1 }{2} +1 \rfloor = \lfloor \frac{64-3-1}{2} +1 \rfloor = \lfloor \frac{60}{2} + 1 \rfloor = 31$
- after second layer: 64 channels of 14x14
  - $W_{out} = \lfloor \frac{31 + 2*0 - 1*(4-1) -1 }{2} +1 \rfloor = \lfloor \frac{31-3-1}{2} +1 \rfloor = \lfloor \frac{27}{2}+1 \rfloor = 14$
- after third layer: 128 channels of 6x6
  - $W_{out} = \lfloor \frac{14 + 2*0 - 1*(4-1) -1 }{2} +1 \rfloor = \lfloor \frac{14-3-1}{2} +1 \rfloor = \lfloor \frac{10}{2}+1\rfloor = 6$
- after fourth layer: 256 channels of 2x2
  - $W_{out} = \lfloor \frac{6 + 2*0 - 1*(4-1) -1 }{2} +1 \rfloor = \lfloor \frac{6-3-1}{2}+1 \rfloor = \lfloor \frac{2}{2}+1 \rfloor = 2$
- fully connected: input 256x2x2 = 1024

## Recurrent NNs

Similar to LSTM, only without passing on the internal cell state (not only the hidden state).

### LSTM

- can process sequences of data
- LSTM unit is composed of cell, input gate, output gate, forget gate
  - cell remembers values
  - gates regulate information flow into and out of cell
- cell state encodes aggregation of data from all previous time steps
  - entire data so far
  - input data is filtered through input and forget gate, result of both is incorporated into cell state
  - gate weights extract features
  - forget weights: determine which time-steps are important or not
  - input weights: how to encode info into cell state
- hidden state encodes characterisation of previous time step's data
  - more concerned with the most recent timestep (not all)
  - can mean different things depending on the task
- two normallising functions: sigmoid and tanh
  - sigmoid: trying to calculate set of scalars by which to multiply something else
    - amplify/diminish
    - between 0 and 1
  - tanh: trying to transform data into normalised encoding of data
    - between -1 and 1
- gates use sigmoid functions, each followed by multiplication operation
  - example: forget gate outputs matrix of values close to 1
    - based on current input, the timeseries history is very important
    - cell continues to retain most of its original value
- "candidate gate": tanh
  - same matrix operations as in input gate (sigmoid)
  - output is encoded, normalised version of hidden state combined with current input
  - output is multiplied with input-gate output
  - does a bit of feature extraction
  - idea: presence of certain features can deem the current state important or unimportant
    - scaled by sigmoid based on what the data looks like before being added to the cell state ("remember-worthiness")
- output gate uses encoding and scaling to get incorporated into cell state and form output hidden state (can be used for prediction or fed back into LSTM)
- stacking LSTM layers: use hidden states as input to the next layer

#### PyTorch

structure: `nn.LSTM` + `nn.Linear` (`nn.LSTM` is a multi-ayer LSTM RNN)

For each element in input sequence, compute:

- $h_t, h_{t-1}$: hidden states
- $c_t$: cell state
- $x_t$: input
  - $x_t^{(l)} = h_t^{(l-1)} \cdot \delta_t^{(l-1)}$
  - $\delta_t^{(l-1)}$: dropout Bernoulli RV
- $i_t$: input gates
- $f_t$: forget gates
- $g_t$: cell gates
- $o_t$: output gates

Parameters:

- `input_size`
- `hidden_size`: number of features in hidden state
- `num_layers`: number of recurrent layers, default 1
- `dropout`: dropout probability, default 0

Inputs:

- `input`: tensor of shape $(L, N, H_{in})$
  - $L$: sequence length
  - $N$: batch size
  - $H_{in}$: `input_size`
- `(h_0, c_0)`: initial hidden state, initial cell state
  - for each element in the batch
  - defaults to zeros

Outputs:

- `output`
- `(h_n, c_n)`: final hidden state and cell state

### GRU

Similar to LSTM with forget gate, fewer parameters, no output gate. Does not maintain internal cell state, that info is incorporated in hidden state and then passed on to the next unit.

Gates: Update (=Output), Reset (=Forget+Input), Current Memory (sub-part of reset gate)
