# MNIST-NN-from-scratch
In this notebook, I implemented a simple two-layer neural network and trained it on the MNIST digit recognizer dataset.

->Forward Propagation

Z[1] = W[1]X + b[1]

A[1] = gReLU(Z[1])

Z[2] = W[2]A[1] + b[2]

A[2] = gsoftmax(Z[2])

->Backward Propagation

dZ[2] = A[2] - Y

dW[2] = (1/m) * dZ[2] * A[1]^T

dB[2] = (1/m) Σ dZ[2]

dZ[1] = W[2]^T * dZ[2] . g'1*

dW[1] = (1/m) * dZ[1] * A[0]^T

dB[1] = (1/m) Σ dZ[1]

->Parameter Updates

W[2] := W[2] - α dW[2]

b[2] := b[2] - α dB[2]

W[1] := W[1] - α dW[1]

b[1] := b[1] - α dB[1]

