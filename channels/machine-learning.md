# Machine learning

The parts of a neural network, each written out and tested: forward passes, the backward passes that compute their gradients, and the optimizers that use those gradients. No machine-learning framework is used.

Every backward pass is checked against central differences of its own forward pass before anything uses it. The forward and backward passes are in `lib/nn`. The Newton-Schulz orthogonalization Muon uses is in `lib/polar`, next to the classical Newton-Schulz polynomial for comparison.

## Transformers
- a-transformer-trained-by-muon-in-this-tab — one attention head, 336 parameters, trained in the browser from chance-level loss to a three-thousandth of it

## Optimizers
- muon-orthogonalizes-the-update — what Muon does to a gradient's singular values, and a race against SGD, momentum and Adam
