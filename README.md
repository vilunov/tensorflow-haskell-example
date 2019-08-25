# Example project for TensorFlow with Haskell

## Running

### Cloning the repository

This project uses git submodules in order to fetch the necessary version of
TensorFlow bindings, which in turn fetches the TensorFlow library to build
itself. Be sure to run `git clone` with the `--recursive` option.

    git clone --recursive <repo_url>

### Building

The build depends on the libtensorflow, which is not included in the standard
TensorFlow distribution.

#### Using Stack with Docker support

This is the recommended variant.

Firstly, you will need to build the development image:

    docker build -t registry.gitlab.com/vilunov/tensorflow-haskell-example/environment:master docker/gpu

The `stack.yaml` is already configured to use this Docker image. You can invoke
all Stack commands on your machine.

#### Using Stack in Docker

TODO

#### Using Stack on host machine

TODO
