Add repo to Spack:

```
git clone https://github.com/teonnik/cosma_spack.git
spack repo add cosma_spack
```

Install cosma:

```
spack install cosma
```

with GPU support:

```
spack install cosma +gpu
```

in debug mode (default is release):

```
spack install cosma +gpu build_type=Debug
```

For more information on Spack: [Spack 101 Tutorial](https://spack.readthedocs.io/en/latest/tutorial.html).

