# foundation-ts

Proyecto base para analizar series temporales con modelos de deep learning en PyTorch.

La base esta pensada para comparar experimentos de forma repetible y ordenada usando:

- `mamba` para gestionar el entorno.
- `hydra` para parametrizar ejecuciones.
- `pytorch` como framework de deep learning.

## Estructura

```text
.
├── configs/              # Configuracion Hydra
├── data/                 # Datos locales, no versionados
├── notebooks/            # Exploracion
├── outputs/              # Salidas de experimentos
├── src/foundation_ts/    # Codigo del proyecto
└── tests/                # Tests
```

## Entorno

```bash
mamba env create -f environment.yml
mamba activate foundation-ts
```

