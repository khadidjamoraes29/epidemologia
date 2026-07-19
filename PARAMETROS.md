# Parâmetros do modelo

## Modelo compartimental

| Parâmetro | Significado | Valor inicial |
|---|---|---:|
| `population` | População total | 10.000 |
| `beta` | Intensidade de exposição ambiental | 0,035/dia |
| `sigma` | Progressão de exposto para sintomático | 1/10 por dia |
| `gamma` | Recuperação | 1/14 por dia |
| `k_saturation` | Saturação da força de exposição | 0,25 |
| `alpha` | Entrada basal de contaminação | 0,002/dia |
| `phi` | Entrada de contaminação durante a enchente | 0,12/dia |
| `eta` | Remoção de contaminação | 0,08/dia |
| `flood_start` | Início da enchente | dia 5 |
| `flood_end` | Fim da enchente | dia 15 |

## Autômato celular

| Parâmetro | Significado | Valor inicial |
|---|---|---:|
| `grid_size` | Lado da grade | 60 células |
| `occupancy_rate` | Proporção de células ocupadas | 70% |
| `beta` | Exposição ambiental local | 0,30/dia |
| `sigma` | Progressão para sintomático | 1/10 por dia |
| `gamma` | Recuperação | 1/14 por dia |
| `diffusion` | Difusão espacial da contaminação | 0,18 |
| `contamination_threshold` | Limiar da área contaminada | 0,25 |
| `seed` | Semente pseudoaleatória | 42 |

Os valores são exploratórios. Para uma aplicação empírica, devem ser calibrados com dados observados.
