# Modelagem espaço-temporal da leptospirose

Projeto acadêmico que combina:

1. modelo compartimental SEIR com reservatório ambiental;
2. autômato celular bidimensional;
3. cinco cenários de enchente e intervenção;
4. dimensão de box-counting;
5. geração automática de métricas, CSVs e figuras.

## Estrutura

```text
leptospirose-seir-automato/
├── run_all.py
├── requirements.txt
├── README.md
├── src/
├── tests/
├── prompts/
└── results/
```

## Instalação

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux ou macOS:

```bash
source .venv/bin/activate
```

```bash
python -m pip install -r requirements.txt
```

## Execução completa

```bash
python run_all.py
```

Apenas o modelo compartimental:

```bash
python run_all.py --model ode
```

Apenas o autômato celular:

```bash
python run_all.py --model ca
```

## Testes

```bash
python -m unittest discover -s tests -v
```

## Cenários

- `referencia`
- `enchente_intensa`
- `protecao_individual`
- `descontaminacao`
- `intervencao_combinada`

## Arquivos gerados

- `metricas_ode.csv`
- `metricas_ca.csv`
- `resumo_resultados.txt`
- curvas epidemiológicas
- gráficos comparativos
- mapas dos estados epidemiológicos
- mapas de contaminação
- gráfico de box-counting

## Reprodutibilidade

O autômato celular usa uma semente pseudoaleatória fixa. Os mesmos parâmetros produzem os mesmos resultados.

## Limitação

Os parâmetros são exploratórios e não foram calibrados com dados epidemiológicos de uma localidade específica. Os resultados não devem ser interpretados como previsões oficiais.
