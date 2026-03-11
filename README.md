## Depression_Annotator_jury_LLMs_Laerty

Notebook para **classificação de severidade de depressão** (0–3) em textos do Reddit, usando **múltiplos LLMs via OpenRouter** e **DSPy** para compilação de programas (prompt/program search).

### O que este notebook faz

- **Configura 7 modelos no OpenRouter** com `dspy` + `litellm`:
  - Premium: `gpt-4o-mini`, `claude-3.5-haiku`, `qwen3-30b`, `mimo-v2-flash`.
  - Free: `nemotron-nano-12b-v2-vl`, `deepseek-r1-0528`, `arcee-trinity-large-preview`.
- Implementa um wrapper (`NemotronPercentCleaningLM`) para limpar `%` do JSON do Nemotron.
- Define a **Signature DSPy** `ClassifyDepression` com:
  - `text` (entrada),
  - `probability` (distribuição em JSON 0–1),
  - `rationale` (raciocínio clínico em 1 sentença),
  - `severity` (rótulo 0, 1, 2 ou 3).
- Cria um **predictor Chain-of-Thought** para classificar textos.

### Dados e baseline

- Usa o arquivo **`gold_standard_depseverity.csv`** com textos e rótulos `severity` (0–3).
- Seleciona 20 amostras com melhor desempenho zero-shot do **GPT-4o-mini** e registra em `gold_df_test`.
- Calcula métricas de baseline (sem few-shot/compilação):
  - Matriz de confusão, Accuracy, Precision/Recall/F1 (macro/weighted),
  - **Quadratic Weighted Kappa (QWK)** para avaliar coerência ordinal.

### DSPy: treino, dev e métricas internas

- Separa:
  - `restantes_df` → **train_set** (exemplos DSPy com `text` e `severity`),
  - `gold_df_teste` → **dev_set** 
- Define duas métricas para o otimizador:
  - `metric_dist_penalty`: penaliza erros mais distantes (ordinais).
  - `classification_metric`: accuracy simples (0/1) sobre `severity`.
- Configura `Evaluate` com `dev_set` e `classification_metric`.

### Compilação dos modelos do júri

- Para cada modelo (`gpt4o_mini`, `claude_haiku`, `qwen3_30b`, `mimo_v2_flash`, `nemotron_12b`, `deepseek_r1`, `arcee_trinity`), o notebook:
  - Executa **BFRS** (BootstrapFewShotWithRandomSearch) do DSPy,
  - Amostra programas, avalia no `dev_set`,
  - Mostra evolução da métrica (ex.: 50%, 60%, 75%),
  - Registra melhor seed e confirma a compilação do modelo.

### Como reproduzir

1. Ter Python 3.10+ e instalar dependências (p. ex.: `dspy-ai`, `litellm`, `pandas`, `scikit-learn`, `jupyter`).
2. Criar `secrets.json` com `OPENROUTER_KEY` (ou setar variável de ambiente).
3. Abrir `Depression_Annotator_jury_LLMs_Laerty copy.ipynb` e executar:
   - Configuração/OpenRouter + modelos,
   - Definição da Signature e teste rápido,
   - Carregamento do `gold_standard_depseverity.csv` e métricas do baseline,
   - Criação de `train_set` / `dev_set`,
   - Compilação DSPy para cada modelo do júri.
  
### Processo de anotação com o júri de LLMs

Depois de compilados pelo DSPy, os modelos do júri são usados em um pipeline de anotação em duas camadas:

- **1. Anotador principal (GPT-4o-mini compilado)**  
  - O programa compilado do `gpt-4o-mini` é usado como **anotador de referência**.  
  - Para cada texto, ele produz:
    - rótulo de severidade (`0–3`),
    - distribuição de probabilidade por classe (JSON),
    - rationale clínico em linguagem natural.

- **2. Júri de LLMs como validadores dos rótulos**  
  - Os demais modelos compilados (`claude_haiku`, `qwen3_30b`, `mimo_v2_flash`, `nemotron_12b`, `deepseek_r1`, `arcee_trinity`) anotam **independentemente** o mesmo texto.  
  - Para cada texto, obtemos um vetor de rótulos de tamanho `N_modelos` (incluindo ou não o anotador principal, conforme o experimento).  
  - Os 5 melhores modelos (`claude_haiku`, `mimo_v2_flash`, `nemotron_12b`, `arcee_trinity`) são avaliado numa amostra de 100 textos e formam um júri para validar os rótulos do GPT-4o-mini, medindo:
    - concordância exata (mesma classe 0–3),
    - erros de distância 1 (por exemplo, 1 vs 2) e >1 (mais graves), usando a métrica de distância definida no notebook.

### Percentual de consenso do júri

Para cada texto anotado pelo júri:

- **Definição de consenso**  
  - Seja `k` o número de modelos no júri e `c` o número de modelos que escolheram a **mesma classe**.  
  - O **percentual de consenso** é calculado como  
    \[
    \text{consenso} = \frac{c}{k} \times 100\%
    \]
- **Consenso obtido:**:
  - 96,8% em média
  - 99% na classe ausente
  - 100% na classe severa


### Validação estatística da amostra de 500 textos (vs. dataset de 50k)

Para garantir que a amostra anotada pelo júri (500 textos) seja representativa do dataset completo de 50k textos (`DepreRedditBR_parallel_50k.csv`), o notebook realiza testes estatísticos:

**Resultado do Teste de Hipótese**:

- Proporção observada: 96.80%
- Meta proposta: 95%
- Estatística Z: 1.8468
- p-value: 0.0324
- Conclusão: A amostra ATINGE a meta de acurácia com confiança estatística.


**Resultado do Teste Qui-Quadrado**:

- Estatística Qui2: 7.2411
- p-value: 0.0646
- Distribuição das classes: equilibrada
- Conclusão: Amostra considerada REPRESENTATIVA.

Em conjunto, esse notebook demonstra o processo de:
1. anotação com programas compilados,  
2. validação via júri de LLMs (percentual de consenso) e  
3. validação estatística da amostra de 500 textos frente ao dataset de 50k  

fornece uma base robusta para confiar nos rótulos gerados pelo GPT-4o-mini + júri, tanto em termos de qualidade quanto de representatividade da amostra analisada.
